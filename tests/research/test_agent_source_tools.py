"""Private per-call profiles, actual MCP processes, and cached review replay."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import stat
import sys

import pytest

from src.review_backend import AgentReviewError, run_fresh_agent_json
from src.research.agent_source_tools import verify_source_tools
from src.research.config import load_config
from src.research.harness import _author_settings, _reviewer_settings
from src.research.novelty_review import (
    InvalidReviewResume, draft_research_agenda, review_novelty, source_packet,
)
from tests.research.test_novelty_gate import study as _study_fixture
from tests.research.test_unified_research import make_task

FAKE = r'''
import json, os, subprocess, sys, tomllib, uuid
from pathlib import Path
assert sys.argv[sys.argv.index("--allow") + 1] == "read-only"
assert "--session" not in sys.argv
profile_name = sys.argv[sys.argv.index("--profile") + 1]
codex_home = Path(os.environ["CODEX_HOME"])
profile = tomllib.loads((codex_home / (profile_name + ".config.toml")).read_text())
assert (codex_home / "auth.json").read_text() == '{"fixture_token":"private-fixture-token"}'
catalog = Path(profile["model_catalog_json"])
assert catalog.is_relative_to(codex_home) and catalog.read_text() == '{"models":[]}'
spec = profile["mcp_servers"]["agsci_sources"]
server = subprocess.Popen([spec["command"], *spec["args"]], cwd=spec["cwd"],
                          stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          text=True, encoding="utf-8")
def rpc(i, method, params=None):
    server.stdin.write(json.dumps({"jsonrpc":"2.0","id":i,"method":method,"params":params or {}}) + "\n")
    server.stdin.flush()
    result = json.loads(server.stdout.readline())
    assert "error" not in result, result
    return result["result"]
def tool(i, name, args):
    result = rpc(i, "tools/call", {"name":name,"arguments":args})
    assert result["isError"] is False, result
    return json.loads(result["content"][0]["text"])
rpc(1, "initialize", {"protocolVersion":"2025-06-18","capabilities":{},
                      "clientInfo":{"name":"fixture-client","version":"1"}})
rpc(2, "tools/list")
listed = tool(3, "list_sources", {})
sid = listed["sources"][0]["id"]
span = tool(4, "read_source_span", {"source_id":sid,"start":0,"end":3})
located = tool(5, "locate_source_quote", {"source_id":sid,"quote":span["quote"]})
tool(6, "find_source_passages", {"source_id":sid,"query":span["quote"]})
server.stdin.close()
server.wait(timeout=10)
assert server.returncode == 0, server.stderr.read()
result = json.loads(__MESSAGE_JSON__) if __MESSAGE_JSON__ else {
    "home":str(Path(os.environ["HOME"])), "codex_home":str(codex_home),
    "catalog":str(catalog), "quote":located["matches"][0],
}
prompt = Path(sys.argv[sys.argv.index("--prompt-file") + 1]).read_text()
if __REFLECT__ and "PREVIOUS AGENDA JSON" in prompt:
    result["_reflection"] = {"done":True,"changes":"","remaining_gaps":[]}
print(json.dumps({"protocolVersion":1,"type":"result","exitCode":0,
                  "data":{"finalMessage":json.dumps(result),"model":sys.argv[sys.argv.index("--model")+1],
                          "provider":"fixture-provider","nativeSessionId":str(uuid.uuid4())}}))
'''


@pytest.fixture(name="study")
def source_study(tmp_path):
    return _study_fixture.__wrapped__(tmp_path)


def environment(tmp_path: Path, message: dict | None = None, *, reflect: bool = False) -> dict[str, str]:
    original = tmp_path / "original-codex"
    original.mkdir()
    (original / "fixture.config.toml").write_text(
        'model_provider = "fixture"\nmodel_catalog_json = "catalog.json"\nmcp_servers = {}\n'
        '[model_providers.fixture]\nbase_url = "http://127.0.0.1:12345/v1"\n',
    )
    (original / "catalog.json").write_text('{"models":[]}')
    (original / "auth.json").write_text('{"fixture_token":"private-fixture-token"}')
    executable = tmp_path / "headless"
    body = FAKE.replace("__MESSAGE_JSON__", repr(json.dumps(message) if message is not None else ""))
    body = body.replace("__REFLECT__", repr(reflect))
    executable.write_text(f"#!{sys.executable}\n" + body)
    executable.chmod(0o700)
    return {"HEADLESS_BIN": str(executable), "HOME": str(tmp_path), "CODEX_HOME": str(original),
            "PATH": os.environ["PATH"], "SECRET_UNRELATED": "do-not-copy"}


def small_packet() -> dict:
    text = "α🙂\nA\x00B\r\nMethod and support."
    return {"sources": [{"id": "primary", "title": "Primary text", "availability": "full_text",
                         "text_sha256": hashlib.sha256(text.encode()).hexdigest(), "full_text": text}]}


def test_real_agent_transport_stages_distinct_profiles_and_replays_actual_tools(tmp_path: Path) -> None:
    env = environment(tmp_path)
    originals = {p.name: p.read_bytes() for p in Path(env["CODEX_HOME"]).iterdir()}
    packet = small_packet()
    results = []
    for i in range(2):
        artifacts = tmp_path / f"call-{i}"
        payload, receipt = run_fresh_agent_json(
            "Read frozen sources.", work_dir=artifacts, agent="codex", model="codex/fixture",
            profile="fixture", environment=env, source_lookup_packet=packet, timeout_seconds=15,
        )
        binding = verify_source_tools(artifacts, receipt["source_lookup"], expected_packet=packet)
        assert binding["tool_calls"] == 4 and binding["tool_errors"] == 0
        assert binding["tool_catalog_delivered"] is True
        assert not Path(payload["home"]).exists()
        assert not Path(payload["codex_home"]).exists()
        assert "auth.json" not in [p.name for p in artifacts.rglob("*")]
        for p in (artifacts / "source_lookup").iterdir():
            assert stat.S_IMODE(p.stat().st_mode) == 0o600
            assert b"private-fixture-token" not in p.read_bytes()
        results.append((payload, receipt))
    assert results[0][0]["codex_home"] != results[1][0]["codex_home"]
    assert results[0][1]["native_session_id"] != results[1][1]["native_session_id"]
    assert originals == {p.name: p.read_bytes() for p in Path(env["CODEX_HOME"]).iterdir()}


@pytest.mark.parametrize("tamper", ["packet", "server", "audit", "binding", "expected_packet"])
def test_source_evidence_tampering_is_rejected(tmp_path: Path, tamper: str) -> None:
    env = environment(tmp_path)
    packet = small_packet()
    artifacts = tmp_path / "call"
    _, receipt = run_fresh_agent_json(
        "Read frozen sources.", work_dir=artifacts, model="codex/fixture", profile="fixture",
        environment=env, source_lookup_packet=packet, timeout_seconds=15,
    )
    names = {"packet": "source-packet.json", "server": "server.py",
             "audit": "calls.jsonl", "binding": "binding.json"}
    if tamper in names:
        p = artifacts / "source_lookup" / names[tamper]
        p.write_bytes(p.read_bytes() + b"\n")
    else:
        packet = copy.deepcopy(packet)
        packet["sources"][0]["title"] = "A different supplied packet"
    with pytest.raises(ValueError):
        verify_source_tools(artifacts, receipt["source_lookup"], expected_packet=packet)


def test_native_review_cache_checks_source_tool_evidence_before_any_new_call(study, tmp_path, monkeypatch) -> None:
    task, cfg, agenda, review = study
    env_dir = tmp_path / "native-env"
    env_dir.mkdir()
    env = environment(env_dir, review)
    settings = {"agent": "codex", "model": "codex/fixture", "profile": "fixture",
                "environment": env, "timeout_seconds": 15, "source_lookup": True}
    output = tmp_path / "reviews"
    assert review_novelty(task, agenda, cfg, output, reviewer_settings=settings)["status"] == "supported"
    saved = [(output / f"reviewer_{i}/receipt.json").read_bytes() for i in (1, 2)]
    for i in (1, 2):
        receipt = json.loads(saved[i - 1])
        assert receipt["source_lookup_requested"] is True
        verify_source_tools(output / f"reviewer_{i}/transport", receipt["source_lookup"],
                            expected_packet=source_packet(task))

    def no_new_call(*args, **kwargs):
        pytest.fail("A completed reviewer must not be rerolled")
    monkeypatch.setattr("src.review_backend.run_fresh_agent_json", no_new_call)
    assert review_novelty(task, agenda, cfg, output, reviewer_settings=settings, resume=True)["status"] == "supported"
    assert saved == [(output / f"reviewer_{i}/receipt.json").read_bytes() for i in (1, 2)]
    audit = output / "reviewer_2/transport/source_lookup/calls.jsonl"
    audit.write_bytes(audit.read_bytes() + b"\n")
    with pytest.raises(InvalidReviewResume):
        review_novelty(task, agenda, cfg, output, reviewer_settings=settings, resume=True)


def test_author_and_v1_reflection_use_the_same_full_frozen_packet(study, tmp_path, monkeypatch) -> None:
    task, cfg, agenda, _ = study
    env_dir = tmp_path / "native-env"
    env_dir.mkdir()
    env = environment(env_dir, agenda, reflect=True)
    settings = {"agent": "codex", "model": "codex/fixture", "profile": "fixture",
                "environment": env, "timeout_seconds": 15, "source_lookup": True}
    output = tmp_path / "authors"
    output.mkdir()
    assert draft_research_agenda(task, cfg, output, reviewer_settings=settings, num_reflections=2) == agenda
    sessions = []
    for name in ("agenda_draft", "agenda_reflection_2"):
        receipt = json.loads((output / name / "receipt.json").read_text())
        sessions.append(receipt["native_session_id"])
        binding = verify_source_tools(output / name / "transport", receipt["source_lookup"],
                                      expected_packet=source_packet(task))
        assert binding["source_ids"] == ["prior"]
    assert len(set(sessions)) == 2
    monkeypatch.setattr("src.review_backend.run_fresh_agent_json", lambda *a, **kw: pytest.fail("cached author reissued"))
    assert draft_research_agenda(task, cfg, output, reviewer_settings=settings, num_reflections=2, resume=True) == agenda


def test_source_tools_reject_non_codex_or_missing_profile_before_agent_start(tmp_path: Path) -> None:
    env = environment(tmp_path)
    for i, (agent, profile) in enumerate((("claude", "fixture"), ("codex", None))):
        with pytest.raises(AgentReviewError, match="(?i)codex"):
            run_fresh_agent_json(
                "Read frozen sources.", work_dir=tmp_path / f"invalid-{i}", agent=agent,
                model="fixture", profile=profile, environment=env,
                source_lookup_packet=small_packet(), timeout_seconds=5,
            )


def test_harness_source_tools_are_opt_in_and_preserve_author_reviewer_models(tmp_path: Path) -> None:
    from types import SimpleNamespace
    cfg = SimpleNamespace(
        agent=SimpleNamespace(name="codex", model="author-model", profile="author", reasoning_effort="low"),
        review=SimpleNamespace(backend="agent", agent_name=None, agent_model="review-model",
                              agent_profile="reviewer", agent_reasoning_effort="high", timeout_seconds=60),
    )
    assert "source_lookup" not in _author_settings(cfg)
    assert "source_lookup" not in _reviewer_settings(cfg)
    assert _author_settings(cfg, source_lookup=True)["model"] == "author-model"
    assert _reviewer_settings(cfg, source_lookup=True)["model"] == "review-model"
    assert _author_settings(cfg, source_lookup=True)["source_lookup"] is True
    assert _reviewer_settings(cfg, source_lookup=True)["source_lookup"] is True
    task = make_task(tmp_path / "task")
    text = (task / "research.yaml").read_text()
    (task / "research.yaml").write_text(text + "\nharness:\n  agent_source_lookup: true\n")
    assert load_config(task).harness.agent_source_lookup is True
    (task / "research.yaml").write_text(text + '\nharness:\n  agent_source_lookup: "true"\n')
    with pytest.raises(ValueError, match="must be boolean"):
        load_config(task)
