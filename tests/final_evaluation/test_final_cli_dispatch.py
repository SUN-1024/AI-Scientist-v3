"""The command entry point preserves the frozen portfolio comparison boundary."""
from __future__ import annotations

import json

import pytest

from src.final_evaluation import cli
from src.final_evaluation.runner import FinalEvaluationError


def arguments(tmp_path):
    return ["--task-dir", str(tmp_path / "task"),
            "--investigation-results-dir", str(tmp_path / "investigation"),
            "--output-dir", str(tmp_path / "final"),
            "--python-bin", str(tmp_path / "python"),
            "--container-image", "sha256:fixture", "--timeout-seconds", "123"]


@pytest.mark.parametrize("planned", [False, True])
def test_cli_dispatches_planned_collection_or_legacy_once(tmp_path, monkeypatch, capsys, planned):
    calls = []
    monkeypatch.setattr(cli, "portfolio_final_set_required", lambda **_: planned)
    def call(kind, values):
        calls.append((kind, values))
        return {"kind": kind}
    monkeypatch.setattr(cli, "run_final_evaluation", lambda **kw: call("legacy", kw))
    monkeypatch.setattr(cli, "run_portfolio_final_evaluation", lambda **kw: call("collection", kw))
    assert cli.main(arguments(tmp_path)) == 0
    expected = "collection" if planned else "legacy"
    assert len(calls) == 1 and calls[0][0] == expected
    assert calls[0][1]["timeout_seconds"] == 123
    assert calls[0][1]["output_dir"] == tmp_path / "final"
    assert json.loads(capsys.readouterr().out) == {"kind": expected}


def test_invalid_collection_does_not_fall_back_to_single_winner(tmp_path, monkeypatch, capsys):
    def required(**_):
        raise FinalEvaluationError("planned method set is not valid")
    monkeypatch.setattr(cli, "portfolio_final_set_required", required)
    monkeypatch.setattr(cli, "run_final_evaluation", lambda **_: pytest.fail("legacy evaluation must not start"))
    monkeypatch.setattr(cli, "run_portfolio_final_evaluation", lambda **_: pytest.fail("invalid collection must not start"))
    assert cli.main(arguments(tmp_path)) == 1
    assert "planned method set is not valid" in capsys.readouterr().err
