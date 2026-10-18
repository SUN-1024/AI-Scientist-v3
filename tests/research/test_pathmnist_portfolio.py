"""Synthetic CPU fixtures for the GPU portfolio boundary; no real data or GPU jobs."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

from examples.pathmnist_mil_portfolio import configuration as cfg
from examples.pathmnist_mil_portfolio import evaluator as ev
from examples.pathmnist_mil_portfolio import prepare as prep
from src.investigation.evaluation_evidence import load_evaluation_evidence
from src.investigation.evaluator_outputs import validate_evaluator_outputs
from src.investigation.task_contract import load_task_contract, parse_task_contract
from src.investigation.task_runtime import _requirements_match_contract, _validate_requirements
from src.research.config import load_config

ROOT = Path(__file__).resolve().parents[2] / "examples/pathmnist_mil_portfolio"


def configuration(**updates):
    return dict(ev.p.DEFAULT_CONFIGURATION, **updates)


def device_report(**updates):
    report = dict(device_type="cuda", cuda_available=True, device_count=1, logical_device_index=0,
                  device_name="Fixture GPU", device_uuid="GPU-fixture", torch_version="2.11.0+cu126",
                  cuda_version="12.6")
    return dict(report, **updates)


def arrays():
    rng = np.random.default_rng(100)
    result = {}
    for split, size in (("train", 64), ("val", 8), ("test", 8)):
        result[split + "_x"] = rng.normal(size=(size, 16 if split == "train" else 64, 4)).astype(np.float32)
        result[split + "_y"] = np.tile([0, 1], size // 2).astype(np.float32)
    return result


def fake_handle(config=None):
    config = configuration() if config is None else config
    requests = []

    def handle(request):
        requests.append(copy.deepcopy(request))
        if request["op"] == "get_configuration":
            return {"configuration": copy.deepcopy(config)}
        if request["op"] == "init":
            assert "val_y" not in request and "test_y" not in request
            return {"parameters": 30, "trainable_parameters": 25, "device": device_report()}
        if request["op"] == "train":
            return {"steps": len(request["batch_indices"])}
        if request["op"] == "predict":
            assert set(request) == {"op", "x"}
            return {"logits": np.asarray(request["x"]).mean(axis=(1, 2)).tolist()}
        return {}

    return handle, requests


@pytest.mark.parametrize("key,value", [
    ("learning_rate", 0), ("learning_rate", float("inf")), ("learning_rate", float("nan")),
    ("batch_size", True), ("batch_size", 32.0), ("batch_size", 128),
    ("hidden_dim", 33), ("weight_decay", -0.001), ("dropout", 0.6),
    ("pooling_temperature", 0), ("gradient_clip_norm", 11), ("hybrid_mix", 1.1),
    ("learning_rate", "0.003"),
])
def test_rejects_illegal_joint_hpo_parameters(key, value):
    with pytest.raises(ValueError, match="configuration"):
        cfg.validate_configuration(configuration(**{key: value}))


def test_configuration_contract_requires_exact_keys_and_rpc_envelope():
    assert cfg.configuration_from_response({"configuration": configuration()}) == configuration()
    for bad in (dict(configuration(), max_steps=1), {"learning_rate": 0.003}, None):
        with pytest.raises(ValueError, match="exactly"):
            cfg.validate_configuration(bad)
    with pytest.raises(ValueError, match="only configuration"):
        cfg.configuration_from_response({"configuration": configuration(), "seed": 19})


@pytest.mark.parametrize("updates", [
    {"device_count": 2}, {"device_count": True}, {"logical_device_index": 1},
    {"device_type": "cpu"}, {"cuda_available": False}, {"device_uuid": ""},
])
def test_gpu_report_rejects_multiple_devices_cpu_fallback_and_missing_identity(updates):
    with pytest.raises(ValueError):
        cfg.validate_device_report(device_report(**updates))


def test_gpu_report_checks_frozen_model_and_runtime():
    report = device_report()
    assert cfg.validate_device_report(report, expected_model="Fixture GPU", expected_torch="2.11.0+cu126") == report
    for expected in ({"expected_model": "Another GPU"}, {"expected_torch": "2.11.0+cpu"}):
        with pytest.raises(ValueError, match="differs"):
            cfg.validate_device_report(report, **expected)


def test_host_keeps_budget_and_selection_fixed_while_joint_hpo_changes_example_cost(monkeypatch):
    monkeypatch.setattr(ev.p, "MAX_STEPS", 4)
    monkeypatch.setattr(ev.p, "CHECK_EVERY", 2)
    data = arrays()
    records, transcripts = [], []
    for batch_size in (16, 64):
        handle, requests = fake_handle(configuration(batch_size=batch_size))
        record = ev.train_condition(handle, data, seed=17, noise_rate=0.15, configuration=None,
                                    expected_model="Fixture GPU", expected_torch="2.11.0+cu126")
        assert [request["op"] for request in requests[:3]] == ["reset", "get_configuration", "init"]
        assert record["steps_run"] == 4 and record["selected_step"] == 2
        assert record["host_training_examples"] == 4 * batch_size
        assert [item["step"] for item in record["selection_trace"]] == [2, 4]
        assert record["flipped_training_bags"] == round(64 * 0.15)
        assert record["host_train_rpc_seconds"] >= 0
        assert record["seconds"] >= record["host_train_rpc_seconds"]
        assert record["reported_parameters"] == 30
        assert record["reported_trainable_parameters"] == 25
        assert all("val_y" not in request and "test_y" not in request for request in requests)
        records.append(record)
        transcripts.append(requests)
    assert records[0]["configuration_sha256"] != records[1]["configuration_sha256"]
    assert transcripts[0][2]["train_y"] == transcripts[1][2]["train_y"]
    assert records[0]["selection_trace"][0]["worst_auc"] == records[1]["selection_trace"][0]["worst_auc"]


def test_configuration_change_is_rejected_before_any_training():
    handle, requests = fake_handle(configuration(batch_size=64))
    with pytest.raises(ValueError, match="changed"):
        ev.train_condition(handle, arrays(), seed=29, noise_rate=0, configuration=configuration())
    assert [request["op"] for request in requests] == ["reset", "get_configuration"]


def test_invalid_configuration_never_reaches_init():
    handle, requests = fake_handle(configuration(max_steps=1))
    with pytest.raises(ValueError, match="exactly"):
        ev.train_condition(handle, arrays(), seed=17, noise_rate=0, configuration=None)
    assert [request["op"] for request in requests] == ["reset", "get_configuration"]


def test_reproducible_host_minibatches_and_training_labels(monkeypatch):
    monkeypatch.setattr(ev.p, "MAX_STEPS", 2)
    monkeypatch.setattr(ev.p, "CHECK_EVERY", 2)
    selected = []
    for _ in range(2):
        handle, requests = fake_handle()
        ev.train_condition(handle, arrays(), seed=29, noise_rate=0.15, configuration=None)
        selected.append([request for request in requests if request["op"] in {"train", "init"}])
    assert selected[0] == selected[1]


def test_seed_model_is_permutation_invariant_and_capacity_bounded_on_cpu():
    torch = pytest.importorskip("torch")
    spec = importlib.util.spec_from_file_location("portfolio_cpu_fixture", ROOT / "program_0/src/program.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.handle({"op": "get_configuration"}) == {"configuration": configuration()}
    with pytest.raises(ValueError, match="accepts no data"):
        module.handle({"op": "get_configuration", "x": []})
    bags = torch.randn(3, 16, 48)
    for width in (16, 32, 64, 128):
        model = module.MIL(48, configuration(hidden_dim=width, dropout=0.5)).eval()
        assert sum(parameter.numel() for parameter in model.parameters()) <= ev.p.MAX_PARAMETERS
        with torch.no_grad():
            assert tuple(model(bags).shape) == (3,)
            torch.testing.assert_close(model(bags), model(bags[:, torch.randperm(16)]))


def test_candidate_disallows_cpu_fallback(monkeypatch):
    torch = pytest.importorskip("torch")
    spec = importlib.util.spec_from_file_location("portfolio_device_fixture", ROOT / "program_0/src/program.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="CPU fallback is disabled"):
        module.assigned_device()


def test_template_contract_and_quality_metrics_are_frozen_and_separate():
    raw = json.loads((ROOT / "task_contract.json").read_text())
    contract = parse_task_contract(raw)
    assert contract.canonical_task.sha256 == hashlib.sha256((ROOT / "task.md").read_bytes()).hexdigest()
    assert contract.compute.search.evaluator_gpu_count == 0
    assert contract.compute.search.candidate_gpu_count == 1
    assert contract.protocol.search_reset_sequence == (17, 17, 29, 29)
    assert tuple(metric.name for metric in contract.metrics.search) == ev.METRIC_NAMES
    assert _validate_requirements(ROOT / "requirements.txt")
    assert _requirements_match_contract(ROOT / "requirements.txt", contract)
    config = load_config(ROOT)
    assert config.methodology.profile == "medical"
    assert config.whole_idea_policy.enabled is True
    assert config.harness.quality.require_ablation is False
    assert "candidate_train_rpc_seconds" not in contract.protocol.promotion_rule


@pytest.mark.parametrize("torch_version,model,index", [
    ("2.11.0+cpu", "GPU", prep.VERIFIED_CUDA_INDEX),
    ("2.11.0", "GPU", prep.VERIFIED_CUDA_INDEX),
    ("2.11.0+cu126", "UNCONFIGURED", prep.VERIFIED_CUDA_INDEX),
    ("2.11.0+cu126", "GPU", "https://download.pytorch.org/whl/cu130"),
    ("2.11.0+cu126", "GPU", "https://user:secret@example.org/cu126"),
])
def test_prepare_requires_explicit_consistent_runtime_selection(torch_version, model, index):
    with pytest.raises(ValueError):
        prep.validate_runtime_selection(torch_version, model, index)


def test_prepare_accepts_the_verified_cuda_lock_without_preparing_data():
    assert prep.validate_runtime_selection("2.11.0+cu126", "NVIDIA GeForce RTX 4080 SUPER") == (
        "2.11.0+cu126", "NVIDIA GeForce RTX 4080 SUPER")


def test_preparation_overlay_reuses_builder_and_freezes_standalone_helpers(tmp_path, monkeypatch):
    """The data-preparation function is mocked; no archive or real resource is read."""
    helper = prep.existing_prepare()
    calls = []

    def fake_data_prepare(archive, task, **options):
        calls.append((archive, dict(options)))
        shutil.copytree(prep.EXISTING_TEMPLATE, task, ignore=shutil.ignore_patterns("__pycache__", "resources"))
        for plane in ("search", "final"):
            directory = task / "resources" / plane
            directory.mkdir(parents=True)
            (directory / "data.npz").write_bytes(b"synthetic hash fixture; not a dataset")
        (task / "resources/final/construction.npz").write_bytes(b"synthetic audit fixture")
        (task / "data_manifest.json").write_text("{}")
        return {"task": str(task), "validation_bags": 8, "test_bags": 8}

    monkeypatch.setattr(helper, "prepare", fake_data_prepare)
    monkeypatch.setattr(prep, "existing_prepare", lambda: helper)
    task = tmp_path / "generated"
    archive = tmp_path / "never-read.npz"
    report = prep.prepare(archive, task, cuda_torch_version="2.11.0+cu126", accelerator_model="Fixture GPU")
    assert calls == [(archive, {"validation_bags": 56, "test_bags": 32})]
    assert report["final_evaluation_performed"] is False
    contract = load_task_contract(task / "task_contract.json")
    assert contract.compute.search.accelerator_model == "Fixture GPU"
    assert _requirements_match_contract(task / "requirements.txt", contract)
    setup = json.loads((task / "runtime_setup.json").read_text())
    assert setup["required_image_environment"] == {"PIP_EXTRA_INDEX_URL": prep.VERIFIED_CUDA_INDEX}
    assert setup["cuda_execution_verified_during_preparation"] is False
    for name in ("evaluator.py", "protocol.py"):
        assert (task / "shared_helpers" / name).read_bytes() == (prep.EXISTING_TEMPLATE / name).read_bytes()
    assert not (task / "prepare_data.py").exists()
    probe = subprocess.run([sys.executable, "-c",
                            "import evaluator; assert evaluator.p.MAX_STEPS == 160; "
                            "assert evaluator.shared.p.MAX_STEPS == 640"], cwd=task,
                           text=True, capture_output=True, timeout=30)
    assert probe.returncode == 0, probe.stderr


def test_evaluator_rejects_direct_or_untrusted_proxy_before_data_access(tmp_path, monkeypatch):
    monkeypatch.delenv("AGENTIC_CANDIDATE_SOCKET", raising=False)
    monkeypatch.setattr(ev.shared, "read_data", lambda *_: pytest.fail("data must remain unread"))
    with pytest.raises(ValueError, match="sealed Harness"):
        ev.evaluate(ROOT / "program_0/src/program.py", tmp_path, plane="search")
    monkeypatch.setenv("AGENTIC_CANDIDATE_SOCKET", "/fixture.sock")
    with pytest.raises(ValueError, match="sealed Harness"):
        ev.evaluate("/untrusted/candidate_proxy.py", tmp_path, plane="final")


@pytest.mark.parametrize("plane,smoke", [("search", True), ("search", False), ("final", False)])
def test_full_host_output_shape_on_mock_rpc_and_synthetic_arrays(tmp_path, monkeypatch, plane, smoke):
    raw = json.loads((ROOT / "task_contract.json").read_text())
    raw["compute"][plane]["accelerator_model"] = "Fixture GPU"
    (tmp_path / "task_contract.json").write_text(json.dumps(raw))
    data = arrays()
    monkeypatch.setenv("TASK_SOURCE_DIR", str(tmp_path))
    monkeypatch.setenv("AGENTIC_CANDIDATE_SOCKET", "/synthetic-test.sock")
    monkeypatch.setattr(ev.shared, "read_data", lambda _root, selected_plane: {
        key: value for key, value in data.items() if selected_plane == "final" or not key.startswith("test_")})
    monkeypatch.setattr(ev.p, "MAX_STEPS", 2)
    monkeypatch.setattr(ev.p, "CHECK_EVERY", 2)
    monkeypatch.setattr(ev.shared, "paired_bootstrap", lambda *_: {"estimate": 0.0, "ci95": [-0.1, 0.1]})
    candidate, transcript = fake_handle()
    reference, _ = fake_handle()
    monkeypatch.setattr(ev.shared, "load_handle", lambda path, *_: candidate if str(path) == "/broker-code/candidate_proxy.py" else reference)
    output = tmp_path / "out"
    metrics = ev.evaluate("/broker-code/candidate_proxy.py", output, plane=plane, smoke=smoke)
    seeds = ev.p.SEARCH_SEEDS if plane == "search" else ev.p.FINAL_SEEDS
    expected_seeds = [seeds[0]] if smoke else [seed for seed in seeds for _ in ev.p.NOISE_RATES]
    assert transcript[0] == {"op": "reset", "seed": seeds[0]}
    assert [request["seed"] for request in transcript if request["op"] == "reset"] == expected_seeds
    assert set(metrics) == set(ev.METRIC_NAMES)
    assert validate_evaluator_outputs(output, ev.METRIC_NAMES) == (metrics, True)
    evidence = load_evaluation_evidence(output)
    solution = evidence["episodes"][0]["solution"]
    assert solution["smoke_only"] is smoke
    assert solution["evaluation_split"] == ("val" if plane == "search" else "test")
    assert solution["configuration"] == configuration()
    assert "source review required" in solution["capacity_and_device_status"]
    assert "not an equally tuned" in solution["reference_status"]
    assert "cross-device" in solution["timing_scope"]["comparability"]
    assert all("test_y" not in request and "val_y" not in request for request in transcript)
