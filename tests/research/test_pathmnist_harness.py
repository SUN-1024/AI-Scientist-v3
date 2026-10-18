"""Contracted CPU task: fresh sampling, candidate boundary and paired evidence."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from examples.pathmnist_mil.prepare_data import make_bags
from examples.pathmnist_mil_harness import evaluator as ev
from examples.pathmnist_mil_harness import prepare_data as prep
from src.investigation.task_contract import TaskContractError, load_task_contract, parse_task_contract
from src.investigation.evaluation_evidence import load_evaluation_evidence
from src.investigation.evaluator_outputs import validate_evaluator_outputs
from src.final_evaluation.runner import _validate_metrics

ROOT = Path(__file__).resolve().parents[2] / "examples/pathmnist_mil_harness"


def test_reconstructs_exact_old_sampler_without_opening_old_archive():
    labels = np.array([8] * 20 + [0] * 1800)
    ids = np.arange(len(labels)) + 10000
    features = ids[:, None].astype(np.float32)
    for length in (16, 64):
        previous = make_bags(features, labels, ids, 12, 29, extended=length == 64)
        orders, targets, witnesses = prep.bag_indices(labels, 12, 29, length)
        np.testing.assert_array_equal(ids[orders], previous[2])
        np.testing.assert_array_equal(targets, previous[1])
        np.testing.assert_array_equal(witnesses, previous[3])


def test_fresh_bags_exclude_used_patches_and_preserve_witnesses():
    labels = np.array([8] * 30 + [0] * 3000)
    used, _, _ = prep.bag_indices(labels, 12, 29, 64)
    fresh, targets, witnesses, capacity = prep.fresh_orders(labels, used, 1000, 829)
    assert len(fresh) == capacity["capacity_bags"]
    assert len(fresh) % 4 == 0
    assert not np.intersect1d(used, fresh).size
    assert len(np.unique(fresh)) == fresh.size
    for length in (16, 32, 64):
        actual = np.count_nonzero(labels[fresh[:, :length]] == 8, axis=1)
        np.testing.assert_array_equal(actual, witnesses)
        np.testing.assert_array_equal(actual > 0, targets.astype(bool))


def test_declared_contract_freezes_seeds_and_boundary():
    raw = json.loads((ROOT / "task_contract.json").read_text())
    contract = parse_task_contract(raw)
    assert contract.candidate.interface == "python_rpc_v1:handle"
    assert contract.candidate.editable_paths == ("program_0/src",)
    assert contract.protocol.search_reset_sequence == (17, 17, 29, 29)
    assert contract.protocol.final_reset_sequence == (101, 101, 211, 211, 307, 307, 401, 401, 503, 503)
    assert not set(contract.protocol.search_seeds) & set(contract.protocol.final_seeds)
    assert hashlib.sha256((ROOT / "task.md").read_bytes()).hexdigest() == contract.canonical_task.sha256


def _arrays():
    rng = np.random.default_rng(4)
    arrays = {}
    for split in ("train", "val", "test"):
        arrays[split + "_x"] = rng.normal(size=(16, 16 if split == "train" else 64, 4)).astype(np.float32)
        arrays[split + "_y"] = np.tile([0, 1], 8).astype(np.float32)
    return arrays


def test_candidate_snapshot_reproducible_and_prediction_has_no_labels():
    handle = ev.load_handle(ROOT / "program_0/src/program.py")
    data = _arrays()
    request = {"op": "init", "train_x": data["train_x"].tolist(), "train_y": data["train_y"].tolist(),
               "learning_rate": 0.003, "weight_decay": 0.0001, "gradient_clip_norm": 5.0}
    initial = []
    for _ in range(2):
        handle({"op": "reset", "seed": 17})
        handle(request)
        initial.append(ev.predict(handle, data["val_x"]))
    np.testing.assert_array_equal(*initial)
    handle({"op": "snapshot", "name": "before"})
    handle({"op": "train", "batch_indices": [list(range(16))]})
    assert not np.array_equal(ev.predict(handle, data["val_x"]), initial[0])
    handle({"op": "restore", "name": "before"})
    np.testing.assert_array_equal(ev.predict(handle, data["val_x"]), initial[0])
    with pytest.raises(ValueError, match="features only"):
        handle({"op": "predict", "x": data["val_x"].tolist(), "labels": data["val_y"].tolist()})
    ev.score_views(handle, data["val_x"], data["val_y"], prediction_seed=701)


def test_paired_bootstrap_identical_models_has_zero_difference():
    rng = np.random.default_rng(5)
    predictions = rng.normal(size=(4, 3, 12))
    report = ev.paired_bootstrap(predictions, predictions, np.tile([0, 1], 6), 2, draws=30)
    assert report["estimate"] == 0
    assert report["ci95"] == [0, 0]


def test_final_labels_never_cross_candidate_boundary(tmp_path, monkeypatch):
    task = tmp_path / "task"
    shutil.copytree(ROOT, task, ignore=shutil.ignore_patterns("__pycache__"))
    data = _arrays()
    for plane in ("search", "final"):
        (task / "resources" / plane).mkdir(parents=True)
        payload = data if plane == "final" else {key: value for key, value in data.items() if not key.startswith("test_")}
        np.savez_compressed(task / "resources" / plane / "data.npz", **payload)
    np.savez_compressed(task / "resources/final/construction.npz", dummy=np.array([1]))
    manifest = {plane + "_sha256": prep.digest(task / "resources" / plane / "data.npz")
                for plane in ("search", "final")}
    (task / "data_manifest.json").write_text(json.dumps(manifest))
    prep.write_contract(task)
    load_task_contract(task / "task_contract.json")
    requests = []
    def handle(request):
        requests.append(request)
        if request["op"] == "init":
            assert set(request) == {"op", "train_x", "train_y", "learning_rate", "weight_decay", "gradient_clip_norm"}
            return {"parameters": 10}
        if request["op"] == "predict":
            assert set(request) == {"op", "x"}
            return {"logits": np.asarray(request["x"]).mean(axis=(1, 2)).tolist()}
        return {}
    monkeypatch.setenv("TASK_SOURCE_DIR", str(task))
    monkeypatch.setattr(ev, "load_handle", lambda *args: handle)
    monkeypatch.setattr(ev.p, "MAX_STEPS", 1)
    monkeypatch.setattr(ev.p, "CHECK_EVERY", 1)
    monkeypatch.setattr(ev, "paired_bootstrap", lambda *args: {"estimate": 0.0, "ci95": [-0.1, 0.1]})
    metrics = ev.evaluate(task / "program_0/src/program.py", tmp_path / "out", plane="final")
    assert [request["seed"] for request in requests if request["op"] == "reset"] == [s for s in ev.p.FINAL_SEEDS for _ in range(2)] * 2
    evidence = load_evaluation_evidence(tmp_path / "out")
    solution = evidence["episodes"][0]["solution"]
    assert solution["paired_predictions"]["labels"] == data["test_y"].tolist()
    assert solution["evaluation_split"] == "test"
    names = tuple(metric.name for metric in load_task_contract(task / "task_contract.json").metrics.final)
    assert _validate_metrics(tmp_path / "out", names) == (metrics, True)
    assert validate_evaluator_outputs(tmp_path / "out", names) == (metrics, True)
    assert all("test_y" not in request and "val_y" not in request for request in requests)
    assert "candidate_timeseries" not in evidence["episodes"][0]
    assert solution["selection_traces"]["candidate"]
    (task / "protocol.py").write_text((task / "protocol.py").read_text() + "\nMAX_STEPS = 999\n")
    with pytest.raises(TaskContractError, match="provenance artifact digest mismatch: protocol.py"):
        load_task_contract(task / "task_contract.json")


def test_rejects_position_dependent_predictions():
    data = _arrays()
    def handle(request):
        return {"logits": np.asarray(request["x"])[:, 0, 0].tolist()}
    with pytest.raises(ValueError, match="MIL interface violation"):
        ev.score_views(handle, data["val_x"], data["val_y"], prediction_seed=701)


def test_bootstrap_direction_and_ties():
    labels = np.tile([0, 1], 6)
    predictions = np.broadcast_to(labels, (4, 3, 12)).astype(float)
    report = ev.paired_bootstrap(predictions, -predictions, labels, 2, draws=30)
    assert report["estimate"] == 1
    assert report["ci95"] == [1, 1]
    assert ev.auc(labels, np.ones(12)) == 0.5


def test_prepare_writes_complete_frozen_contract_on_fixture(tmp_path, monkeypatch):
    rng = np.random.default_rng(7)
    arrays = {}
    for split, count in (("train", 160), ("val", 1100), ("test", 1100)):
        arrays[split + "_images"] = rng.integers(0, 256, (count, 28, 28, 3), dtype=np.uint8)
        arrays[split + "_labels"] = np.concatenate([np.full(20, 8), np.zeros(count - 20)]).astype(np.uint8)[:, None]
    archive = tmp_path / "official-fixture.npz"
    np.savez_compressed(archive, **arrays)
    monkeypatch.setattr(prep, "DATA_MD5", prep.digest(archive, "md5"))
    monkeypatch.setattr(prep, "OLD_BAGS", {"train": (8, 17, 16), "val": (8, 29, 64), "test": (8, 41, 64)})
    report = prep.prepare(archive, tmp_path / "task", validation_bags=8, test_bags=8)
    assert report["validation_bags"] == report["test_bags"] == 8
    load_task_contract(tmp_path / "task/task_contract.json")
    with np.load(tmp_path / "task/resources/final/construction.npz") as audit:
        for split in ("val", "test"):
            assert not np.intersect1d(audit[split + "_ids"], audit["previous_" + split + "_ids"]).size
    with pytest.raises(ValueError, match="already exists"):
        prep.prepare(archive, tmp_path / "task")


@pytest.mark.parametrize("count,construction_seed", [(32, 1009), (12, 1013)])
def test_rejects_public_label_order_spoof_at_actual_full_batch_sizes(count, construction_seed):
    """Regression for a perfect-label spoof that returned zeros on the old small probes."""
    labels = (np.arange(count) % 2)[np.random.default_rng(construction_seed).permutation(count)]
    known_old_order = np.random.default_rng(631).permutation(count)
    x = np.random.default_rng(19).normal(size=(count, 64, 4)).astype(np.float32)
    def spoof(request):
        size = len(request["x"])
        return {"logits": labels[known_old_order].tolist() if size == count else [0.0] * size}
    with pytest.raises(ValueError, match="MIL interface violation"):
        ev.score_views(spoof, x, labels, prediction_seed=903)


def test_checks_scored_full_batch_against_independent_probes():
    data = _arrays()
    def batch_size_shortcut(request):
        size = len(request["x"])
        return {"logits": [1.0 if size == len(data["val_y"]) else 0.0] * size}
    with pytest.raises(ValueError, match="scored full batch versus independent probe"):
        ev.score_views(batch_size_shortcut, data["val_x"], data["val_y"], prediction_seed=907)


def test_prediction_order_changes_but_scored_pointwise_values_do_not():
    data = _arrays()
    observed = []
    def honest(request):
        x = np.asarray(request["x"])
        observed.append(x)
        return {"logits": x.mean(axis=(1, 2)).tolist()}
    score_a, prediction_a = ev.score_views(honest, data["val_x"], data["val_y"], prediction_seed=911)
    first_order = observed[0].mean(axis=(1, 2))
    assert sum(len(value) == 1 for value in observed) >= len(data["val_y"]) * 3
    observed.clear()
    score_b, prediction_b = ev.score_views(honest, data["val_x"], data["val_y"], prediction_seed=919)
    second_order = observed[0].mean(axis=(1, 2))
    assert not np.array_equal(first_order, second_order)
    np.testing.assert_allclose(prediction_a, prediction_b)
    assert score_a["worst_auc"] == score_b["worst_auc"]
    assert score_a["prediction_seed"] != score_b["prediction_seed"]


def test_required_canonical_comparisons_have_exact_ids_kinds_and_budget():
    from src.research.config import load_config
    from src.variants_pipeline.config import VariantsExecutionConfig
    from src.variants_pipeline.generate import _parse_variant_bundle
    from src.variants_pipeline.select import _apply_hard_caps

    baseline_ids = ["baseline_attention", "baseline_lme", "baseline_mean",
                    "baseline_max", "baseline_top2"]
    ablation_id = "ablation_no_primary_mechanism"
    cfg = load_config(ROOT)
    assert cfg.harness.quality.required_comparisons == [*baseline_ids, "control_best", ablation_id]
    assert cfg.harness.quality.require_ablation is True
    task_text = (ROOT / "task.md").read_text()
    for variant_id in baseline_ids:
        assert f"| `{variant_id}` | `baseline` | `best` |" in task_text
    assert f"| `{ablation_id}` | `ablation` | `best` |" in task_text
    assert "| `control_best` | `control` | `best` |" in task_text
    assert "ablations.execution.max_extra_baselines=5" in task_text
    assert "already in the seed" in task_text
    raw = {"decision": "propose_variants", "candidates": [
        {"variant_id": variant_id, "title": variant_id, "kind": kind, "base_program": "best"}
        for variant_id, kind in [*((name, "baseline") for name in baseline_ids), (ablation_id, "ablation")]
    ]}
    _, candidates = _parse_variant_bundle(raw)
    defaults = VariantsExecutionConfig()
    assert defaults.max_ablations >= 1
    default_selection = _apply_hard_caps(candidates=candidates, llm_decisions={},
                                        max_ablations=defaults.max_ablations,
                                        max_extra_baselines=defaults.max_extra_baselines)
    assert len(default_selection.rejected) == 2
    configured = _apply_hard_caps(candidates=candidates, llm_decisions={},
                                 max_ablations=defaults.max_ablations, max_extra_baselines=5)
    assert not configured.rejected
    assert [item.variant_id for item in configured.accepted] == [*baseline_ids, ablation_id]
    assert configured.controls == ["control_best"]


def test_linux_runtime_lock_requires_cpu_torch_and_matches_contract():
    from packaging.specifiers import SpecifierSet
    from src.investigation.task_runtime import _requirements_match_contract, _validate_requirements

    requirements = ROOT / "requirements.txt"
    contract = parse_task_contract(json.loads((ROOT / "task_contract.json").read_text()))
    assert _validate_requirements(requirements)
    assert _requirements_match_contract(requirements, contract)
    torch = next(item for item in contract.provenance.dependencies if item.name == "torch")
    assert torch.version == "2.11.0+cpu"
    pin = SpecifierSet("==" + torch.version)
    assert "2.11.0+cpu" in pin
    assert "2.11.0" not in pin
    assert "2.11.0+cu130" not in pin
