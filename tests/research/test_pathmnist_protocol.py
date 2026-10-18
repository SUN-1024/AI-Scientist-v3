"""Focused regression tests for development evaluation, not paper-level validation."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
pytest.importorskip("sklearn")

from examples.pathmnist_mil import calibrate, evaluate
from examples.pathmnist_mil.protocol import EvaluationProtocol, load_protocol


def toy_data():
    rng = np.random.default_rng(33)
    train_x = torch.from_numpy(rng.normal(size=(16, 16, 2)).astype(np.float32))
    val_x = torch.from_numpy(rng.normal(size=(8, 64, 2)).astype(np.float32))
    return train_x, np.array([0, 1] * 8, np.float32), val_x, np.array([0, 1] * 4, np.float32)


class MeanModel(torch.nn.Module):
    def forward(self, x):
        return x.mean(dim=(1, 2))


class CountModel(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.used = torch.nn.Linear(input_dim, 1)
        self.unused = torch.nn.Linear(input_dim, 1)
        self.connected_zero = torch.nn.Parameter(torch.zeros(()))
        self.frozen = torch.nn.Parameter(torch.zeros(()), requires_grad=False)

    def forward(self, x):
        return self.used(x.mean(dim=1)).squeeze(-1) + 0 * self.connected_zero + self.frozen


def search_archive(tmp_path):
    train_x, train_y, val_x, val_y = toy_data()
    path = tmp_path / "search.npz"
    np.savez(path, train_x=train_x.numpy(), train_y=train_y, val_x=val_x.numpy(), val_y=val_y)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_name("data_manifest.json").write_text(json.dumps({"search_sha256": digest}))
    return path


def toy_program(path):
    path.write_text('''import torch
from torch import nn
import numpy as np
class Model(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.layer = nn.Linear(dim, 1)
        with torch.no_grad():
            self.layer.bias.fill_(float(np.random.uniform(-1, 1)))
    def forward(self, x):
        return self.layer(x.mean(dim=1)).squeeze(-1)
def build_model(input_dim):
    return Model(input_dim)
''')
    return path


def test_protocol_defaults_and_canonical_hash(tmp_path):
    protocol = load_protocol()
    assert (protocol.max_steps, protocol.seeds, protocol.patience) == (160, (7,), 3)
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol.to_dict(), sort_keys=True))
    assert load_protocol(path).sha256 == protocol.sha256
    assert replace(protocol, max_steps=320).sha256 != protocol.sha256
    with pytest.raises(ValueError, match="override seeds"):
        load_protocol(path, seeds=(19,))
    assert load_protocol(seeds=(11, 19)).seeds == (11, 19)


@pytest.mark.parametrize("values", [
    {"seeds": []}, {"seeds": [7, 7]}, {"seeds": [True]}, {"max_steps": 0},
    {"max_steps": True}, {"patience": 0}, {"noise_rates": [0.7]},
    {"noise_rates": [0, 0]}, {"max_fit_seconds": float("nan")},
    {"learning_rate": 0.5}, {"check_every": 1.5}, {"unknown": 1}, {"min_delta": 2},
])
def test_protocol_rejects_invalid_settings(tmp_path, values):
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(values))
    with pytest.raises(ValueError):
        load_protocol(path)


def test_auc_ranks_raw_logits_even_when_sigmoid_saturates():
    x = torch.tensor([20.0, 21.0, 22.0, 23.0])[:, None, None].repeat(1, 64, 1)
    scores, small, large = evaluate.score_views(MeanModel(), x, np.array([0, 0, 1, 1]))
    assert len(torch.sigmoid(torch.from_numpy(small)).unique()) == 1
    assert scores["base_auc"] == scores["diluted_auc"] == 1.0
    assert np.array_equal(small, large)
    assert scores["positive_probability_change"] == scores["negative_probability_change"] == 0


def test_both_class_shifts_and_margin_change_are_reported():
    x = torch.zeros(4, 64, 1)
    x[:, :16, 0] = torch.tensor([0.0, 1.0, 2.0, 3.0])[:, None]
    x[:, 16:, 0] = torch.tensor([4.0, 5.0, 2.0, 3.0])[:, None]
    scores, small, large = evaluate.score_views(MeanModel(), x, np.array([0, 0, 1, 1]))
    assert scores["positive_logit_change"] == 0
    assert scores["negative_logit_change"] == 3
    assert scores["negative_probability_change"] > 0
    assert scores["positive_probability_change"] == 0
    assert scores["logit_margin_change"] == -3
    assert scores["auc_change"] < 0
    assert list(small) == [0, 1, 2, 3]  # Predictions restored to original bag order.


def test_observed_connected_counts_include_zero_grad_but_exclude_unused_and_frozen():
    protocol = EvaluationProtocol(max_steps=3, check_every=2, batch_size=4, patience=None)
    _, record = evaluate.fit(SimpleNamespace(build_model=CountModel), *toy_data(), seed=7, noise_rate=0, protocol=protocol)
    assert record["total_parameters"] == 8
    assert record["trainable_parameters"] == 7
    assert record["observed_gradient_connected_parameters"] == 4
    assert record["inactive_parameter_names"] == ["frozen", "unused.bias", "unused.weight"]
    assert record["frozen_parameter_names"] == ["frozen"]
    assert record["steps_run"] == record["last_validation_step"] == 3
    assert record["training_hit_step_cap"] is True
    assert [row["step"] for row in record["trace"]] == [2, 3]


def test_parameter_budget_counts_unused_and_frozen_parameters():
    protocol = EvaluationProtocol(max_parameters=7, max_steps=1, batch_size=4)
    with pytest.raises(ValueError, match="Parameter budget exceeded: 8 > 7"):
        evaluate.fit(SimpleNamespace(build_model=CountModel), *toy_data(), seed=7, noise_rate=0, protocol=protocol)


def test_short_budget_validates_final_step_and_time_cap_counts_completed_steps(monkeypatch):
    protocol = EvaluationProtocol(max_steps=1, check_every=20, batch_size=4)
    _, record = evaluate.fit(SimpleNamespace(build_model=CountModel), *toy_data(), seed=7, noise_rate=0, protocol=protocol)
    assert record["selected_step"] == 1
    assert record["selected_checkpoint_at_step_cap"] is True
    clock = iter([0.0, 1.0, 1.1])
    monkeypatch.setattr(evaluate.time, "monotonic", lambda: next(clock))
    protocol = EvaluationProtocol(max_steps=10, check_every=1, max_fit_seconds=0.5, batch_size=4)
    _, record = evaluate.fit(SimpleNamespace(build_model=CountModel), *toy_data(), seed=7, noise_rate=0, protocol=protocol)
    assert record["steps_run"] == 1
    assert record["stop_reason"] == "time_limit"
    assert record["training_hit_step_cap"] is False


@pytest.mark.parametrize("kind", ["row", "slice"])
def test_interface_rejects_order_shortcuts(kind):
    class BadModel(torch.nn.Module):
        def forward(self, x):
            if kind == "row":
                return torch.arange(len(x)).float() % 2
            return x[:, :16].mean(dim=(1, 2))
    _, _, x, y = toy_data()
    with pytest.raises(ValueError, match="MIL interface violation"):
        evaluate.score_views(BadModel(), x, y)


def test_archived_gated_and_sparse_controls_pass_tied_instance_scores():
    for name in ("baseline", "sparse_gated_attention"):
        module = evaluate.load_candidate(calibrate.default_programs()[name])
        model = module.build_model(2).eval()
        with torch.no_grad():
            model.attention_w.weight.zero_()
            evaluate.check_mil_interface(model, toy_data()[2])


def test_evaluate_multiseed_is_reproducible_and_reads_only_search(tmp_path, monkeypatch):
    path = search_archive(tmp_path)
    program = toy_program(tmp_path / "program.py")
    protocol = EvaluationProtocol(seeds=(7, 11), noise_rates=(0,), max_steps=2, check_every=2, batch_size=4)
    protocol_file = tmp_path / "protocol.json"
    protocol_file.write_text(json.dumps(protocol.to_dict()))
    real_load, reads = np.load, []
    def tracked_load(source, *args, **kwargs):
        reads.append(Path(source).name)
        assert Path(source).name == "search.npz"
        return real_load(source, *args, **kwargs)
    monkeypatch.setattr(np, "load", tracked_load)
    results = [evaluate.evaluate(program, tmp_path / f"out{i}", protocol_path=protocol_file, search_data=path) for i in range(2)]
    assert reads == ["search.npz", "search.npz"]
    assert results[0]["combined_score"] == results[1]["combined_score"]
    assert results[0]["protocol_sha256"] == protocol.sha256
    assert len(results[0]["conditions"]) == 2
    assert json.loads((tmp_path / "out0/correct.json").read_text())["correct"] is True
    assert results[0]["evaluation_split"] == "validation"
    assert results[0]["ranking_score_space"] == "raw_logits"


def test_calibration_freezes_inputs_before_first_evaluation_and_detects_changes(tmp_path, monkeypatch):
    path = search_archive(tmp_path)
    program = toy_program(tmp_path / "program.py")
    programs = {name: program for name in calibrate.PROGRAM_NAMES}
    output = tmp_path / "calibration"
    calls = []
    def stop_after_inspection(command, **kwargs):
        plan = json.loads((output / "calibration_plan.json").read_text())
        calibrate.verify_frozen(output, plan)
        assert plan["held_out_data_used"] is False
        assert plan["budgets"] == [160, 320, 640]
        assert plan["seeds"] == [7, 11, 19]
        assert "--holdout" not in command
        assert command[command.index("--search-data") + 1].endswith("frozen/resources/search.npz")
        calls.append(command)
        return SimpleNamespace(returncode=1, stdout="", stderr="deliberate fixture stop")
    monkeypatch.setattr(calibrate.subprocess, "run", stop_after_inspection)
    with pytest.raises(ValueError, match="evaluation failed"):
        calibrate.run_calibration(path, output, programs=programs)
    assert len(calls) == 1
    plan = json.loads((output / "calibration_plan.json").read_text())
    (output / "frozen/programs/baseline/program.py").write_text("changed")
    with pytest.raises(ValueError, match="input changed"):
        calibrate.verify_frozen(output, plan)
    with pytest.raises(ValueError, match="already exists"):
        calibrate.freeze_calibration(path, output, programs)


def test_calibration_rejects_holdout_path_before_opening_it(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "read_text", lambda *_a, **_k: pytest.fail("must reject before any read"))
    with pytest.raises(ValueError, match="only accepts search.npz"):
        calibrate.freeze_calibration(tmp_path / "holdout.npz", tmp_path / "new")


@pytest.mark.parametrize("tamper", ["program_hash", "missing_condition"])
def test_calibration_summary_checks_result_provenance_and_condition_coverage(tmp_path, tamper):
    path = search_archive(tmp_path)
    program = toy_program(tmp_path / "program.py")
    output = tmp_path / "calibration"
    plan = calibrate.freeze_calibration(path, output, {name: program for name in calibrate.PROGRAM_NAMES})
    result = output / "evaluations/baseline/160"
    result.mkdir(parents=True)
    metrics = {
        "evaluation_split": "validation", "protocol_sha256": plan["protocols"]["160"]["sha256"],
        "conditions": [{"seed": seed, "noise_rate": noise} for seed in plan["seeds"] for noise in plan["noise_rates"]],
    }
    for key, path in {
        "search_sha256": "frozen/resources/search.npz",
        "program_sha256": "frozen/programs/baseline/program.py",
        "evaluator_sha256": "frozen/evaluate.py", "protocol_code_sha256": "frozen/protocol.py",
    }.items():
        metrics[key] = plan["input_file_sha256"][path]
    if tamper == "program_hash":
        metrics["program_sha256"] = "wrong"
    else:
        metrics["conditions"].pop()
    (result / "metrics.json").write_text(json.dumps(metrics))
    (result / "correct.json").write_text(json.dumps({"correct": True}))
    with pytest.raises(ValueError, match="provenance|exactly once"):
        calibrate.summarize(output, plan)
