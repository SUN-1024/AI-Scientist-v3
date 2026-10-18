"""A campaign declaration must match the host and remain source-bound."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path

import pytest

from src.investigation.portfolio_controller import PortfolioConfig
from src.research import campaign_plan as plans
from tests.research import test_novelty_gate as novelty_fixtures

study = novelty_fixtures.study


def policy(**changes):
    return asdict(PortfolioConfig(enabled=True, initial_ideas=6, promote_ideas=2,
                                tuning_trials_per_idea=4, replication_evals=2,
                                max_valid_evals=16, **changes))


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def rehash(value, field):
    value[field] = hashlib.sha256(json.dumps({key: child for key, child in value.items() if key != field},
        ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


@pytest.fixture
def task(tmp_path):
    (tmp_path / "task.md").write_text("Optimize the public objective under a fixed training budget.")
    (tmp_path / "program_0/src").mkdir(parents=True)
    (tmp_path / "program_0/src/program.py").write_text("def handle(x):\n    return x\n")
    (tmp_path / "program_0/program_spec.toml").write_text('schema_version = 1\ncode_root = "src"\n')
    contract = json.loads((Path(__file__).resolve().parents[2] /
                          "examples/pathmnist_mil_portfolio/task_contract.json").read_text())
    contract["canonical_task"]["sha256"] = hashlib.sha256((tmp_path / "task.md").read_bytes()).hexdigest()
    dump(tmp_path / "task_contract.json", contract)
    dump(tmp_path / "configuration_contract.json", {"schema_version": 1, "parameters": {}})
    return tmp_path


def context(task, actual_policy=None):
    return plans.build_host_context(task, actual_policy or policy(), workers=2,
        downstream_limits={"ablations": {"max_ablations": 3, "max_evals_per_variant": 5}})


def card(index):
    return {
        "title": f"Complete idea {index}", "role": "control" if index == 0 else "proposal",
        "hypothesis": f"Changing strategy {index} improves the fixed development metric.",
        "complete_implementation": f"Implement strategy {index} in the existing candidate entrypoint.\nKeep the evaluator fixed.",
        "falsifier": "No gain under the same development protocol.",
        "decision_rule": "Promote by the host development metric.",
        "whole_idea": {"schema_version": 1, "innovation_axis": "optimization",
            "problem_statement": "Find a robust training strategy.", "contribution": f"Strategy {index}",
            "minimal_validation": {"protocol_id": "public-development-v1", "decision_rule": "Rank development scores.",
                                   "stop_or_pivot": "Stop after the reserved trials."},
            "parameter_space": [{"name": "rate", "type": "float", "bounds": [0.01, 0.1], "default": 0.03, "owner": "candidate"}],
            "selection_reason": "Discriminates plausible optimization mechanisms."},
    }


def agenda(ctx):
    return {"schema_version": 1, "research_question": "Which strategy improves the fixed objective?",
            "execution_plan": {"schema_version": 1, "roots": [card(i) for i in range(ctx["allocation"]["initial_ideas"])],
                               "allocation": deepcopy(ctx["allocation"])}}


def publish(task):
    ctx = context(task)
    proposed = agenda(ctx)
    plan = plans.compile_campaign_plan(proposed, ctx)
    dump(task / "research_agenda.json", proposed)
    plans.write_campaign_artifacts(task, ctx, plan)
    return ctx, proposed, plan


def test_context_uses_resolved_policy_and_unmeasured_seed(task):
    ctx = context(task)
    assert ctx["portfolio_policy"] == policy()
    assert ctx["allocation"]["total_evaluations"] == 16
    assert ctx["allocation"]["guaranteed_tuning_trials_per_root"] == 0
    assert ctx["downstream_limits"]["ablations"]["max_evals_per_variant"] == 5
    assert ctx["seed"]["status"] == "not_evaluated"
    assert ctx["seed"]["source_paths"] == ["program_0/program_spec.toml", "program_0/src/program.py"]
    assert plans.validate_host_context(ctx) == ctx
    assert context(task) == ctx


def test_complete_cards_and_repeated_axes_compile_and_reload(task):
    ctx, proposed, plan = publish(task)
    assert len(plan["roots"]) == 6
    assert {row["whole_idea"]["innovation_axis"] for row in plan["roots"]} == {"optimization"}
    assert plan["roots"][0]["complete_implementation"] == proposed["execution_plan"]["roots"][0]["complete_implementation"]
    assert plan["roots"][0]["role"] == "control"
    assert plan["roots"][0]["root_idea_id"] == "idea_0001"
    assert plan["roots"][0]["idea_key"] == plan["roots"][0]["card_sha256"][:16]
    assert plans.validate_campaign_plan(plan) == plan
    assert plans.validate_campaign_plan(plan, ctx, proposed, policy()) == plan
    assert plans.load_campaign_plan(task, portfolio_policy=policy()) == plan
    plans.write_campaign_artifacts(task, ctx, plan)


@pytest.mark.parametrize("field,value", [("total_evaluations", 30), ("guaranteed_tuning_trials_per_root", 4),
    ("screen_evals_per_root", 2), ("tuning_trials_per_promoted_idea", 5), ("replication_evals", True)])
def test_allocations_cannot_overstate_host_execution(task, field, value):
    ctx = context(task)
    proposed = agenda(ctx)
    proposed["execution_plan"]["allocation"][field] = value
    with pytest.raises(plans.PlanValidationError) as caught:
        plans.compile_campaign_plan(proposed, ctx)
    assert caught.value.errors[0]["code"] == "allocation_mismatch"
    assert caught.value.errors[0]["path"].endswith(field)


def test_all_roots_only_get_guaranteed_tuning_when_all_promote(task):
    actual = policy()
    actual.update(promote_ideas=6, max_valid_evals=32)
    ctx = context(task, actual)
    assert ctx["allocation"]["guaranteed_tuning_trials_per_root"] == 4
    assert plans.compile_campaign_plan(agenda(ctx), ctx)["allocation"]["total_evaluations"] == 32


@pytest.mark.parametrize("change", ["count", "missing_implementation", "unknown_root", "unknown_whole", "unknown_parameter", "host_identity", "bad_role", "empty_implementation", "bad_root_shape", "duplicate"])
def test_bad_cards_fail_closed(task, change):
    ctx = context(task)
    proposed = agenda(ctx)
    roots = proposed["execution_plan"]["roots"]
    if change == "count":
        roots.pop()
    elif change == "missing_implementation":
        roots[0].pop("complete_implementation")
    elif change == "unknown_root":
        roots[0]["trial_budget"] = 100
    elif change == "unknown_whole":
        roots[0]["whole_idea"]["extra"] = "ignored?"
    elif change == "unknown_parameter":
        roots[0]["whole_idea"]["parameter_space"][0]["extra"] = True
    elif change == "host_identity":
        roots[0]["root_idea_id"] = "model-owned"
    elif change == "bad_role":
        roots[0]["role"] = "baseline_authority"
    elif change == "empty_implementation":
        roots[0]["complete_implementation"] = " "
    elif change == "bad_root_shape":
        roots[0] = []
    else:
        roots[1] = deepcopy(roots[0])
    with pytest.raises(plans.PlanValidationError):
        plans.compile_campaign_plan(proposed, ctx)


def test_all_card_errors_are_returned_for_repair(task):
    ctx = context(task)
    proposed = agenda(ctx)
    proposed["execution_plan"]["roots"][0]["complete_implementation"] = ""
    proposed["execution_plan"]["roots"][1]["role"] = "bad"
    with pytest.raises(plans.PlanValidationError) as caught:
        plans.compile_campaign_plan(proposed, ctx)
    assert len(caught.value.errors) == 2


@pytest.mark.parametrize("where", ["agenda", "context", "plan"])
def test_nonfinite_json_rejected_everywhere(task, where):
    ctx = context(task)
    proposed = agenda(ctx)
    plan = plans.compile_campaign_plan(proposed, ctx)
    with pytest.raises(plans.PlanValidationError, match="finite"):
        if where == "agenda":
            proposed["evidence"] = float("nan")
            plans.compile_campaign_plan(proposed, ctx)
        elif where == "context":
            ctx["downstream_limits"]["bad"] = float("inf")
            plans.validate_host_context(ctx)
        else:
            plan["roots"][0]["whole_idea"]["parameter_space"][0]["default"] = float("-inf")
            plans.validate_campaign_plan(plan)


@pytest.mark.parametrize("artifact", ["context", "plan", "card"])
def test_tampered_hashes_rejected(task, artifact):
    ctx = context(task)
    plan = plans.compile_campaign_plan(agenda(ctx), ctx)
    with pytest.raises(plans.PlanValidationError):
        if artifact == "context":
            ctx["workers"] = 3
            plans.validate_host_context(ctx)
        elif artifact == "plan":
            plan["agenda_sha256"] = "0" * 64
            plans.validate_campaign_plan(plan)
        else:
            plan["roots"][0]["complete_implementation"] = "Different intervention."
            rehash(plan, "plan_sha256")
            plans.validate_campaign_plan(plan)


@pytest.mark.parametrize("path", ["task.md", "task_contract.json", "configuration_contract.json", "program_0/program_spec.toml", "program_0/src/program.py"])
def test_loader_binds_each_live_source(task, path):
    publish(task)
    with (task / path).open("a") as stream:
        stream.write("\n ")
    with pytest.raises(plans.PlanValidationError, match="changed"):
        plans.load_campaign_plan(task)


def test_loader_binds_new_source_and_agenda_and_actual_policy(task):
    _, proposed, _ = publish(task)
    (task / "program_0/src/added.py").write_text("x=2")
    with pytest.raises(plans.PlanValidationError, match="changed"):
        plans.load_campaign_plan(task)
    (task / "program_0/src/added.py").unlink()
    proposed["research_question"] += " Different scope."
    dump(task / "research_agenda.json", proposed)
    with pytest.raises(plans.PlanValidationError, match="another agenda"):
        plans.load_campaign_plan(task)
    proposed["research_question"] = "Which strategy improves the fixed objective?"
    dump(task / "research_agenda.json", proposed)
    actual = policy()
    actual["max_transport_failures"] += 1
    with pytest.raises(plans.PlanValidationError, match="Resolved host policy"):
        plans.load_campaign_plan(task, portfolio_policy=actual)


def test_plan_cannot_be_rehashed_to_bind_unrelated_cards(task):
    ctx = context(task)
    proposed = agenda(ctx)
    plan = plans.compile_campaign_plan(proposed, ctx)
    changed = deepcopy(proposed)
    changed["execution_plan"]["roots"][0]["hypothesis"] = "Different hypothesis."
    plan["agenda_sha256"] = plans._digest(changed)
    rehash(plan, "plan_sha256")
    with pytest.raises(plans.PlanValidationError, match="cards differ"):
        plans.validate_campaign_plan(plan, agenda=changed)


def test_legacy_absence_and_declared_or_partial_plan(task):
    assert plans.load_campaign_plan(task) is None
    with pytest.raises(plans.PlanValidationError, match="missing"):
        plans.load_campaign_plan(task, required=True)
    dump(task / "research_agenda.json", agenda(context(task)))
    with pytest.raises(plans.PlanValidationError, match="missing"):
        plans.load_campaign_plan(task)
    (task / "research_agenda.json").unlink()
    dump(task / plans.HOST_CONTEXT_PATH, context(task))
    with pytest.raises(plans.PlanValidationError, match="missing"):
        plans.load_campaign_plan(task)


def test_context_never_reads_heldout_results_evaluators_or_environment(task, monkeypatch):
    forbidden = ["resources/final/labels.json", "resources/final/data.npz", "evaluate.py", "evaluate_final.py", ".env",
                 "program_0/evaluation_manifest.json", "program_0/src/final.py", "program_0/src/labels.json",
                 "program_0/src/evaluator.py", "program_0/src/auth.py", "program_0/src/payload.npz",
                 "program_0/src/held-out.json", "program_0/src/holdout.json", "program_0/src/held_out.json",
                 "program_0/src/hold-out.json", "program_0/src/held.out.json", "program_0/src/HoldOut.py",
                 "program_0/src/held-out/ordinary.py"]
    for relative in forbidden:
        path = task / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("TOP_SECRET")
    original = os.open
    accessed = []
    def guarded(path, *args, **kwargs):
        accessed.append(str(path))
        assert str(path) not in {Path(relative).name for relative in forbidden}
        return original(path, *args, **kwargs)
    monkeypatch.setattr(plans.os, "open", guarded)
    ctx = context(task)
    assert "TOP_SECRET" not in json.dumps(ctx)
    assert not any("resources" in path for path in accessed)


def test_manifest_includes_scalar_whitelist_only_and_binds_raw_bytes(task):
    manifest = {"schema_version": 1, "dataset_name": "generic", "feature_dim": 8,
                "labels": [1, 0], "final_sha256": "secret", "source": {"token": "secret"},
                "splits": {"heldout": {"labels": [1]}}, "access_token": "secret"}
    dump(task / "data_manifest.json", manifest)
    _, _, plan = publish(task)
    ctx = context(task)
    row = next(row for row in ctx["sources"] if row["path"] == "data_manifest.json")
    assert row["safe_fields"] == {"schema_version": 1, "dataset_name": "generic", "feature_dim": 8}
    assert "secret" not in json.dumps(ctx)
    assert plans.load_campaign_plan(task) == plan
    manifest["labels"] = [0, 1]
    dump(task / "data_manifest.json", manifest)
    with pytest.raises(plans.PlanValidationError, match="changed"):
        plans.load_campaign_plan(task)


@pytest.mark.parametrize("relative", ["task.md", "program_0/src/program.py", "program_0/src", "research_context"])
def test_symlinks_are_rejected(task, tmp_path, relative):
    if relative == "research_context":
        outside = tmp_path / "outside"
        outside.mkdir()
        (task / relative).symlink_to(outside, target_is_directory=True)
        with pytest.raises(plans.PlanValidationError, match="symlink"):
            plans.load_campaign_plan(task)
    elif relative == "program_0/src":
        moved = task / "moved"
        (task / relative).rename(moved)
        (task / relative).symlink_to(moved, target_is_directory=True)
        with pytest.raises(plans.PlanValidationError, match="symlink"):
            context(task)
    else:
        path = task / relative
        moved = path.with_name("target_file")
        path.rename(moved)
        path.symlink_to(moved)
        with pytest.raises(plans.PlanValidationError, match="symlink"):
            context(task)


def test_size_limit_and_unknown_keys(task):
    (task / "program_0/src/program.py").write_bytes(b"a" * (plans.MAX_SOURCE_BYTES + 1))
    with pytest.raises(plans.PlanValidationError, match="bounded"):
        context(task)
    (task / "program_0/src/program.py").write_text("pass")
    ctx = context(task)
    ctx["credential"] = "not allowed"
    with pytest.raises(plans.PlanValidationError, match="Unknown keys"):
        plans.validate_host_context(ctx)


def test_downstream_auth_fields_and_unknown_execution_keys_fail(task):
    with pytest.raises(plans.PlanValidationError, match="Authentication"):
        plans.build_host_context(task, policy(), 2, {"provider": {"api_key": "do not disclose"}})
    ctx = context(task)
    proposed = agenda(ctx)
    proposed["execution_plan"]["max_trials"] = 30
    with pytest.raises(plans.PlanValidationError, match="Unknown keys"):
        plans.compile_campaign_plan(proposed, ctx)


def test_duplicate_json_keys_rejected_on_load(task):
    publish(task)
    (task / plans.CAMPAIGN_PLAN_PATH).write_text('{"schema_version":1,"schema_version":2}')
    with pytest.raises(plans.PlanValidationError, match="Duplicate"):
        plans.load_campaign_plan(task)


def test_published_artifacts_cannot_be_overwritten(task):
    ctx, proposed, _ = publish(task)
    proposed["research_question"] += " New claim."
    changed = plans.compile_campaign_plan(proposed, ctx)
    with pytest.raises(plans.PlanValidationError, match="already exists"):
        plans.write_campaign_artifacts(task, ctx, changed)


def test_contract_uses_existing_public_projection_and_binds_private_bytes(task):
    from src.investigation.public_task_contract import render_public_search_contract
    from src.investigation.task_contract import parse_task_contract

    contract_path = task / "task_contract.json"
    raw = json.loads(contract_path.read_text())
    raw["compute"]["search"]["precision"] = "PUBLIC_SEARCH_PRECISION"
    raw["compute"]["final"]["precision"] = "SEALED_FINAL_PRECISION"
    raw["metrics"]["final"][0]["name"] = "SEALED_FINAL_METRIC"
    raw["feedback"]["final_sealed"] = ["SEALED_FINAL_FEEDBACK"]
    raw["protocol"]["final_seeds"] = [987654321]
    raw["protocol"]["final_reset_sequence"] = [987654321]
    raw["protocol"].pop("final_smoke_reset_sequence", None)
    raw["protocol"]["baselines"] = ["SEALED_SHARED_BASELINE"]
    dump(contract_path, raw)
    ctx, _, plan = publish(task)
    row = next(row for row in ctx["sources"] if row["path"] == "task_contract.json")
    assert "text" not in row
    assert row["bytes"] == len(contract_path.read_bytes())
    assert row["sha256"] == hashlib.sha256(contract_path.read_bytes()).hexdigest()
    assert row["safe_fields"] == {"candidate_editable_paths": ["program_0/src"],
        "public_search_contract": render_public_search_contract(parse_task_contract(raw))}
    rendered = json.dumps(ctx)
    assert "PUBLIC_SEARCH_PRECISION" in rendered
    for sealed in ("SEALED_FINAL_PRECISION", "SEALED_FINAL_METRIC", "SEALED_FINAL_FEEDBACK",
                   "SEALED_SHARED_BASELINE", "987654321", "final_seeds", "final_reset_sequence", "final_sealed"):
        assert sealed not in rendered
    assert plans.load_campaign_plan(task) == plan
    raw["compute"]["final"]["precision"] = "CHANGED_SEALED_PRECISION"
    dump(contract_path, raw)
    changed = context(task)
    changed_row = next(row for row in changed["sources"] if row["path"] == "task_contract.json")
    assert changed_row["safe_fields"] == row["safe_fields"]
    assert changed_row["sha256"] != row["sha256"]
    with pytest.raises(plans.PlanValidationError, match="changed"):
        plans.load_campaign_plan(task)


def test_contract_scope_and_sealed_artifact_role_precede_filename_heuristics(task, monkeypatch):
    raw = json.loads((task / "task_contract.json").read_text())
    raw["candidate"]["editable_paths"] = ["program_0/src/program.py", "program_0/src/model"]
    raw["provenance"]["artifacts"] = [{"role": "sealed_test", "path": "program_0/src/model/values.json",
        "sha256": "0" * 64, "source": "sealed fixture"}]
    dump(task / "task_contract.json", raw)
    (task / "program_0/src/model").mkdir()
    (task / "program_0/src/model/strategy.py").write_text("PUBLIC_CANDIDATE_SOURCE = True")
    (task / "program_0/src/model/values.json").write_text('{"private":[42]}')
    (task / "program_0/src/helper.py").write_text("PRIVATE_NONCANDIDATE_CODE")
    original = os.open
    def guarded(path, *args, **kwargs):
        assert str(path) not in {"helper.py", "values.json"}
        return original(path, *args, **kwargs)
    monkeypatch.setattr(plans.os, "open", guarded)
    ctx = context(task)
    paths = [row["path"] for row in ctx["sources"]]
    assert "program_0/src/model/strategy.py" in paths
    assert "program_0/src/program.py" in paths
    assert "program_0/src/model/values.json" not in paths
    assert "program_0/src/helper.py" not in paths
    assert "PRIVATE_NONCANDIDATE_CODE" not in json.dumps(ctx)


def test_invalid_contract_cannot_authorize_candidate_or_evaluator_reads(task, monkeypatch):
    raw = json.loads((task / "task_contract.json").read_text())
    raw["evaluators"]["final"] = "program_0/src/ordinary.py"
    dump(task / "task_contract.json", raw)
    (task / "program_0/src/ordinary.py").write_text("SEALED_EVALUATOR")
    original = os.open
    def guarded(path, *args, **kwargs):
        assert str(path) not in {"ordinary.py", "program.py"}
        return original(path, *args, **kwargs)
    monkeypatch.setattr(plans.os, "open", guarded)
    with pytest.raises(plans.PlanValidationError, match="evaluators.final"):
        context(task)


def test_legacy_sources_stay_visible_without_a_contract(task):
    (task / "task_contract.json").unlink()
    (task / "program_0/src/helper.py").write_text("PUBLIC_HELPER = True")
    (task / "program_0/src/holdout.json").write_text('{"labels":[42]}')
    ctx = context(task)
    paths = [row["path"] for row in ctx["sources"]]
    assert "program_0/src/helper.py" in paths and "program_0/src/program.py" in paths
    assert "program_0/src/holdout.json" not in paths
    assert plans.validate_host_context(ctx) == ctx


@pytest.mark.parametrize("corruption", ["raw_text", "extra_field", "nested_final", "oversized"])
def test_public_contract_projection_cannot_smuggle_full_contract(task, corruption):
    ctx = context(task)
    row = next(row for row in ctx["sources"] if row["path"] == "task_contract.json")
    if corruption == "raw_text":
        row["text"] = (task / "task_contract.json").read_text()
        row.pop("safe_fields")
    elif corruption == "extra_field":
        row["safe_fields"]["raw_contract"] = json.loads((task / "task_contract.json").read_text())
    elif corruption == "nested_final":
        rendered = row["safe_fields"]["public_search_contract"]
        before, payload = rendered.rsplit("\n\n", 1)
        value = json.loads(payload)
        value["search"]["final_seeds"] = [987654321]
        row["safe_fields"]["public_search_contract"] = before + "\n\n" + json.dumps(value)
    else:
        row["safe_fields"]["public_search_contract"] += "a" * 20000
    rehash(ctx, "host_context_sha256")
    with pytest.raises(plans.PlanValidationError):
        plans.validate_host_context(ctx)


def test_non_candidate_source_cannot_be_added_to_rehashed_context(task):
    raw = json.loads((task / "task_contract.json").read_text())
    raw["candidate"]["editable_paths"] = ["program_0/src/program.py"]
    dump(task / "task_contract.json", raw)
    ctx = context(task)
    data = b"private = 42"
    ctx["sources"].append({"path": "program_0/src/ordinary.py", "bytes": len(data),
                           "sha256": hashlib.sha256(data).hexdigest(), "text": data.decode()})
    ctx["sources"].sort(key=lambda row: row["path"])
    ctx["seed"] = plans._seed(ctx["sources"])
    rehash(ctx, "host_context_sha256")
    with pytest.raises(plans.PlanValidationError, match="editable paths"):
        plans.validate_host_context(ctx)


def test_actual_first_author_prompt_omits_contract_secrets_and_holdout_files(study, tmp_path, monkeypatch):
    from src.research.novelty_review import draft_research_agenda
    from src.research.research_agenda import normalize_agenda

    task, cfg, base_agenda, _review = study
    raw = json.loads((Path(__file__).resolve().parents[2] /
                      "examples/pathmnist_mil_portfolio/task_contract.json").read_text())
    raw["canonical_task"]["sha256"] = hashlib.sha256((task / "task.md").read_bytes()).hexdigest()
    raw["compute"]["search"]["precision"] = "PUBLIC_PRECISION_IN_PROMPT"
    raw["compute"]["final"]["precision"] = "PRIVATE_PRECISION_NOT_IN_PROMPT"
    raw["metrics"]["final"][0]["name"] = "PRIVATE_METRIC_NOT_IN_PROMPT"
    raw["feedback"]["final_sealed"] = ["PRIVATE_FEEDBACK_NOT_IN_PROMPT"]
    raw["protocol"]["final_seeds"] = [987654321]
    raw["protocol"]["final_reset_sequence"] = [987654321]
    raw["protocol"].pop("final_smoke_reset_sequence", None)
    dump(task / "task_contract.json", raw)
    (task / "program_0/src").mkdir(parents=True)
    (task / "program_0/src/program.py").write_text("PUBLIC_CANDIDATE_IN_PROMPT = True")
    (task / "program_0/src/held-out.json").write_text("PRIVATE_HELD_OUT_NOT_IN_PROMPT")
    (task / "program_0/src/holdout.json").write_text("PRIVATE_HOLDOUT_NOT_IN_PROMPT")
    ctx = context(task)
    proposed = normalize_agenda({**base_agenda, "execution_plan": agenda(ctx)["execution_plan"]})
    calls = novelty_fixtures.install_provider(monkeypatch, [proposed])
    assert draft_research_agenda(task, cfg, tmp_path / "author", host_context=ctx,
                                 num_reflections=1) == proposed
    prompt = calls[0]["messages"][0]["content"]
    assert "PUBLIC_PRECISION_IN_PROMPT" in prompt and "PUBLIC_CANDIDATE_IN_PROMPT" in prompt
    for sealed in ("PRIVATE_PRECISION_NOT_IN_PROMPT", "PRIVATE_METRIC_NOT_IN_PROMPT",
                   "PRIVATE_FEEDBACK_NOT_IN_PROMPT", "PRIVATE_HELD_OUT_NOT_IN_PROMPT",
                   "PRIVATE_HOLDOUT_NOT_IN_PROMPT", "987654321", "final_seeds", "final_reset_sequence"):
        assert sealed not in prompt
