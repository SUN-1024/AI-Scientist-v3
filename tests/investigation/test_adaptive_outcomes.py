"""Adaptive planning consumes valid negative results without promoting them."""
import hashlib
import json
import sys
from pathlib import Path

import pytest

from src.investigation.native_harness import (
    run_native_evaluation,
    source_fingerprint,
    validated_native_outcome,
    validated_native_score,
)
from tests.program_testutils import make_program_dir


@pytest.fixture
def native_task(tmp_path: Path) -> tuple[Path, Path]:
    task = tmp_path / "task"
    task.mkdir()
    program = task / "program_1"
    make_program_dir(program, content="SCORE = 0.4\nCORRECT = False\n")
    evaluator = task / "evaluate.py"
    evaluator.write_text(
        """import argparse, json, runpy
from pathlib import Path
parser = argparse.ArgumentParser()
parser.add_argument('--program_path')
parser.add_argument('--results_dir')
args = parser.parse_args()
counter = Path(__file__).with_name('execution_count.txt')
counter.write_text(str(int(counter.read_text()) + 1 if counter.exists() else 1))
values = runpy.run_path(args.program_path)
out = Path(args.results_dir)
(out / 'metrics.json').write_text(json.dumps({'combined_score': values['SCORE']}))
(out / 'correct.json').write_text(json.dumps({'correct': values['CORRECT']}))
""",
        encoding="utf-8",
    )
    (task / "protocol.json").write_text('{"mode": "development"}')
    (task / "native_evaluation_inputs.json").write_text(json.dumps({
        "schema_version": 1, "required": ["protocol.json"],
    }))
    return program, evaluator


def evaluate(program: Path, evaluator: Path, **identity: str) -> int:
    return run_native_evaluation(
        program_dir=program, evaluator=evaluator, python_bin=sys.executable,
        timeout_seconds=10, environment={},
        attempt_id=identity.get("attempt_id", "attempt_01"),
        idea_id=identity.get("idea_id", "idea_01"),
    )


@pytest.mark.parametrize("correct", [False, True])
def test_completed_outcome_is_distinct_from_promotion(native_task, correct):
    program, evaluator = native_task
    (program / "src/program.py").write_text(f"SCORE = 0.4\nCORRECT = {correct!r}\n")
    assert evaluate(program, evaluator) == 0
    outcome = validated_native_outcome(program, evaluator)
    assert outcome == {
        "receipt_id": hashlib.sha256((program / "evaluation_manifest.json").read_bytes()).hexdigest(),
        "score": 0.4,
        "correct": correct,
        "program_dir": str(program.resolve()),
        "source_sha256": source_fingerprint(program),
        "attempt_id": "attempt_01",
        "idea_id": "idea_01",
    }
    assert validated_native_score(program, evaluator) == (0.4 if correct else None)


@pytest.mark.parametrize("relative", [
    "program_1/src/program.py", "evaluate.py", "protocol.json",
    "program_1/metrics.json", "program_1/correct.json",
])
def test_tampered_measurements_are_not_adaptive_evidence(native_task, relative):
    program, evaluator = native_task
    assert evaluate(program, evaluator) == 0
    original = validated_native_outcome(program, evaluator)
    path = evaluator.parent / relative
    before = path.read_bytes()
    path.write_bytes(before + b"\n")
    assert validated_native_outcome(program, evaluator) is None
    path.write_bytes(before)
    assert validated_native_outcome(program, evaluator) == original


@pytest.mark.parametrize("field,value", [
    ("producer", "candidate"), ("execution_boundary", "sandbox"),
    ("program_id", "another_program"), ("exit_code", 1), ("returncode", 1),
])
def test_outcome_rejects_invalid_host_manifest(native_task, field, value):
    program, evaluator = native_task
    assert evaluate(program, evaluator) == 0
    manifest_path = program / "evaluation_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest[field] = value
    manifest_path.write_text(json.dumps(manifest))
    assert validated_native_outcome(program, evaluator) is None


@pytest.mark.parametrize("source", [
    "SCORE = 0.4\nCORRECT = None\n",
    "SCORE = 0.4\nCORRECT = 0\n",
    "SCORE = 0.4\nCORRECT = 'false'\n",
    "SCORE = True\nCORRECT = False\n",
    "SCORE = float('nan')\nCORRECT = False\n",
    "SCORE = float('inf')\nCORRECT = False\n",
    "SCORE = 10 ** 400\nCORRECT = False\n",
])
def test_sealed_outputs_still_require_boolean_correctness_and_finite_score(native_task, source):
    program, evaluator = native_task
    (program / "src/program.py").write_text(source)
    assert evaluate(program, evaluator) == 0
    assert validated_native_outcome(program, evaluator) is None


@pytest.mark.parametrize("correct", [False, True])
def test_matching_identity_reuses_receipt_without_running_again(native_task, correct):
    program, evaluator = native_task
    (program / "src/program.py").write_text(f"SCORE = 0.4\nCORRECT = {correct!r}\n")
    assert evaluate(program, evaluator) == 0
    manifest = (program / "evaluation_manifest.json").read_bytes()
    assert evaluate(program, evaluator) == 0
    assert (evaluator.parent / "execution_count.txt").read_text() == "1"
    assert (program / "evaluation_manifest.json").read_bytes() == manifest


@pytest.mark.parametrize("identity", [
    {"attempt_id": "attempt_02"}, {"idea_id": "idea_02"},
])
def test_different_identity_produces_a_new_receipt(native_task, identity):
    program, evaluator = native_task
    assert evaluate(program, evaluator) == 0
    previous = validated_native_outcome(program, evaluator)
    assert evaluate(program, evaluator, **identity) == 0
    current = validated_native_outcome(program, evaluator)
    assert previous is not None and current is not None
    assert current["receipt_id"] != previous["receipt_id"]
    for name, value in identity.items():
        assert current[name] == value
    assert (evaluator.parent / "execution_count.txt").read_text() == "2"


def test_changed_code_with_same_identity_must_run_again(native_task):
    program, evaluator = native_task
    assert evaluate(program, evaluator) == 0
    (program / "src/program.py").write_text("SCORE = 0.6\nCORRECT = True\n")
    assert evaluate(program, evaluator) == 0
    assert validated_native_score(program, evaluator) == 0.6
    assert (evaluator.parent / "execution_count.txt").read_text() == "2"


def test_rerun_cannot_reseal_outputs_left_by_an_older_evaluation(native_task):
    program, evaluator = native_task
    assert evaluate(program, evaluator) == 0
    evaluator.write_text("# A broken evaluator exits without producing any outputs.\n")
    assert evaluate(program, evaluator, attempt_id="attempt_02") == 1
    assert validated_native_outcome(program, evaluator) is None
    assert not (program / "metrics.json").exists()
    assert not (program / "correct.json").exists()
