"""Host replication retains the canonical sealed-runtime interpreter contract."""
from pathlib import Path
from unittest.mock import patch

import pytest

from src.investigation.adaptive_runtime import run_frozen_replication
from src.investigation.search_tree import _fingerprint
from tests.program_testutils import make_program_dir


def test_frozen_sealed_replication_uses_canonical_managed_python(tmp_path):
    task = tmp_path / "task"
    candidate = task / "program_0"
    make_program_dir(candidate, content="print('fixed')\n")
    (task / "task_contract.json").write_text("{}")
    (task / "evaluate.py").write_text("# canonical evaluator\n")
    python = task / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text("container-only managed runtime placeholder\n")
    rubrics = tmp_path / "rubrics.json"
    rubrics.write_text("[]")
    with patch("src.investigation.harness_eval.run_harness_evaluation", return_value=0) as evaluate:
        result = run_frozen_replication(task_dir=task, experiment_dir=tmp_path / "attempt",
            candidate_dir=candidate, source_hash=_fingerprint(candidate), attempt_id="repeat_01", idea_id="repeat",
            rubrics_file=rubrics, global_memory_dir=tmp_path / "memory", timeout_seconds=10,
            containerize=True, container_image="sha256:fixture", gpu_evaluation=False)
    assert result.exit_code == 0
    assert evaluate.call_args.kwargs["python_bin"] == str(python)
    assert evaluate.call_args.kwargs["sealed_config"].task_source_dir == task
    assert evaluate.call_args.kwargs["sealed_config"].sandbox_image == "sha256:fixture"
    assert _fingerprint(tmp_path / "attempt/program_1") == _fingerprint(candidate)


def test_contracted_replication_never_falls_back_to_native_evaluation(tmp_path):
    task = tmp_path / "task"
    candidate = task / "program_0"
    make_program_dir(candidate, content="print('fixed')\n")
    (task / "task_contract.json").write_text("{}")
    with patch("src.investigation.harness_eval.run_harness_evaluation") as evaluate:
        with pytest.raises(ValueError, match="sealed evaluation boundary"):
            run_frozen_replication(task_dir=task, experiment_dir=tmp_path / "attempt",
                candidate_dir=candidate, source_hash=_fingerprint(candidate), attempt_id="repeat_01", idea_id="repeat",
                rubrics_file=Path("unused"), global_memory_dir=tmp_path / "memory", timeout_seconds=10,
                containerize=False, container_image="", gpu_evaluation=False)
    evaluate.assert_not_called()
