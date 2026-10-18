"""Freezing a final method collection closes the development selection phase."""
import pytest

from src.research.harness import _resume_plan


def test_frozen_set_without_invocation_prevents_new_development(tmp_path):
    final = tmp_path / "run/final_evaluation"
    final.mkdir(parents=True)
    (final / "comparison_set.json").write_text("{}")
    with pytest.raises(ValueError, match="method set has been frozen"):
        _resume_plan(tmp_path, {}, {"investigation", "final_evaluation"})
    assert _resume_plan(tmp_path, {}, {"final_evaluation", "paper"}) == {"final_evaluation", "paper"}
