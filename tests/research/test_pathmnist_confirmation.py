"""Prevent a development continuation from reusing its already-viewed holdout."""
import json

import pytest

from examples.pathmnist_mil.evaluate_holdout import prepare_confirmation_protocol


def test_archival_holdout_is_rejected_before_data_access(tmp_path):
    (tmp_path / "study_plan.json").write_text(json.dumps({"independent_holdout_available": False}))
    with pytest.raises(ValueError, match="no unused independent holdout"):
        prepare_confirmation_protocol(tmp_path)
    assert list(tmp_path.iterdir()) == [tmp_path / "study_plan.json"]


def test_new_confirmation_seeds_preserve_training_protocol(tmp_path):
    protocol = {"schema_version": 1, "seeds": [7, 11, 19, 101], "max_steps": 640, "patience": None}
    source = tmp_path / "evaluation_protocol.json"
    source.write_text(json.dumps(protocol))
    actual = prepare_confirmation_protocol(tmp_path)
    assert actual == dict(protocol, seeds=[211, 307])
    assert json.loads(source.read_text()) == protocol


def test_legacy_task_keeps_legacy_confirmation_path(tmp_path):
    assert prepare_confirmation_protocol(tmp_path) is None
