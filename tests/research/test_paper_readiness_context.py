from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.research.evidence import render_context


@pytest.mark.parametrize("with_catalog", [False, True])
def test_paper_receives_actual_configured_targets_without_provider_settings(tmp_path: Path, with_catalog: bool) -> None:
    body = """schema_version: 1
methodology:
  enabled: false
literature:
  model_search:
    request_overrides:
      private_marker: do-not-copy-provider-settings
harness:
  quality:
    min_confirmation_seeds: 5
    required_comparisons: [baseline_lme, control_best]
    positive_effect_metric: candidate_minus_lme
    positive_effect_interval_low_metric: paired_ci95_low
    positive_effect_min: 0.02
"""
    (tmp_path / "research.yaml").write_text(body, encoding="utf-8")
    if with_catalog:
        context = tmp_path / "research_context"
        context.mkdir()
        (context / "catalog.json").write_text(json.dumps({"schema_version": 1, "status": "ready", "sources": []}))
    rendered = render_context(tmp_path, for_paper=True)
    target = json.loads("{" + rendered.rsplit("\n{", 1)[1])
    assert target["config_sha256"] == hashlib.sha256(body.encode()).hexdigest()
    assert target["requirements"]["min_confirmation_seeds"] == 5
    assert target["requirements"]["required_comparisons"] == ["baseline_lme", "control_best"]
    assert target["scientific_effect_rule"] == {
        "enabled": True,
        "operands": ["candidate_minus_lme", "paired_ci95_low"],
        "operator": ">", "threshold": 0.02, "combine": "AND", "missing_metric_name_status": "undeclared",
    }
    assert "requirements, not achieved results" in rendered
    assert "planned analyses as completed" in rendered
    assert "do-not-copy-provider-settings" not in rendered
    assert "CONFIGURED HARNESS READINESS TARGETS" not in render_context(tmp_path)


def test_missing_config_preserves_legacy_paper_context(tmp_path: Path) -> None:
    assert render_context(tmp_path, for_paper=True) == (
        "No structured literature dossier is attached. Do not infer that novelty has been verified."
    )


def test_symlinked_config_cannot_supply_paper_targets(tmp_path: Path) -> None:
    source = tmp_path / "external.yaml"
    source.write_text("schema_version: 1\n")
    (tmp_path / "research.yaml").symlink_to(source)
    with pytest.raises(ValueError, match="Invalid research configuration"):
        render_context(tmp_path, for_paper=True)
