"""Create a new development task with explicit evaluation and literature budgets."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

from src.research.workspace import copy_task

try:
    from .protocol import EvaluationProtocol
except ImportError:
    from protocol import EvaluationProtocol


def create_study(source: Path, output: Path, *, max_steps: int, seeds: list[int]) -> dict:
    """Copy prepared inputs. Never modify a previous task or open its holdout."""
    protocol = EvaluationProtocol(
        seeds=tuple(seeds), max_steps=max_steps, patience=None,
        learning_rate=0.003, label_smoothing=0.0,
    )
    source, output = source.resolve(), output.absolute()
    data = source / "resources/search.npz"
    if not data.is_file():
        raise ValueError("Prepare resources/search.npz before creating a study")
    data_hash = hashlib.sha256(data.read_bytes()).hexdigest()
    manifest = json.loads((data.parent / "data_manifest.json").read_text())
    if data_hash != manifest["search_sha256"]:
        raise ValueError("Search feature hash changed; restore or prepare a new dataset")
    config = yaml.safe_load((source / "research.yaml").read_text())
    model = config["literature"]["model_search"]
    model.update(max_output_tokens=8000, max_search_calls=6, timeout_seconds=300)
    model.setdefault("request_overrides", {}).setdefault("reasoning", {})["effort"] = "high"
    config["literature"].update(max_papers=10, results_per_query=5)
    config["search"].update(width=6, explore_every=2)
    plan = {
        "schema_version": 1,
        "purpose": "development search; no independent confirmation claim",
        "search_data_sha256": data_hash,
        "protocol": protocol.to_dict(),
        "protocol_sha256": protocol.sha256,
        "independent_holdout_available": False,
        "holdout_policy": "Previously viewed holdout is archival; do not read or tune against it.",
    }
    copy_task(source, output)
    (output / "evaluation_protocol.json").write_text(json.dumps(protocol.to_dict(), indent=2) + "\n")
    (output / "study_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    (output / "research.yaml").write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
    brief = (output / "task.md").read_text()
    brief += (
        "\n## Development-only continuation\n\n"
        "Use the frozen evaluation_protocol.json for all methods. The earlier external\n"
        "holdout has already been viewed and must not be opened in this campaign.\n"
        "Report development scores only. A later independent confirmation requires\n"
        "new data and a separate predeclared plan.\n"
    )
    (output / "task.md").write_text(brief)
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, required=True, help="Choose after shared-budget calibration")
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 11, 19])
    args = parser.parse_args()
    plan = create_study(args.source, args.output, max_steps=args.max_steps, seeds=args.seeds)
    print(json.dumps({"task": str(args.output.absolute()), **plan}, indent=2))


if __name__ == "__main__":
    main()
