"""Reuse an accepted design after interruption; all scaffold gates still run."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from .design import TaskDesignError, load_task_design, load_task_design_review


def restore_design(source: Path, task: Path, destination: Path) -> None:
    names = ('task_design.json', 'accepted_review.json', 'design_receipt.json')
    blobs = {}
    for name in names:
        path = source / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 1024 * 1024:
            raise TaskDesignError(f'Invalid recovery artifact: {name}')
        blobs[name] = path.read_bytes()
    receipt = json.loads(blobs['design_receipt.json'])
    if receipt.get('schema_version') != 1 or receipt.get('verdict') != 'pass':
        raise TaskDesignError('Recovery requires an accepted design receipt')
    for name, key in [('task_design.json', 'design_sha256'), ('accepted_review.json', 'review_sha256')]:
        if hashlib.sha256(blobs[name]).hexdigest() != receipt.get(key):
            raise TaskDesignError(f'Recovery digest mismatch: {name}')
    design = load_task_design(source / 'task_design.json', canonical_task=task)
    review = load_task_design_review(source / 'accepted_review.json', design=design)
    if review.verdict != 'pass' or review.failed_dimensions:
        raise TaskDesignError('Recovery review did not pass')
    destination.mkdir(mode=0o700)  # Never overwrite an existing design.
    for name, blob in blobs.items():
        (destination / name).write_bytes(blob)
    (destination / 'recovery.json').write_text(json.dumps({
        'source': str(source.resolve()), 'design_fingerprint': design.fingerprint,
        'scaffold_validation_reused': False,
    }, indent=2) + '\n')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--task', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    restore_design(args.source, args.task, args.destination)


if __name__ == '__main__':
    main()
