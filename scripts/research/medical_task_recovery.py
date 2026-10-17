"""Recover task files without replacing scientific acceptance or ownership checks."""
from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

MAX_RUNTIME_BYTES = 17179869184
IGNORED_ROOT_ENTRIES = {
    'resources', '.venv', '.agentic_runtime.json', '.agentic_task_snapshot.json',
    '.agentic_task_validation.json', '.agentic_task_publication.json',
    '.agentic_task_qualification.json',
}


def stage_recovered_task(source: Path, destination: Path) -> dict:
    """Create a new seed copy; the resource publisher owns its resource copy."""
    from src.task_drafting.publication import _publish_resources, _RESOURCE_MANIFEST
    from src.task_drafting.publication_fs import resource_tree_digest

    source = source.resolve(strict=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError('Recovery destination must be new')
    canonical = (source / 'task.md').read_bytes()

    def ignore(directory, names):
        excluded = {name for name in names if name == '__pycache__' or name.endswith('.pyc')}
        if Path(directory) == source:
            excluded |= IGNORED_ROOT_ENTRIES.intersection(names)
        return excluded

    shutil.copytree(source, destination, ignore=ignore)
    resources = source / 'resources'
    digest = None
    if resources.exists() or resources.is_symlink():
        digest = resource_tree_digest(resources, _RESOURCE_MANIFEST)
        _publish_resources(resources, destination / 'resources',
                           max_runtime_bytes=MAX_RUNTIME_BYTES, expected_sha256=digest)
    if (destination / 'task.md').read_bytes() != canonical:
        raise ValueError('Recovery changed the canonical task')
    return {'canonical_task_sha256': hashlib.sha256(canonical).hexdigest(),
            'resource_sha256': digest, 'scientific_validation_reused': False}


def metric_digest(path: Path) -> str:
    nodes = {node.name: node for node in ast.parse(path.read_text()).body
             if isinstance(node, ast.FunctionDef)}
    tree = ast.Module(body=[nodes[name] for name in ('boundary_f1_counts', 'macro_boundary_f1')],
                      type_ignores=[])
    return hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()


def publish_validated_recovery(task: Path, staged: Path, output: Path,
                               recovery: Path, expected_metric: str) -> dict:
    """Finish only publication after genuine, unchanged scientific gates passed."""
    from src.investigation.task_alignment import load_task_alignment_report
    from src.investigation.task_runtime import _tooling_digest
    from src.investigation.task_runtime_cache import (
        TaskRuntimeInstallCache, TaskRuntimeInstallCacheKey, prepare_task_runtime_install_cache,
    )
    from src.task_drafting.publication import (
        _RESOURCE_MANIFEST, publish_task_scaffold, task_publication_plan_sha256,
        validate_published_task_plan,
    )
    from src.task_drafting.publication_fs import resource_tree_digest
    from src.task_drafting.qualification import (
        activate_qualified_task_runtime, load_task_qualification_receipt,
    )

    alignment = output / '_task_validation/round_0/task_alignment.json'
    qualification = output / '_task_validation/qualification_0_receipt.json'
    report = load_task_alignment_report(alignment, scaffold_dir=staged)
    if report.verdict != 'pass':
        raise ValueError('Independent alignment must pass')
    plan = task_publication_plan_sha256(task_dir=task, staged_task_dir=staged,
                                       max_runtime_bytes=MAX_RUNTIME_BYTES)
    receipt = load_task_qualification_receipt(qualification,
                                             expected_publication_plan_sha256=plan)
    if not receipt.baseline_correct or not receipt.baseline_feasible:
        raise ValueError('Full production baseline must be correct and feasible')
    if _tooling_digest() != receipt.tooling_sha256:
        raise ValueError('Scientific tooling differs from qualification')
    for root in (task, staged):
        for name in ('evaluate.py', 'evaluate_final.py'):
            if metric_digest(root / name) != expected_metric:
                raise ValueError('Corrected boundary metric changed')
    manifest = json.loads((staged / '.agentic_runtime.json').read_text())
    for name in ('image_id', 'tooling_sha256', 'requirements_sha256',
                 'evaluator_tree_sha256', 'baseline_combined_score',
                 'baseline_correct', 'baseline_feasible'):
        if manifest[name] != getattr(receipt, name):
            raise ValueError('Qualified runtime metadata differs: ' + name)
    resource_digest = resource_tree_digest(staged / 'resources', _RESOURCE_MANIFEST)
    target_resources = task / 'resources'
    unmanaged = target_resources.exists() and not (target_resources / _RESOURCE_MANIFEST).exists()
    if unmanaged and resource_tree_digest(target_resources, _RESOURCE_MANIFEST) != resource_digest:
        raise ValueError('Existing resources differ; refusing to move them')
    recovery.mkdir(mode=0o700, exist_ok=True)
    if (recovery / 'receipt.json').exists():
        raise FileExistsError('Publication recovery receipt already exists')

    # Rebuild a discarded install cache from the still-intact qualified runtime.
    # store() copies it and validates its tree; it does not manufacture a gate receipt.
    cache_root = prepare_task_runtime_install_cache(task, reset=False)
    cache = TaskRuntimeInstallCache(cache_root)
    key = TaskRuntimeInstallCacheKey(receipt.image_id, receipt.requirements_sha256,
                                    receipt.tooling_sha256)
    if key.digest != receipt.runtime_cache_key:
        raise ValueError('Qualified runtime cache identity changed')
    print('[RECOVERY] Scientific gate receipts and code binding verified; restoring qualified install cache', file=sys.stderr, flush=True)
    cache.store(key, staged / '.venv')
    backup = None
    if unmanaged:
        backup = recovery / 'prior_resources'
        target_resources.rename(backup)
    print('[RECOVERY] Publishing unchanged qualified scaffold through the framework publisher', file=sys.stderr, flush=True)
    publish_task_scaffold(task_dir=task, staged_task_dir=staged, output_dir=output,
                          alignment_report_path=alignment, qualification_receipt_path=qualification,
                          max_runtime_bytes=MAX_RUNTIME_BYTES)
    if validate_published_task_plan(task) != plan:
        raise ValueError('Published task differs from its qualified plan')
    print('[RECOVERY] Activating and verifying the qualified runtime', file=sys.stderr, flush=True)
    activation = activate_qualified_task_runtime(
        task, container_image=receipt.image_id, gpu_evaluation=True,
        timeout_seconds=1800, base_environment=os.environ,
    )
    if not activation.ready:
        raise RuntimeError('Qualified runtime activation failed: ' + activation.diagnostic)
    if validate_published_task_plan(task) != plan:
        raise ValueError('Runtime activation changed the publication')
    published = load_task_qualification_receipt(task / '.agentic_task_qualification.json',
                                               expected_publication_plan_sha256=plan)
    result = {'status': 'pass', 'publication_plan_sha256': plan,
              'baseline_correct': published.baseline_correct,
              'baseline_feasible': published.baseline_feasible,
              'resource_sha256': resource_digest, 'prior_resources': str(backup) if backup else None,
              'runtime_cache_key': key.digest, 'runtime_activated': True,
              'scientific_protocol_changed': False, 'scientific_gates_rerun': False,
              'reason': 'Complete publication with existing matching, genuine gate receipts'}
    (recovery / 'receipt.json').write_text(json.dumps(result, indent=2) + '\n')
    return result
