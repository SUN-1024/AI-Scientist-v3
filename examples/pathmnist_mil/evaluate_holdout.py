"""Evaluate a frozen baseline and selected program after the search has ended.

This is a native development check, not V3's sealed container final-evaluation
stage. No holdout result is returned to the search loop.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import os
import subprocess
import sys

from src.investigation.program_clone import clone_program_dir
from src.investigation.native_harness import source_fingerprint, validated_native_score
from src.research.workspace import resume


def prepare_confirmation_protocol(task):
    """Check the study boundary before opening data; freeze new training seeds."""
    study_plan = task / 'study_plan.json'
    if study_plan.is_file():
        plan = json.loads(study_plan.read_text())
        if plan.get('independent_holdout_available') is False:
            raise ValueError('This development study has no unused independent holdout; do not reuse archival data')
    source = task / 'evaluation_protocol.json'
    if not source.is_file():
        return None
    protocol = json.loads(source.read_text())
    prior_seeds = set(protocol.get('seeds', [7]))
    seeds = [seed for seed in (101, 211, 307, 401) if seed not in prior_seeds][:2]
    if len(seeds) != 2:
        raise ValueError('Choose two new confirmation seeds in a separate predefined protocol')
    protocol['seeds'] = seeds
    return protocol


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--holdout', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--include-best-challenger', action='store_true',
                        help='Also freeze the best correct candidate by validation score before reading holdout')
    args = parser.parse_args()
    workspace, holdout, output = (p.expanduser().resolve() for p in (args.workspace, args.holdout, args.output))
    if output.exists() or output.is_relative_to(workspace):
        raise ValueError('Choose a new output directory outside the search workspace')
    resume(workspace)  # Verify that the task and literature inputs remain frozen.
    report = json.loads((workspace / 'research_report.json').read_text())
    if report.get('state') != 'pipeline_returned' or report.get('pipeline_exit_code') != 0:
        raise ValueError('Finish the search successfully before opening the holdout')
    task = workspace / 'input_task'
    confirmation_protocol = prepare_confirmation_protocol(task)
    confirmation_seeds = confirmation_protocol['seeds'] if confirmation_protocol else [11, 19]
    manifest = json.loads((workspace / 'run/investigation/best_program_manifest.json').read_text())
    selected = Path(manifest['program_dir']).resolve()
    if not selected.is_relative_to(workspace / 'run') or manifest.get('correct') is not True:
        raise ValueError('Selected program must be a correct program inside the completed run')
    sources = [('baseline', task / 'program_0'), ('selected', selected)]
    if args.include_best_challenger:
        candidates = []
        ideas = workspace / 'run/investigation/ideas'
        candidates_on_disk = set(ideas.glob('*/program_1')) | set(ideas.glob('*/*_attempt/program_1'))
        for candidate in sorted(candidates_on_disk):
            if candidate.is_symlink() or not candidate.resolve().is_relative_to(workspace / 'run'):
                raise ValueError('Candidate must stay inside the completed run')
            score = validated_native_score(candidate, task / 'evaluate.py')
            if score is not None:
                candidates.append((score, candidate))
        if not candidates:
            raise ValueError('No correctly evaluated challenger is available')
        sources.append(('best_challenger', max(candidates, key=lambda item: item[0])[1]))
    data_manifest = json.loads((task / 'resources/data_manifest.json').read_text())
    if hashlib.sha256(holdout.read_bytes()).hexdigest() != data_manifest['holdout_sha256']:
        raise ValueError('Holdout hash does not match the original construction manifest')
    output.mkdir(parents=True)
    if confirmation_protocol:
        (output / 'confirmation_protocol.json').write_text(json.dumps(confirmation_protocol, indent=2) + '\n')
    frozen_sources = []
    for label, source in sources:
        frozen = output / label / 'program_0'
        clone_program_dir(src_dir=source, dst_dir=frozen)
        frozen_sources.append((label, frozen, source_fingerprint(frozen)))
    (output / 'comparison_plan.json').write_text(json.dumps({
        'seeds': confirmation_seeds, 'evaluation_protocol': confirmation_protocol,
        'selection_rule': 'Search incumbent and optional highest validation-score correct challenger',
        'programs': {label: code for label, _, code in frozen_sources},
    }, indent=2) + '\n')
    records = {}
    for label, frozen, code in frozen_sources:
        reused = next((name for name, record in records.items() if record['source_sha256'] == code), None)
        if reused is not None:
            records[label] = dict(records[reused], reused_from=reused)
            continue
        result_dir = output / label / 'evaluation'
        command = [sys.executable, str(task / 'evaluate.py'), '--program_path', str(frozen / 'src/program.py'),
                   '--results_dir', str(result_dir), '--holdout', str(holdout)]
        if confirmation_protocol:
            command.extend(['--protocol', str(output / 'confirmation_protocol.json')])
        else:
            command.extend(['--seeds', '11,19'])
        result = subprocess.run(command, env=dict(os.environ, TASK_SOURCE_DIR=str(task)),
                                capture_output=True, text=True, timeout=180, check=False)
        (output / label / 'stdout.log').write_text(result.stdout)
        (output / label / 'stderr.log').write_text(result.stderr)
        if result.returncode or source_fingerprint(frozen) != code:
            raise ValueError(f'{label} holdout evaluation failed; inspect {output / label}')
        records[label] = {'source_sha256': code, 'metrics': json.loads((result_dir / 'metrics.json').read_text())}
    summary = {'status': 'completed', 'execution_boundary': 'host', 'seeds': confirmation_seeds,
               'selected_search_score': manifest['combined_score'],
               'holdout_sha256': data_manifest['holdout_sha256'], 'programs': records,
               'note': 'Descriptive synthetic-bag pilot. No patient-level or clinical WSI claim; do not tune on holdout.'}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({name:record['metrics']['combined_score'] for name,record in records.items()}, indent=2))


if __name__ == '__main__':
    main()
