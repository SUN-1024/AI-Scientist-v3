#!/usr/bin/env python3
"""Read one medical job's process, container, exit and latest saved-draft state."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import re
import stat
import subprocess

JOB_LABEL = 'ai.agentic-scientist.research-job'


def journal_container_ids(directory: Path, job: str) -> set[str]:
    # A trusted host-only journal covers already-running legacy containers.
    path = directory / 'owned-container-ids.json'
    if not path.exists() and not path.is_symlink():
        return set()
    metadata = path.lstat()
    if (not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.getuid()
            or metadata.st_mode & 0o077 or metadata.st_size > 16384):
        raise ValueError('Unsafe owned-container journal')
    payload = json.loads(path.read_text())
    ids = payload.get('container_ids')
    if (payload.get('job_id') != job or not isinstance(ids, list) or len(ids) > 64
            or not all(isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) for value in ids)):
        raise ValueError('Invalid owned-container journal')
    return set(ids)


def within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (OSError, ValueError):
        return False


def latest_draft(result_root: Path) -> Path | None:
    direct = result_root / 'task_drafting'
    choices = [direct, *direct.glob('_task_drafting/draft_*')]
    choices = [p for p in choices if (p / 'task/task.md').is_file()
               and (p / '_task_design/design_receipt.json').is_file()]
    if not choices:
        return None
    return max(choices, key=lambda p: max(
        f.stat().st_mtime_ns for f in (p / 'task').rglob('*')
        if f.is_file() and not f.is_symlink()))


def inspect_job(base: Path, job: str, *, process_ids=None, container_ids=None) -> dict:
    if not re.fullmatch(r'\d{8}_\d{6}_[0-9a-f]{8}', job):
        raise ValueError('Invalid job identifier')
    directory, results = base / 'jobs' / job, base / 'results' / job
    if not directory.is_dir() or not results.is_dir():
        raise ValueError('Recorded job directory is missing')
    if process_ids is None:
        process_ids = []
        for proc in Path('/proc').glob('[0-9]*'):
            try:
                if int(proc.name) == os.getpid() or proc.stat().st_uid != os.getuid():
                    continue
                cwd = (proc / 'cwd').resolve(strict=True)
                if within(cwd, directory) or within(cwd, results):
                    process_ids.append(int(proc.name))
            except (OSError, ValueError):
                continue
    if container_ids is None:
        # Do not print Docker environment data (it may contain provider secrets).
        ids = subprocess.check_output(['docker', 'ps', '-q'], text=True).split()
        containers = json.loads(subprocess.check_output(['docker', 'inspect', *ids], text=True)) if ids else []
        journal = journal_container_ids(directory, job)
        container_ids = [c['Id'][:12] for c in containers
                         if c['Id'] in journal
                         or (c.get('Config', {}).get('Labels') or {}).get(JOB_LABEL) == job
                         or any(within(Path(m['Source']), directory) or within(Path(m['Source']), results)
                                for m in c.get('Mounts', []) if m.get('Source'))]
    receipt = directory / 'exit-code'
    exit_code = int(receipt.read_text().strip()) if receipt.is_file() else None
    draft = latest_draft(results)
    return {'job_id': job, 'running': bool(process_ids or container_ids),
            'process_ids': process_ids, 'container_ids': container_ids,
            'exit_code': exit_code, 'source_draft': str(draft) if draft else None,
            'published_task': str(results / 'task'),
            'has_qualification_receipt': (results / 'task/.agentic_task_qualification.json').is_file()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('base', type=Path)
    parser.add_argument('job')
    args = parser.parse_args()
    print(json.dumps(inspect_job(args.base, args.job)))


if __name__ == '__main__':
    main()
