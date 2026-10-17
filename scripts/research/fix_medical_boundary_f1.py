#!/usr/bin/env python3
"""Correct directional matching counts in the pilot's tolerance-based boundary F1.

Definition reference: DAVIS official evaluator, f_measure precision/recall:
https://github.com/davisvideochallenge/davis2017-evaluation/blob/master/davis2017/metrics.py
The existing boundary extractor, tolerance neighbourhood and pooling remain fixed.
"""
from __future__ import annotations
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys

NAMES = ('boundary_f1_counts', 'macro_boundary_f1')
COUNTS = '''def boundary_f1_counts(pred_class, labels, c):
    # Tolerance matching can be many-to-one. Precision and recall need separate
    # matched-boundary counts, pooled over slices before class/patient means.
    matched_pred = total_pred = matched_gt = total_gt = 0
    for s in range(labels.shape[0]):
        gt = labels[s] == c
        pr = pred_class[s] == c
        if not gt.any() and not pr.any():
            continue
        gt_boundary = gt & ~ndimage.binary_erosion(gt, structure=SQUARE, border_value=0)
        pr_boundary = pr & ~ndimage.binary_erosion(pr, structure=SQUARE, border_value=0)
        gt_dilated = ndimage.binary_dilation(gt_boundary, structure=SQUARE)
        pr_dilated = ndimage.binary_dilation(pr_boundary, structure=SQUARE)
        matched_pred += int(np.logical_and(pr_boundary, gt_dilated).sum())
        total_pred += int(pr_boundary.sum())
        matched_gt += int(np.logical_and(gt_boundary, pr_dilated).sum())
        total_gt += int(gt_boundary.sum())
    return matched_pred, total_pred, matched_gt, total_gt'''
MACRO = '''def macro_boundary_f1(pred_class, labels):
    per_class = []
    for c in FOREGROUND_CLASSES:
        if not (labels == c).any():
            continue  # Preserve the declared GT-presence inclusion rule.
        matched_pred, total_pred, matched_gt, total_gt = boundary_f1_counts(pred_class, labels, c)
        precision = matched_pred / total_pred if total_pred else 0.0
        recall = matched_gt / total_gt if total_gt else 0.0
        denom = precision + recall
        per_class.append(0.0 if denom == 0 else float(2.0 * precision * recall / denom))
    return float(np.mean(per_class, dtype=np.float64)) if per_class else 0.0'''


def metric_ast_digest(source: str) -> str:
    nodes = {n.name: n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name in NAMES}
    if set(nodes) != set(NAMES):
        raise ValueError('Expected both boundary F1 functions')
    tree = ast.Module(body=[nodes[name] for name in NAMES], type_ignores=[])
    return hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()


EXPECTED_AST = metric_ast_digest(COUNTS + '\n\n' + MACRO + '\n')


def corrected_source(source: str) -> str:
    if metric_ast_digest(source) == EXPECTED_AST:
        return source
    nodes = {n.name: n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name in NAMES}
    old_counts, old_macro = (ast.get_source_segment(source, nodes[name]) for name in NAMES)
    if 'return tp, fp, fn' not in old_counts or 'denom = 2 * tp + fp + fn' not in old_macro:
        raise ValueError('Unexpected metric implementation; refusing an unreviewed replacement')
    lines = source.splitlines(keepends=True)
    for name, replacement in sorted(zip(NAMES, (COUNTS, MACRO)), key=lambda item: nodes[item[0]].lineno, reverse=True):
        node = nodes[name]
        lines[node.lineno - 1:node.end_lineno] = [replacement + '\n']
    result = ''.join(lines)
    compile(result, '<repaired-evaluator>', 'exec')
    assert metric_ast_digest(result) == EXPECTED_AST
    return result


def repair(task: Path, output: Path) -> dict:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.task_drafting.contract_sync import synchronize_declared_digests
    from src.investigation.task_contract import load_task_contract
    canonical_sha = hashlib.sha256((task / 'task.md').read_bytes()).hexdigest()
    contract_path = task / 'task_contract.json'
    original = json.loads(contract_path.read_text())
    # Build both replacements before making any task edits.
    replacements = {name: corrected_source((task / name).read_text()) for name in ('evaluate.py', 'evaluate_final.py')}
    output.mkdir(mode=0o700)  # Refuse to overwrite any prior correction receipt.
    (output / 'before').mkdir(mode=0o700)
    hashes = {}
    for name in ('evaluate.py', 'evaluate_final.py', 'task_contract.json'):
        shutil.copy2(task / name, output / 'before' / name)
    for name, source in replacements.items():
        hashes[name] = {'before': hashlib.sha256((task / name).read_bytes()).hexdigest(),
                        'after': hashlib.sha256(source.encode()).hexdigest()}
        (task / name).write_text(source)
    synchronize_declared_digests(contract_path, canonical_task=task / 'task.md')
    current = json.loads(contract_path.read_text())
    expected = copy.deepcopy(original)
    for artifact in expected['provenance']['artifacts']:
        if artifact['path'] in hashes:
            artifact['sha256'] = hashes[artifact['path']]['after']
    if current != expected or hashlib.sha256((task / 'task.md').read_bytes()).hexdigest() != canonical_sha:
        raise ValueError('Correction changed protocol fields beyond evaluator provenance hashes')
    load_task_contract(contract_path)
    receipt = {'status':'fixed_pending_full_validation','canonical_task_sha256':canonical_sha,
               'expected_metric_ast_sha256':EXPECTED_AST,'files':hashes,
               'semantic_change':'Separate precision-side and recall-side tolerance matches; preserve metric weight, boundary extractor, tolerance and class/patient pooling',
               'reference':'https://github.com/davisvideochallenge/davis2017-evaluation/blob/master/davis2017/metrics.py',
               'required_next':'Rerun contract, isolated smoke, alignment and full production baseline qualification'}
    (output / 'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(repair(args.task,args.output)))


if __name__=='__main__':
    main()
