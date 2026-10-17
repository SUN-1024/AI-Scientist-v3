"""Independent geometric regressions for the pilot boundary metric."""
from pathlib import Path
import ast
import os
import runpy
try:
    import numpy as np
    from scipy import ndimage
except ImportError:
    import pytest
    pytest.skip('Scientific metric checks require the medical runtime NumPy/SciPy', allow_module_level=True)

ROOT = Path(__file__).resolve().parents[1]
M = runpy.run_path(str(ROOT / 'scripts/research/fix_medical_boundary_f1.py'))
OLD = '''# immutable prefix\ndef boundary_f1_counts(pred_class, labels, c):
    tp = fp = fn = 0
    for s in range(labels.shape[0]):
        gt = labels[s] == c
        pr = pred_class[s] == c
        if not gt.any() and not pr.any():
            continue
        gb = gt & ~ndimage.binary_erosion(gt, structure=SQUARE, border_value=0)
        pb = pr & ~ndimage.binary_erosion(pr, structure=SQUARE, border_value=0)
        gd = ndimage.binary_dilation(gb, structure=SQUARE)
        pd = ndimage.binary_dilation(pb, structure=SQUARE)
        tp += int((pb & gd).sum()); fp += int((pb & ~gd).sum()); fn += int((gb & ~pd).sum())
    return tp, fp, fn

def macro_boundary_f1(pred_class, labels):
    vals = []
    for c in FOREGROUND_CLASSES:
        if not (labels == c).any(): continue
        tp, fp, fn = boundary_f1_counts(pred_class, labels, c)
        denom = 2 * tp + fp + fn
        vals.append(0. if not denom else 2.*tp/denom)
    return float(np.mean(vals)) if vals else 0.
# immutable suffix\n'''


def namespace(source):
    ns = {'np':np,'ndimage':ndimage,'SQUARE':np.ones((3,3),dtype=bool),'FOREGROUND_CLASSES':(1,2,3)}
    nodes = [n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name in M['NAMES']]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<metric-under-test>','exec'),ns)
    return ns


def fixed():
    path = os.environ.get('MEDICAL_BOUNDARY_EVALUATOR_PATH')
    source = Path(path).read_text() if path else M['corrected_source'](OLD)
    assert M['metric_ast_digest'](source) == M['EXPECTED_AST']
    return namespace(source)['macro_boundary_f1']


def masks():
    a=np.zeros((1,12,12),dtype=np.uint8);a[:,2:9,2:9]=1
    b=np.zeros_like(a);b[:,1:2,3:8]=1
    return a,b


def reference_boundary(mask):
    # Explicit neighbour comparisons, independent of SciPy erosion/dilation.
    padded=np.pad(mask,1,constant_values=False)
    points=[]
    for y,x in np.argwhere(mask):
        if not padded[y:y+3,x:x+3].all():points.append((y,x))
    return np.asarray(points,dtype=int).reshape(-1,2)


def oracle(pred,gt):
    pm=pn=gm=gn=0
    for p,g in zip(pred,gt):
        a,b=reference_boundary(p==1),reference_boundary(g==1)
        pn+=len(a);gn+=len(b)
        if len(a) and len(b):
            distance=np.abs(a[:,None,:]-b[None,:,:]).max(2)
            pm+=int((distance.min(1)<=1).sum());gm+=int((distance.min(0)<=1).sum())
    precision=pm/pn if pn else 0.;recall=gm/gn if gn else 0.
    return 2*precision*recall/(precision+recall) if precision+recall else 0.


def test_original_defect_reproduces():
    a,b=masks();f=namespace(OLD)['macro_boundary_f1']
    assert abs(f(a,b)-f(b,a))>.01


def test_fixed_metric_is_symmetric_on_counterexample():
    a,b=masks();f=fixed()
    assert abs(f(a,b)-14/31)<1e-12
    assert abs(f(b,a)-14/31)<1e-12


def test_exact_shift_empty_and_disjoint():
    f=fixed();g=np.zeros((1,12,12),dtype=np.uint8);g[:,3:7,3:7]=1
    p=np.zeros_like(g);p[:,4:8,3:7]=1
    q=np.zeros_like(g);q[:,0:1,0:1]=1
    assert f(g,g)==1.;assert f(p,g)==1.;assert f(np.zeros_like(g),g)==0.;assert f(q,g)==0.


def test_slice_pooling_matches_independent_geometry_oracle():
    a,b=masks();p=np.concatenate((a,a));g=np.concatenate((b,a));f=fixed()
    assert abs(f(p,g)-oracle(p,g))<1e-12
    assert abs(f(p,g)-(f(a,b)+f(a,a))/2)>1e-3


def test_random_masks_match_oracle():
    rng=np.random.default_rng(77);f=fixed()
    for _ in range(20):
        a=(rng.random((2,7,9))>.65).astype(np.uint8);b=(rng.random((2,7,9))>.75).astype(np.uint8)
        a[:,0,0]=b[:,0,0]=1
        assert abs(f(a,b)-oracle(a,b))<1e-12
        assert abs(f(a,b)-f(b,a))<1e-12


def test_gt_absent_class_rule_preserved():
    f=fixed();g=np.zeros((1,12,12),dtype=np.uint8);g[:,3:7,3:7]=1
    p=g.copy();p[:,0,0]=2
    assert f(p,g)==1.
    assert f(p,np.zeros_like(g))==0.


def test_patch_is_narrow_and_idempotent():
    s=M['corrected_source'](OLD)
    assert s.startswith('# immutable prefix\n') and s.endswith('# immutable suffix\n')
    assert M['corrected_source'](s)==s
    try:M['corrected_source'](OLD.replace('denom = 2 * tp + fp + fn','denom = tp + fp + fn'))
    except ValueError:pass
    else:raise AssertionError('Unexpected metric implementation must not be replaced')
