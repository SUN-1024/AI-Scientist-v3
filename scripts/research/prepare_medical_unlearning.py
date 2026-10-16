#!/usr/bin/env python3
"""Build an independent development packet from the existing audited TRAIN split.
Run with the source project's Python on server40. Never writes to the source.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--destination', type=Path, required=True)
    a = ap.parse_args()
    source, dest = a.source.resolve(), a.destination.resolve()
    if dest == source or source in dest.parents:
        raise ValueError('Destination must be outside the source project')
    resources = dest / 'resources'
    if resources.exists():
        raise FileExistsError('Refuse to overwrite an existing research packet')
    manifest = json.loads((source/'manifests/datasets/acdc.json').read_text())
    splits = json.loads((source/'manifests/splits/acdc.json').read_text())['splits']
    assert manifest['data_qc_passed']
    allowed = set(splits['train'])
    assert not allowed.intersection(set(splits['test']) | set(splits['validation']) | set(splits['calibration']))
    records = {p['patient_id']: p for p in manifest['patients'] if p['patient_id'] in allowed}
    assert set(records) == allowed and len(allowed) == 78
    rng = np.random.default_rng(20261016)
    planes = {'search': [], 'final': []}
    for group in sorted({p['group'] for p in records.values()}):
        ids = sorted(p for p in records if records[p]['group'] == group)
        rng.shuffle(ids)
        # Alternate the extra patient to balance plane sizes across pathology groups.
        offset = int(len(planes['search']) > len(planes['final']))
        for i, pid in enumerate(ids):
            planes[('search','final')[(i+offset)%2]].append(pid)
    raw_hashes, aliases, processed_hashes = {}, {}, {}
    for plane, ids in planes.items():
        raw_hashes[plane] = set()
        aliases[plane] = set()
        processed_hashes[plane] = set()
        for pid in ids:
            aliases[plane].update(records[pid].get('source_patient_ids', [pid]))
            raw_hashes[plane].update(ph['voxel_sha256'] for ph in records[pid]['phases'])
    assert not raw_hashes['search'] & raw_hashes['final']
    assert not aliases['search'] & aliases['final']
    resources.mkdir(parents=True)
    receipt = {'kind':'development_only', 'source_project':str(source),
               'source_git_head':subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip(),
               'source_manifest_sha256':sha(source/'manifests/datasets/acdc.json'),
               'source_split_sha256':sha(source/'manifests/splits/acdc.json'),
               'original_test_read':False, 'source_modified':False,
               'preprocessing':'all ED/ES axial slices; per-volume z-score; bilinear image/nearest mask to 128x128; low-resolution 2D pilot only',
               'planes':{}}
    torch.set_num_threads(2)
    for plane, ids in planes.items():
        root = resources/plane
        root.mkdir()
        patients = []
        for pid in sorted(ids):
            p = records[pid]
            xs, ys, phase_lengths = [], [], []
            for ph in p['phases']:
                assert sha(ph['image']) == ph['image_sha256']
                assert sha(ph['label']) == ph['label_sha256']
                x = np.asarray(nib.load(ph['image']).dataobj, dtype=np.float32)
                y = np.asarray(nib.load(ph['label']).dataobj, dtype=np.int64)
                # Content hashes are also checked from current raw bytes, before resampling.
                raw = np.asarray(nib.load(ph['image']).dataobj)
                assert hashlib.sha256(raw.tobytes()).hexdigest() == ph['voxel_sha256']
                x = (x-x.mean()) / max(float(x.std()),1e-6)
                xt = torch.from_numpy(x.transpose(2,0,1).copy())[:,None]
                yt = torch.from_numpy(y.transpose(2,0,1).copy())[:,None].float()
                xs.append(F.interpolate(xt,size=(128,128),mode='bilinear',align_corners=False).numpy().astype(np.float32))
                ys.append(F.interpolate(yt,size=(128,128),mode='nearest')[:,0].numpy().astype(np.uint8))
                phase_lengths.append(len(xt))
            x,y = np.concatenate(xs),np.concatenate(ys)
            assert x.shape[0] == y.shape[0] and np.isfinite(x).all()
            assert set(np.unique(y)).issubset({0,1,2,3})
            content_hash = hashlib.sha256(x.tobytes()).hexdigest()
            processed_hashes[plane].add(content_hash)
            path=root/(pid+'.npz')
            np.savez_compressed(path,images=x,labels=y)
            patients.append({'patient_id':pid,'group':p['group'],'file':path.name,
                             'sha256':sha(path),'image_content_sha256':content_hash,
                             'source_aliases':p.get('source_patient_ids',[pid]),
                             'raw_voxel_sha256':[ph['voxel_sha256'] for ph in p['phases']],
                             'phase_lengths':phase_lengths,'slices':len(x)})
        # 10 unseen evaluation patients (two per pathology); train on all remaining.
        heldout=[]
        for group in sorted({p['group'] for p in patients}):
            members=[p['patient_id'] for p in patients if p['group']==group]
            heldout.extend(members[-2:])
        train=sorted(set(ids)-set(heldout))
        # One patient per pathology: a fixed patient-level deletion request.
        forget=[next(pid for pid in train if records[pid]['group']==g)
                for g in sorted({p['group'] for p in patients})]
        packet={'schema_version':1,'plane':plane,'evidence_scope':'development_only',
                'patients':patients,'train':train,'forget':forget,
                'retain':sorted(set(train)-set(forget)),'heldout':sorted(heldout),
                'class_labels':{'0':'background','1':'RV','2':'myocardium','3':'LV'},
                'seed':1701 if plane=='search' else 2701}
        (root/'manifest.json').write_text(json.dumps(packet,indent=2)+'\n')
        receipt['planes'][plane]={'patients':len(ids),'train':len(train),'forget':len(forget),
                                 'heldout':len(heldout),'manifest_sha256':sha(root/'manifest.json')}
    assert not processed_hashes['search'] & processed_hashes['final']
    upstream=dest/'upstream'/'puseg'
    upstream.mkdir(parents=True)
    receipt['source_files']={}
    for name in ['models.py','losses.py','unlearning.py']:
        src=source/'src/puseg'/name
        shutil.copyfile(src,upstream/name)
        receipt['source_files'][name]=sha(src)
    (dest/'upstream'/'README.md').write_text('Read-only source snapshot. models.py and losses.py can be reused. unlearning.py is a reference excerpt with project-specific imports; port its method equations, do not run its original project controllers. Original project and its test cohort must remain untouched.\n')
    (resources/'preparation_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))

if __name__ == '__main__':
    main()
