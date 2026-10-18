"""Prepare a reproducible, small MIL pilot from the official PathMNIST archive.

Patch labels are used only here to construct synthetic bags. They are absent
from search.npz and from all model training inputs. No WSI/patient claim is made.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

DATA_URL = "https://zenodo.org/records/10519652/files/pathmnist.npz?download=1"
DATA_MD5 = "a8b06965200029087d5bd730944a56c1"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def image_features(images):
    """Fixed, label-free color/texture features; no learned image backbone."""
    x = images.astype(np.float32) / 255
    spatial = x.reshape(-1, 7, 4, 7, 4, 3).mean(axis=(2, 4)).reshape(len(x), -1)
    means = x.mean(axis=(1, 2))
    stds = x.std(axis=(1, 2))
    dx = np.abs(np.diff(x, axis=1)).mean(axis=(1, 2))
    dy = np.abs(np.diff(x, axis=2)).mean(axis=(1, 2))
    histogram = np.stack([(x >= i / 8).mean(axis=(1, 2)) for i in range(1, 8)], axis=1)
    return np.concatenate([spatial, means, stds, dx, dy, histogram.reshape(len(x), -1)], axis=1)


def unique_indices(images, seen):
    selected = []
    for index, im in enumerate(images):
        digest = hashlib.sha256(im.tobytes()).digest()
        if digest not in seen:
            selected.append(index)
            seen.add(digest)
    return np.asarray(selected, dtype=np.int64)


def make_bags(features, labels, ids, count, seed, *, extended):
    rng = np.random.default_rng(seed)
    positive = rng.permutation(np.flatnonzero(labels == 8))
    negative = rng.permutation(np.flatnonzero(labels != 8))
    pos_cursor = neg_cursor = 0
    bag_length = 64 if extended else 16
    bags, bag_labels, used_ids, witnesses = [], [], [], []
    for index in range(count):
        is_positive = index % 2 == 1
        n_positive = (1 + (index // 2) % 2) if is_positive else 0
        n_negative = bag_length - n_positive
        p = positive[pos_cursor:pos_cursor + n_positive]
        n = negative[neg_cursor:neg_cursor + n_negative]
        if len(p) != n_positive or len(n) != n_negative:
            raise ValueError("Insufficient unique patches; reduce the bag count")
        pos_cursor += n_positive
        neg_cursor += n_negative
        # Every added instance is non-tumor. The same positives remain in both views.
        base = np.concatenate([p, n[:16 - n_positive]])
        rng.shuffle(base)
        order = np.concatenate([base, n[16 - n_positive:]])
        bags.append(features[order])
        bag_labels.append(int(is_positive))
        used_ids.append(ids[order])
        witnesses.append(n_positive)
    return (np.stack(bags).astype(np.float32), np.asarray(bag_labels, dtype=np.float32),
            np.stack(used_ids), np.asarray(witnesses))


def prepare(archive: Path, task: Path, holdout: Path):
    if hashlib.md5(archive.read_bytes()).hexdigest() != DATA_MD5:
        raise ValueError("PathMNIST MD5 does not match the official Zenodo record")
    resource = task / "resources"
    resource.mkdir(parents=True, exist_ok=True)
    if (resource / "search.npz").exists() or holdout.exists():
        raise ValueError("Data already prepared; use a new output path to avoid changing a run")
    data = np.load(archive, allow_pickle=False)
    seen = set()
    arrays = {}
    counts = {}
    for split in ("train", "val", "test"):
        images, labels = data[f"{split}_images"], data[f"{split}_labels"].reshape(-1)
        indices = unique_indices(images, seen)
        # Batch extraction limits peak memory on the local machine.
        vectors = np.concatenate([image_features(images[indices[i:i + 2048]])
                                  for i in range(0, len(indices), 2048)])
        arrays[split] = [vectors, labels[indices], indices]
        counts[split] = {"official": len(images), "exact_unique_after_prior_splits": len(indices)}
    with threadpool_limits(limits=4):
        scaler = StandardScaler().fit(arrays["train"][0])
        train = scaler.transform(arrays["train"][0])
        pca = PCA(n_components=48, random_state=17, svd_solver="randomized").fit(train)
        for split in arrays:
            arrays[split][0] = pca.transform(scaler.transform(arrays[split][0])).astype(np.float32)
    train_x, train_y, train_ids, _ = make_bags(*arrays["train"], 600, 17, extended=False)
    val_x, val_y, val_ids, val_witnesses = make_bags(*arrays["val"], 100, 29, extended=True)
    test_x, test_y, test_ids, test_witnesses = make_bags(*arrays["test"], 80, 41, extended=True)
    # Do not put patch IDs or patch labels into the model-visible feature archive.
    np.savez_compressed(resource / "search.npz", train_x=train_x, train_y=train_y,
                        val_x=val_x, val_y=val_y)
    holdout.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(holdout, test_x=test_x, test_y=test_y, witnesses=test_witnesses)
    audit = holdout.with_suffix(".construction.npz")
    np.savez_compressed(audit, train_ids=train_ids, val_ids=val_ids, test_ids=test_ids,
                        val_witnesses=val_witnesses, test_witnesses=test_witnesses,
                        scaler_mean=scaler.mean_, scaler_scale=scaler.scale_,
                        pca_mean=pca.mean_, pca_components=pca.components_)
    manifest = {
        "dataset": "PathMNIST 28", "source": DATA_URL, "license": "CC BY 4.0",
        "source_md5": DATA_MD5, "source_sha256": sha256(archive), "counts": counts,
        "feature_recipe": "fixed color/texture -> train-only StandardScaler -> train-only PCA(48)",
        "feature_training_uses_patch_labels": False, "model_training_uses_patch_labels": False,
        "weak_supervision": "synthetic bag label: at least one tumor patch (class 8)",
        "train_bags": 600, "validation_bags": 100, "holdout_bags": 80,
        "base_length": 16, "diluted_length": 64, "positive_witness_count": [1, 2],
        "patch_reuse_between_distinct_bags_within_each_split": False,
        "paired_views_reuse_same_base_bag": True, "patient_disjointness_verified": False,
        "search_sha256": sha256(resource / "search.npz"), "holdout_sha256": sha256(holdout),
        "audit_sha256": sha256(audit), "scope": "development pilot; synthetic bags, not clinical WSI evaluation",
    }
    (resource / "data_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--task", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--holdout", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.archive, args.task, args.holdout)

