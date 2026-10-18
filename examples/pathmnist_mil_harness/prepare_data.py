"""Create a fresh frozen task from the official archive without opening old holdouts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

DATA_URL = "https://zenodo.org/records/10519652/files/pathmnist.npz?download=1"
DATA_MD5 = "a8b06965200029087d5bd730944a56c1"
OLD_BAGS = {"train": (600, 17, 16), "val": (100, 29, 64), "test": (80, 41, 64)}
NEW_SEEDS = {"val": 829, "test": 941}
TEMPLATE = Path(__file__).resolve().parent


def digest(path, algorithm="sha256"):
    result = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            result.update(block)
    return result.hexdigest()


def image_features(images):
    """Same fixed, label-free color/texture recipe used by the earlier pilot."""
    x = images.astype(np.float32) / 255
    spatial = x.reshape(-1, 7, 4, 7, 4, 3).mean(axis=(2, 4)).reshape(len(x), -1)
    parts = [spatial, x.mean((1, 2)), x.std((1, 2)),
             np.abs(np.diff(x, axis=1)).mean((1, 2)), np.abs(np.diff(x, axis=2)).mean((1, 2))]
    histogram = np.stack([(x >= i / 8).mean((1, 2)) for i in range(1, 8)], axis=1)
    return np.concatenate([*parts, histogram.reshape(len(x), -1)], axis=1)


def unique_indices(images, seen):
    selected = []
    for index, image in enumerate(images):
        fingerprint = hashlib.sha256(image.tobytes()).digest()
        if fingerprint not in seen:
            seen.add(fingerprint)
            selected.append(index)
    return np.asarray(selected, dtype=np.int64)


def bag_indices(labels, count, seed, length):
    """Exact historical sampling procedure, including shuffle RNG consumption."""
    rng = np.random.default_rng(seed)
    positive = rng.permutation(np.flatnonzero(labels == 8))
    negative = rng.permutation(np.flatnonzero(labels != 8))
    pos_cursor = neg_cursor = 0
    orders, targets, witnesses = [], [], []
    for index in range(count):
        is_positive = index % 2 == 1
        n_positive = (1 + (index // 2) % 2) if is_positive else 0
        n_negative = length - n_positive
        pos = positive[pos_cursor:pos_cursor + n_positive]
        neg = negative[neg_cursor:neg_cursor + n_negative]
        if len(pos) != n_positive or len(neg) != n_negative:
            raise ValueError("insufficient unused patches; lower requested bag count")
        pos_cursor += n_positive
        neg_cursor += n_negative
        base = np.concatenate([pos, neg[:16 - n_positive]])
        rng.shuffle(base)
        orders.append(np.concatenate([base, neg[16 - n_positive:]]))
        targets.append(int(is_positive))
        witnesses.append(n_positive)
    return np.stack(orders), np.asarray(targets, dtype=np.float32), np.asarray(witnesses)


def fresh_orders(labels, used_orders, requested, seed):
    available = np.setdiff1d(np.arange(len(labels)), np.unique(used_orders), assume_unique=True)
    remaining = labels[available]
    # Each block of four 64-patch bags uses exactly 3 tumor and 253 background patches.
    capacity = 4 * min(np.count_nonzero(remaining == 8) // 3,
                       np.count_nonzero(remaining != 8) // 253)
    count = int(min(requested, capacity) // 4 * 4)
    if count < 8:
        raise ValueError("too few unused patches for a balanced fresh confirmation")
    orders, targets, witnesses = bag_indices(remaining, count, seed, 64)
    return available[orders], targets, witnesses, {"available_patches": len(available),
            "available_positive": int(np.count_nonzero(remaining == 8)),
            "available_negative": int(np.count_nonzero(remaining != 8)),
            "capacity_bags": int(capacity), "requested_bags": requested, "actual_bags": count}


def write_contract(task):
    contract = json.loads((TEMPLATE / "task_contract.json").read_text())
    contract["canonical_task"]["sha256"] = digest(task / "task.md")
    contract["provenance"]["artifacts"] = [
        {"role": role, "path": path, "sha256": digest(task / path), "source": source}
        for role, path, source in [
            ("development", "resources/search/data.npz", DATA_URL),
            ("sealed_test", "resources/final/data.npz", DATA_URL),
            ("construction_audit", "resources/final/construction.npz", "deterministic raw-archive reconstruction"),
            ("data_manifest", "data_manifest.json", "deterministic fresh-split preparation"),
            ("dependency_lock", "requirements.txt", "pinned task dependencies"),
            ("evaluator_helper", "evaluator.py", "frozen host evaluation and metric implementation"),
            ("evaluator_helper", "reference.py", "frozen ordinary-LME and classical pooling controls"),
            ("evaluator_helper", "protocol.py", "frozen training and evaluation protocol"),
        ]
    ]
    (task / "task_contract.json").write_text(json.dumps(contract, indent=2) + "\n")


def prepare(archive, task, *, validation_bags=56, test_bags=32):
    archive, task = Path(archive), Path(task)
    if digest(archive, "md5") != DATA_MD5:
        raise ValueError("archive MD5 differs from the official PathMNIST 28 archive")
    if task.exists():
        raise ValueError("task output already exists; use a new path to preserve frozen evidence")
    if validation_bags < 8 or test_bags < 8:
        raise ValueError("request at least eight bags per fresh split")
    shutil.copytree(TEMPLATE, task, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "resources"))
    for plane in ("search", "final"):
        (task / "resources" / plane).mkdir(parents=True)
    seen, arrays, counts = set(), {}, {}
    with np.load(archive, allow_pickle=False) as raw:
        for split in ("train", "val", "test"):
            images, labels = raw[f"{split}_images"], raw[f"{split}_labels"].reshape(-1)
            ids = unique_indices(images, seen)
            vectors = np.concatenate([image_features(images[ids[i:i + 2048]])
                                      for i in range(0, len(ids), 2048)])
            arrays[split] = (vectors, labels[ids], ids)
            counts[split] = {"official": len(images), "exact_unique_after_prior_splits": len(ids)}
    old, fresh = {}, {}
    for split, (_, labels, _) in arrays.items():
        count, seed, length = OLD_BAGS[split]
        old[split] = bag_indices(labels, count, seed, length)
        if split != "train":
            requested = validation_bags if split == "val" else test_bags
            orders, targets, witnesses, capacity = fresh_orders(labels, old[split][0], requested, NEW_SEEDS[split])
            fresh[split] = (orders, targets, witnesses)
            counts[split]["fresh"] = capacity
    with threadpool_limits(limits=4):
        scaler = StandardScaler().fit(arrays["train"][0])
        pca = PCA(n_components=48, random_state=17, svd_solver="randomized").fit(
            scaler.transform(arrays["train"][0]))
        vectors = {split: pca.transform(scaler.transform(value[0])).astype(np.float32)
                   for split, value in arrays.items()}
    payload, audit = {}, {}
    for split in ("train", "val", "test"):
        orders, targets, witnesses = old[split] if split == "train" else fresh[split]
        row_order = np.random.default_rng({"train": 997, "val": 1009, "test": 1013}[split]).permutation(len(orders))
        orders, targets, witnesses = orders[row_order], targets[row_order], witnesses[row_order]
        payload[split + "_x"], payload[split + "_y"] = vectors[split][orders], targets
        audit[split + "_ids"] = arrays[split][2][orders]
        audit[split + "_witnesses"] = witnesses
        audit["previous_" + split + "_ids"] = arrays[split][2][old[split][0]]
        if split != "train" and np.intersect1d(audit[split + "_ids"], audit["previous_" + split + "_ids"]).size:
            raise AssertionError("fresh split overlaps a previously used patch")
    audit.update(scaler_mean=scaler.mean_, scaler_scale=scaler.scale_,
                 pca_mean=pca.mean_, pca_components=pca.components_)
    search_path, final_path = task / "resources/search/data.npz", task / "resources/final/data.npz"
    np.savez_compressed(search_path, **{key: value for key, value in payload.items() if not key.startswith("test_")})
    np.savez_compressed(final_path, **payload)
    np.savez_compressed(task / "resources/final/construction.npz", **audit)
    manifest = {"schema_version": 1, "dataset": "PathMNIST 28", "source": DATA_URL,
                "license": "CC BY 4.0", "source_md5": DATA_MD5, "source_sha256": digest(archive),
                "counts": counts, "historical_sampling": {split: {"bags": value[0], "seed": value[1], "length": value[2]}
                                                          for split, value in OLD_BAGS.items()},
                "fresh_sampling_seeds": NEW_SEEDS, "feature_recipe": "fixed color/texture, train-only StandardScaler, train-only PCA(48)",
                "feature_fit_scope": "all exact-deduplicated official train patches; no patch labels used",
                "train_bags_reused": True, "previous_validation_and_test_patches_excluded": True,
                "old_holdout_files_read": False, "paired_bag_sizes": [16, 32, 64],
                "paired_views_preserve_tumor_witnesses": True, "positive_witnesses": [1, 2],
                "patient_disjointness_verified": False, "external_generalization_evaluated": False,
                "scope": "fresh patch-disjoint within-dataset synthetic bag confirmation after prior holdout exposure",
                "search_sha256": digest(search_path), "final_sha256": digest(final_path),
                "construction_sha256": digest(task / "resources/final/construction.npz")}
    (task / "data_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    write_contract(task)
    return {"task": str(task), "validation_bags": len(payload["val_y"]), "test_bags": len(payload["test_y"]),
            "search_sha256": manifest["search_sha256"], "final_sha256": manifest["final_sha256"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--validation-bags", type=int, default=56)
    parser.add_argument("--test-bags", type=int, default=32)
    args = parser.parse_args()
    print(json.dumps(prepare(args.archive, args.task, validation_bags=args.validation_bags,
                             test_bags=args.test_bags), indent=2))
