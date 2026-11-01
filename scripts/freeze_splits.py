"""Freeze audited manifests and training-only area tertiles before model fitting."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    destination = Path("data/splits/freeze.json")
    if destination.exists():
        raise SystemExit("Freeze exists; refusing to overwrite")
    audit = json.loads(Path("results/split_audit_after/summary.json").read_text())
    manifests, assignments = {}, {}
    for split in ("train", "val", "test"):
        path = Path(f"data/splits/kvasir_{split}.csv")
        rows = list(csv.DictReader(path.open()))
        assert sha(path) == audit["split_sha256"][split]
        manifests[split] = {"path": str(path), "sha256": sha(path), "count": len(rows)}
        for row in rows:
            assert row["id"] not in assignments
            assignments[row["id"]] = split
    pairs_path = Path("data/splits/near_duplicate_candidates.csv")
    pairs = list(csv.DictReader(pairs_path.open()))
    for pair in pairs:
        assert assignments[pair["id_a"]] == assignments[pair["id_b"]]
    assert audit["cross_split_exact"] == 0 and audit["cross_split_review_pairs"] == 0
    inventory = Path("results/kvasir_seg/inventory.csv")
    rows = list(csv.DictReader(inventory.open()))
    assert set(assignments) == {row["id"] for row in rows}
    fractions = [float(row["foreground_fraction"]) for row in rows if assignments[row["id"]] == "train"]
    lock = {
        "version": 1, "manifests": manifests,
        "inventory_sha256": sha(inventory), "similarity_pairs_sha256": sha(pairs_path),
        "similarity_grouping": "All 40 high-priority screening pairs held together conservatively; not all are true duplicates.",
        "size_thresholds": np.quantile(fractions, [1 / 3, 2 / 3], method="linear").tolist(),
        "size_definition": "Native mask foreground fraction; training-only tertiles, small <q1, medium [q1,q2), large >=q2.",
        "mask_threshold": 128, "prediction_threshold": 0.5,
        "boundary_metric": "Inner 8-connected contour, outside image background, Chebyshev distance tolerance max(1, floor(0.005*diagonal+0.5)) native pixels.",
        "test_evaluated": False, "formal_training_protocol_frozen": False,
        "limitation": "Image-level split; heuristic near-duplicate screening cannot establish patient/video independence.",
    }
    destination.write_text(json.dumps(lock, indent=2) + "\n")
    print(json.dumps(lock, indent=2))


if __name__ == "__main__":
    main()
