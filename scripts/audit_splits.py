"""Read-only exact/near-duplicate screening; no predictions or test metrics."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def hashes(path):
    with Image.open(path) as image:
        gray = image.convert("L")
        pixels = np.asarray(gray.resize((32, 32), Image.Resampling.LANCZOS), dtype=float)
        small = np.asarray(gray.resize((9, 8), Image.Resampling.LANCZOS), dtype=float)
        center = np.asarray(gray.resize((64, 64), Image.Resampling.BILINEAR),
                            dtype=float)[8:56, 8:56].flatten()
        center -= center.mean()
        center /= np.linalg.norm(center) + 1e-8
    # DCT-II low frequencies, excluding DC; implemented with numpy to avoid dependencies.
    basis = np.cos(np.pi / 32 * (np.arange(32) + 0.5)[None, :] * np.arange(8)[:, None])
    coefficients = (basis @ pixels @ basis.T).flatten()[1:]
    phash = sum(int(bit) << i for i, bit in enumerate(coefficients > np.median(coefficients)))
    dhash = sum(int(bit) << i for i, bit in enumerate((small[:, 1:] > small[:, :-1]).flatten()))
    return phash, dhash, center


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/split_audit"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    inventory = Path("results/kvasir_seg/inventory.csv")
    rows = list(csv.DictReader(inventory.open()))
    assignment = {}
    manifests = {}
    for split in ("train", "val", "test"):
        path = Path(f"data/splits/kvasir_{split}.csv")
        manifests[split] = sha(path)
        for row in csv.DictReader(path.open()):
            if row["id"] in assignment:
                raise ValueError(f"Overlapping split ID: {row['id']}")
            assignment[row["id"]] = split
    assert set(assignment) == {row["id"] for row in rows}
    computed = [hashes(row["image"]) for row in rows]
    pairs = []
    for i, first in enumerate(rows):
        for j in range(i + 1, len(rows)):
            second = rows[j]
            p = (computed[i][0] ^ computed[j][0]).bit_count()
            d = (computed[i][1] ^ computed[j][1]).bit_count()
            exact = first["image_pixel_sha256"] == second["image_pixel_sha256"]
            if exact or p <= 10 or d <= 8:
                pairs.append({
                    "id_a": first["id"], "id_b": second["id"], "phash_distance": p,
                    "dhash_distance": d, "exact": exact, "split_a": assignment[first["id"]],
                    "split_b": assignment[second["id"]],
                    "center_ncc": float(computed[i][2] @ computed[j][2]),
                })
    pairs.sort(key=lambda pair: -pair["center_ncc"])
    fields = ["id_a", "id_b", "phash_distance", "dhash_distance", "exact", "split_a", "split_b",
              "center_ncc"]
    with (args.output / "candidate_pairs.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(pairs)
    by_id = {row["id"]: row for row in rows}
    priority = [p for p in pairs if p["exact"] or p["center_ncc"] >= 0.85
                or p["phash_distance"] <= 4 or p["dhash_distance"] <= 3]
    with (args.output / "review_pairs.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(priority)
    # Review includes all splits and is independent of model output.
    for start in range(0, len(priority), 10):
        page = priority[start:start + 10]
        canvas = Image.new("RGB", (640, 190 * len(page)), "white")
        draw = ImageDraw.Draw(canvas)
        for index, pair in enumerate(page):
            for column, key in enumerate(("id_a", "id_b")):
                with Image.open(by_id[pair[key]]["image"]) as image:
                    image = image.convert("RGB")
                    image.thumbnail((310, 152))
                    canvas.paste(image, (column * 320, index * 190 + 35))
                draw.text((column * 320 + 2, index * 190 + 3), pair[key], fill="black")
            draw.text((2, index * 190 + 18),
                      f"pair {start + index}: p={pair['phash_distance']} d={pair['dhash_distance']} ncc={pair['center_ncc']:.3f}",
                      fill="black")
        canvas.save(args.output / f"review_{start // 10:02d}.jpg")
    report = {
        "images": len(rows), "split_counts": {s: list(assignment.values()).count(s) for s in manifests},
        "split_sha256": manifests, "inventory_sha256": sha(inventory),
        "screen": "63-bit pHash distance <=10 OR 64-bit dHash distance <=8 OR exact decoded pixels",
        "candidate_pairs": len(pairs),
        "review_rule": "Broad candidate AND (central 75% NCC >=0.85 OR pHash<=4 OR dHash<=3)",
        "review_pairs": len(priority),
        "cross_split_review_pairs": sum(p["split_a"] != p["split_b"] for p in priority),
        "cross_split_candidates": sum(p["split_a"] != p["split_b"] for p in pairs),
        "cross_split_exact": sum(p["exact"] and p["split_a"] != p["split_b"] for p in pairs),
        "limitation": "Heuristic screening can miss crops, viewpoint changes and same-patient frames. No patient independence claim.",
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
