"""Validate a saved model; held-out evaluation requires a matching frozen protocol."""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch.utils.data import DataLoader

from src.dataset import ROOT, PolypDataset
from src.experiment import evaluate, load_model, seed_everything, sha, verify_lock, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--manifest", type=Path, help="Defaults to the checkpoint's validation split")
    parser.add_argument("--frozen-protocol", type=Path, help="Required for every non-validation manifest")
    args = parser.parse_args()
    device = torch.device(args.device)
    model, saved = load_model(args.checkpoint, device)
    config, lock = saved["config"], saved["split_lock"]
    seed_everything(config["seed"], config["cpu_threads"])
    if verify_lock(config) != lock:
        raise ValueError("Current split lock differs from checkpoint")
    manifest = args.manifest or ROOT / config["val_manifest"]
    if sha(manifest) != lock["manifests"]["val"]["sha256"]:
        if not args.frozen_protocol:
            raise ValueError("Held-out evaluation is sealed until --frozen-protocol is provided")
        protocol = json.loads(args.frozen_protocol.read_text())
        expected = {
            "status": "frozen",
            "checkpoint_sha256": sha(args.checkpoint), "manifest_sha256": sha(manifest),
            "threshold": config["threshold"],
            "boundary_tolerance_fraction": config["boundary_tolerance_fraction"],
            "split_lock_sha256": sha(ROOT / config["split_lock"]),
        }
        if any(protocol.get(key) != value for key, value in expected.items()):
            raise ValueError("Frozen protocol does not authorize this checkpoint/manifest/metric configuration")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Choose an empty output directory")
    loader = DataLoader(PolypDataset(manifest, config["image_size"]), batch_size=config["batch_size"],
                        num_workers=config["num_workers"], shuffle=False)
    metrics = evaluate(model, loader, device, config, lock, output=args.output, visual_count=6)
    write_json(args.output / "provenance.json", {
        "checkpoint": str(args.checkpoint), "checkpoint_sha256": sha(args.checkpoint),
        "manifest": str(manifest), "manifest_sha256": sha(manifest), "epoch": saved["epoch"],
        "purpose": config["purpose"], "fixed_visual_rule": "First 6 manifest IDs; no score-based selection",
        "frozen_protocol_sha256": sha(args.frozen_protocol) if args.frozen_protocol else None,
    })
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
