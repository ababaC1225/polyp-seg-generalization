"""Memorize four TRAIN images; this is a pipeline check, never a test-set result."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.dataset import ROOT, PolypDataset
from src.experiment import save_checkpoint, seed_everything, sync, verify_lock, write_json
from src.metrics import overlap, training_objective
from src.model import build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/unet_pilot.json")
    parser.add_argument("--output", type=Path, default=Path("results/overfit_unet"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=400)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Choose an empty output directory")
    config = json.loads(Path(args.config).read_text())
    config.update(augment=False, learning_rate=0.001, weight_decay=0.0, purpose="four_training_images_overfit")
    lock = verify_lock(config)
    seed_everything(config["seed"], config["cpu_threads"])
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    batch = next(iter(DataLoader(PolypDataset(ROOT / config["train_manifest"],
                                            config["image_size"], limit=4), batch_size=4)))
    images, masks = batch["image"].to(device), batch["mask"].to(device)
    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=0)
    amp = config["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    history = []
    args.output.mkdir(parents=True, exist_ok=True)
    sync(device)
    began = time.perf_counter()
    for step in range(args.steps + 1):
        if step:
            model.train()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                outputs = model(images, return_boundary=config.get("boundary_supervision", {}).get("enabled", False))
            loss, _ = training_objective(outputs, masks, config.get("boundary_supervision"))
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        if step % 25 == 0 or step == args.steps:
            model.eval()
            with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=amp):
                outputs = model(images, return_boundary=config.get("boundary_supervision", {}).get("enabled", False))
            loss = float(training_objective(outputs, masks, config.get("boundary_supervision"))[0])
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            predictions = (logits.float().sigmoid() >= config["threshold"]).cpu().numpy()
            targets = masks.cpu().numpy()
            dice = [overlap(prediction, truth)["dice"] for prediction, truth in zip(predictions, targets)]
            row = {"step": step, "loss": loss, "mean_dice": float(np.mean(dice)), "min_dice": min(dice)}
            history.append(row)
            print(json.dumps(row), flush=True)
            if row["mean_dice"] >= 0.95 and row["min_dice"] >= 0.90:
                break
    sync(device)
    passed = history[-1]["mean_dice"] >= 0.95 and history[-1]["min_dice"] >= 0.90
    report = {
        "passed": passed, "criterion": "mean Dice >=0.95 AND minimum Dice >=0.90 on 4 training images",
        "metric_resolution": config["image_size"], "ids": batch["id"], "steps": step,
        "history": history, "wall_seconds": time.perf_counter() - began,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2 ** 20 if device.type == "cuda" else 0,
        "purpose": "Pipeline sanity check, not generalization evidence",
    }
    generator = torch.Generator().manual_seed(config["seed"])
    save_checkpoint(args.output / "last.pt", model, optimizer, scaler, generator, config, lock,
                    0, history[-1]["mean_dice"], [],)
    write_json(args.output / "report.json", report)
    write_json(args.output / "config.json", config)
    if not passed:
        raise SystemExit("Overfit check failed; inspect report before pilot training")


if __name__ == "__main__":
    main()
