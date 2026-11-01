"""Frozen-split U-Net training and deterministic epoch-boundary resume."""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch.utils.data import DataLoader

from src.dataset import ROOT, PolypDataset
from src.experiment import (curve_svg, evaluate, restore_rng, save_checkpoint, seed_everything,
                            source_hashes, sync, verify_lock, write_json)
from src.metrics import training_objective
from src.model import build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/unet_pilot.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if args.epochs is not None:
        config["epochs"] = args.epochs
    lock = verify_lock(config)
    device = torch.device(args.device)
    seed_everything(config["seed"], config["cpu_threads"])
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    generator = torch.Generator().manual_seed(config["seed"])
    train = DataLoader(PolypDataset(ROOT / config["train_manifest"], config["image_size"], config["augment"],
                                   appearance=config.get("appearance_augmentation")),
                       batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"],
                       pin_memory=device.type == "cuda", generator=generator)
    validation = DataLoader(PolypDataset(ROOT / config["val_manifest"], config["image_size"]),
                            batch_size=config["batch_size"], shuffle=False,
                            num_workers=config["num_workers"], pin_memory=device.type == "cuda")
    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    amp = config["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    history, best, start = [], -1.0, 1
    if args.resume:
        if args.resume.resolve().parent != args.output.resolve():
            raise ValueError("Resume in the same output directory to preserve the best checkpoint/history")
        saved = torch.load(args.resume, map_location="cpu", weights_only=True)
        if {k: v for k, v in config.items() if k != "epochs"} != {
            k: v for k, v in saved["config"].items() if k != "epochs"
        } or lock != saved["split_lock"]:
            raise ValueError("Resume config or split lock differs; only epochs may change")
        if (args.output / "history.json").exists():
            existing = json.loads((args.output / "history.json").read_text())
            if existing != saved["history"]:
                raise ValueError("Resume requires checkpoint matching current history")
        for name, digest in saved["source_sha256"].items():
            if name.startswith("src/") or name == "scripts/train.py":
                if source_hashes().get(name) != digest:
                    raise ValueError(f"Resume source changed: {name}")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        restore_rng(saved["rng"], generator)
        history, best, start = saved["history"], saved["best_val_dice"], saved["epoch"] + 1
    elif args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Output is not empty; choose a new run directory or --resume")
    if start > config["epochs"]:
        raise ValueError("Requested epoch limit must exceed completed epochs")
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "config.json", config)
    write_json(args.output / "environment.json", {
        "torch": str(torch.__version__), "cuda_runtime": torch.version.cuda,
        "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "deterministic_algorithms": True, "source_sha256": source_hashes(),
        "resume_from_epoch": start - 1, "split_lock": lock,
    })
    for epoch in range(start, config["epochs"] + 1):
        model.train()
        sync(device)
        began = time.perf_counter()
        total, count, skipped = 0.0, 0, 0
        segmentation_total, boundary_total = 0.0, 0.0
        for step, batch in enumerate(train, 1):
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                outputs = model(images, return_boundary=config.get("boundary_supervision", {}).get("enabled", False))
            loss, components = training_objective(outputs, masks, config.get("boundary_supervision"))
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss")
            scaler.scale(loss).backward()
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            skipped += int(scaler.get_scale() < previous_scale)
            total += float(loss.detach()) * len(images)
            segmentation_total += float(components["segmentation"].detach()) * len(images)
            boundary_total += float(components["boundary"].detach()) * len(images)
            count += len(images)
            if step % 25 == 0:
                print(f"epoch={epoch} batch={step}/{len(train)} loss={total/count:.5f}", flush=True)
        sync(device)
        train_seconds = time.perf_counter() - began
        metrics = evaluate(model, validation, device, config, lock)
        row = {
            "epoch": epoch, "train_loss": total / count, "val_loss": metrics["loss_at_training_resolution"],
            "train_segmentation_loss": segmentation_total / count,
            "train_boundary_loss": boundary_total / count,
            "val_dice": metrics["dice"], "val_iou": metrics["iou"], "val_boundary_f1": metrics["boundary_f1"],
            "train_seconds": train_seconds, "val_seconds": metrics["evaluation_wall_seconds"],
            "epoch_seconds": time.perf_counter() - began, "amp_skipped_steps": skipped,
            "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2 ** 20 if device.type == "cuda" else 0,
            "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2 ** 20 if device.type == "cuda" else 0,
        }
        history.append(row)
        improved = metrics["dice"] > best
        best = max(best, metrics["dice"])
        save_checkpoint(args.output / "last.pt", model, optimizer, scaler, generator, config, lock,
                        epoch, best, history)
        if improved:
            save_checkpoint(args.output / "best.pt", model, optimizer, scaler, generator, config, lock,
                            epoch, best, history)
        write_json(args.output / "history.json", history)
        with (args.output / "history.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)
        curve_svg(history, args.output / "curves.svg")
        print(json.dumps(row), flush=True)
    print(f"Complete: best validation Dice={best:.6f}; artifacts={args.output}", flush=True)


if __name__ == "__main__":
    main()
