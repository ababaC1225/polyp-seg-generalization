"""Experiment bookkeeping, native-resolution validation and visual outputs."""
import csv
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.nn import functional as F

from src.dataset import ROOT
from src.metrics import boundary_f1, overlap, segmentation_loss, size_group
from src.model import build_model


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def seed_everything(seed, threads=4):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def source_hashes():
    paths = sorted((ROOT / "src").glob("*.py")) + sorted((ROOT / "scripts").glob("*.py"))
    return {str(path.relative_to(ROOT)): sha(path) for path in paths}


def verify_lock(config):
    lock = json.loads((ROOT / config["split_lock"]).read_text())
    for name, record in lock["manifests"].items():
        if sha(ROOT / record["path"]) != record["sha256"]:
            raise ValueError(f"Frozen {name} manifest has changed")
    for name in ("train", "val"):
        if sha(ROOT / config[f"{name}_manifest"]) != lock["manifests"][name]["sha256"]:
            raise ValueError(f"Config does not use frozen {name} manifest")
    return lock


def rng_state(generator):
    np_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [np_state[0], np_state[1].tolist(), np_state[2], np_state[3], np_state[4]],
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "loader": generator.get_state(),
    }


def restore_rng(state, generator):
    random.setstate(state["python"])
    ns = state["numpy"]
    np.random.set_state((ns[0], np.array(ns[1], dtype=np.uint32), ns[2], ns[3], ns[4]))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])
    generator.set_state(state["loader"].cpu())


def save_checkpoint(path, model, optimizer, scaler, generator, config, lock, epoch, best, history):
    value = {
        "format_version": 1, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(), "rng": rng_state(generator), "config": config,
        "split_lock": lock, "epoch": epoch, "best_val_dice": best, "history": history,
        "source_sha256": source_hashes(),
    }
    temporary = Path(path).with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def load_model(checkpoint, device):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = build_model(saved["config"]).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    return model, saved


def panel(image_path, truth, prediction, destination, title=""):
    with Image.open(image_path) as image:
        rgb = np.array(image.convert("RGB"))
    overlay = rgb.copy()
    overlay[prediction] = (0.55 * overlay[prediction] + 0.45 * np.array([255, 60, 40])).astype(np.uint8)
    arrays = [rgb]
    labels = ["Image"]
    if truth is not None:
        arrays.append(truth.astype(np.uint8) * 255)
        labels.append("Ground truth")
    arrays.extend([prediction.astype(np.uint8) * 255, overlay])
    labels.extend(["Prediction", "Overlay"])
    canvas = Image.new("RGB", (256 * len(arrays), 290), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (array, label) in enumerate(zip(arrays, labels)):
        tile = Image.fromarray(array).convert("RGB")
        tile.thumbnail((256, 250))
        canvas.paste(tile, (index * 256, 38))
        draw.text((index * 256 + 4, 20), label, fill="black")
    draw.text((4, 3), title, fill="black")
    canvas.save(destination)


@torch.inference_mode()
def evaluate(model, loader, device, config, lock, output=None, visual_count=0):
    model.eval()
    rows, loss_total, forward_seconds, count = [], 0.0, 0.0, 0
    amp = config["amp"] and device.type == "cuda"
    if output:
        Path(output).mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    for batch in loader:
        images, masks = batch["image"].to(device), batch["mask"].to(device)
        sync(device)
        begin = time.perf_counter()
        with torch.autocast(device_type=device.type, enabled=amp):
            logits = model(images)
        sync(device)
        forward_seconds += time.perf_counter() - begin
        loss_total += float(segmentation_loss(logits, masks)) * len(images)
        count += len(images)
        probabilities = logits.float().sigmoid()
        for index, identifier in enumerate(batch["id"]):
            with Image.open(batch["mask_path"][index]) as image:
                truth = np.asarray(image.convert("L")) >= 128
            probability = F.interpolate(probabilities[index:index + 1], size=truth.shape,
                                        mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
            prediction = probability >= config["threshold"]
            metrics = overlap(prediction, truth)
            fraction = float(truth.mean())
            row = {
                "id": identifier, **metrics,
                "boundary_f1": boundary_f1(prediction, truth, config["boundary_tolerance_fraction"]),
                "foreground_fraction": fraction,
                "size_group": size_group(fraction, lock["size_thresholds"]),
            }
            rows.append(row)
            if output and len(rows) <= visual_count:
                Image.fromarray(prediction.astype(np.uint8) * 255).save(Path(output) / f"{identifier}_mask.png")
                panel(batch["image_path"][index], truth, prediction, Path(output) / f"{identifier}_panel.jpg",
                      f"Dice={metrics['dice']:.4f} IoU={metrics['iou']:.4f}")
    summary = {
        "count": count, "loss_at_training_resolution": loss_total / count,
        **{key: float(np.mean([row[key] for row in rows])) for key in ("dice", "iou", "boundary_f1")},
        "forward_ms_per_image_amortized": 1000 * forward_seconds / count,
        "evaluation_wall_seconds": time.perf_counter() - started,
        "metric_resolution": "original", "threshold": config["threshold"],
        "groups": {
            name: {
                "count": sum(row["size_group"] == name for row in rows),
                **{key: float(np.mean([row[key] for row in rows if row["size_group"] == name]))
                   if any(row["size_group"] == name for row in rows) else None
                   for key in ("dice", "iou", "boundary_f1")},
            } for name in ("small", "medium", "large")
        },
        "timing_note": "Synchronized forward time divided by images, includes first batch; not single-image latency.",
    }
    if output:
        with (Path(output) / "per_image.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        write_json(Path(output) / "metrics.json", summary)
    return summary


def curve_svg(history, destination):
    """Export standard Matplotlib figures plus exact history.csv values."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    for axis, keys, title in zip(axes, (("train_loss", "val_loss"), ("val_dice", "val_iou")),
                                 ("BCE + Dice loss", "Native-resolution validation")):
        for key in keys:
            axis.plot([row["epoch"] for row in history], [row[key] for row in history],
                      marker="o", markersize=3, label=key)
        axis.set(xlabel="Epoch", title=title)
        axis.grid(alpha=0.2)
        axis.legend()
    axes[1].set_ylim(0, 1)
    figure.savefig(destination)
    figure.savefig(Path(destination).with_suffix(".png"), dpi=160)
    plt.close(figure)
