"""Single image prediction, native-resolution outputs and warmed inference timing."""
import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from src.experiment import load_model, panel, seed_everything, sha, sync, write_json
from src.metrics import boundary_f1, overlap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--mask", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Choose an empty output directory")
    device = torch.device(args.device)
    model, saved = load_model(args.checkpoint, device)
    config = saved["config"]
    seed_everything(config["seed"], config["cpu_threads"])
    amp = config["amp"] and device.type == "cuda"
    # Warmup at the configured input shape; excludes model loading and first CUDA initialization.
    dummy = torch.zeros(1, 3, config["image_size"][1], config["image_size"][0], device=device)
    with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=amp):
        for _ in range(5):
            model(dummy)
    sync(device)
    started = time.perf_counter()
    with Image.open(args.image) as image:
        size = (image.height, image.width)
        array = np.array(image.convert("RGB").resize(tuple(config["image_size"]),
                                                    Image.Resampling.BILINEAR), dtype=np.float32) / 255
    tensor = torch.from_numpy(array.transpose(2, 0, 1).copy())[None].to(device)
    sync(device)
    forward_start = time.perf_counter()
    with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=amp):
        logits = model(tensor)
    sync(device)
    first_ms = (time.perf_counter() - forward_start) * 1000
    probability = F.interpolate(logits.float().sigmoid(), size=size, mode="bilinear",
                                align_corners=False)[0, 0].cpu().numpy()
    prediction = probability >= config["threshold"]
    end_to_end_ms = (time.perf_counter() - started) * 1000
    times = []
    with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=amp):
        for _ in range(30):
            sync(device)
            began = time.perf_counter()
            model(tensor)
            sync(device)
            times.append((time.perf_counter() - began) * 1000)
    truth = None
    report = {
        "checkpoint_sha256": sha(args.checkpoint), "image": str(args.image),
        "threshold": config["threshold"], "prediction_foreground_fraction": float(prediction.mean()),
        "forward_first_warmed_ms": first_ms, "forward_median_ms": float(np.median(times)),
        "forward_p95_ms": float(np.quantile(times, 0.95)), "repeat_count": len(times), "warmup_count": 5,
        "end_to_end_ms": end_to_end_ms, "batch_size": 1, "device": str(device),
        "timing_scope": "End-to-end: image read/preprocess, transfer, forward, native resize, threshold; excludes model load, warmup, metrics and disk output.",
    }
    if args.mask:
        with Image.open(args.mask) as image:
            truth = np.asarray(image.convert("L")) >= 128
        report["metrics"] = {**overlap(prediction, truth),
                             "boundary_f1": boundary_f1(prediction, truth, config["boundary_tolerance_fraction"])}
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "probability.npy", probability)
    Image.fromarray((probability * 65535).round().astype(np.uint16)).save(args.output / "probability.png")
    Image.fromarray(prediction.astype(np.uint8) * 255).save(args.output / "mask.png")
    panel(args.image, truth, prediction, args.output / "panel.jpg")
    write_json(args.output / "prediction.json", report)
    print(report)


if __name__ == "__main__":
    main()
