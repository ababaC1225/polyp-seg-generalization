"""FastAPI service that exposes the trained segmentation model to the Vue UI."""

import base64
import io
import os
import threading
import time
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from torch.nn import functional as F

from src.experiment import load_model


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / "results/unet_baseline_v1/seed20261020/best.pt"
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000


def image_data_url(image: Image.Image, image_format: str = "PNG", **save_options) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format=image_format, **save_options)
    mime = "image/jpeg" if image_format.upper() == "JPEG" else "image/png"
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:{mime};base64,{payload}"


class PredictionService:
    """Lazily load one checkpoint and serialize inference on the selected device."""

    def __init__(self):
        configured = os.environ.get("POLYP_CHECKPOINT")
        self.checkpoint = Path(configured).expanduser().resolve() if configured else DEFAULT_CHECKPOINT
        requested_device = os.environ.get("POLYP_DEVICE", "cuda:0")
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            requested_device = "cpu"
        self.device = torch.device(requested_device)
        self.model = None
        self.config = None
        self.load_lock = threading.Lock()
        self.inference_lock = threading.Lock()

    @property
    def ready(self) -> bool:
        return self.checkpoint.is_file()

    def load(self):
        if self.model is not None:
            return
        with self.load_lock:
            if self.model is not None:
                return
            if not self.ready:
                raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint}")
            model, saved = load_model(self.checkpoint, self.device)
            self.model = model
            self.config = saved["config"]
            width, height = self.config["image_size"]
            dummy = torch.zeros(1, 3, height, width, device=self.device)
            amp = self.config["amp"] and self.device.type == "cuda"
            with torch.inference_mode(), torch.autocast(device_type=self.device.type, enabled=amp):
                self.model(dummy)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)

    def predict(self, image: Image.Image, threshold: float) -> dict:
        self.load()
        original = image.convert("RGB")
        original_width, original_height = original.size
        resized = original.resize(tuple(self.config["image_size"]), Image.Resampling.BILINEAR)
        array = np.asarray(resized, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array.transpose(2, 0, 1).copy())[None].to(self.device)
        amp = self.config["amp"] and self.device.type == "cuda"

        with self.inference_lock:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            started = time.perf_counter()
            with torch.inference_mode(), torch.autocast(device_type=self.device.type, enabled=amp):
                logits = self.model(tensor)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            inference_ms = (time.perf_counter() - started) * 1000

        probability = F.interpolate(
            logits.float().sigmoid(),
            size=(original_height, original_width),
            mode="bilinear",
            align_corners=False,
        )[0, 0].cpu().numpy()
        prediction = probability >= threshold

        rgb = np.asarray(original).copy()
        overlay = rgb.copy()
        overlay[prediction] = (
            0.48 * overlay[prediction] + 0.52 * np.array([132, 88, 255])
        ).astype(np.uint8)
        mask_image = Image.fromarray(prediction.astype(np.uint8) * 255, mode="L")
        overlay_image = Image.fromarray(overlay, mode="RGB")
        probability_image = Image.fromarray((probability * 255).round().astype(np.uint8), mode="L")

        return {
            "model": "U-Net Baseline",
            "device": str(self.device),
            "width": original_width,
            "height": original_height,
            "threshold": threshold,
            "foreground_fraction": float(prediction.mean()),
            "inference_ms": round(inference_ms, 2),
            "mask_data_url": image_data_url(mask_image),
            "probability_data_url": image_data_url(probability_image),
            "overlay_data_url": image_data_url(overlay_image, "JPEG", quality=92),
        }


service = PredictionService()
app = FastAPI(title="PolypLab API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5274", "http://127.0.0.1:5274"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {
        "status": "ready" if service.ready else "missing_checkpoint",
        "model_loaded": service.model is not None,
        "device": str(service.device),
        "checkpoint_available": service.ready,
    }


@app.post("/api/predict")
async def predict(
    image: UploadFile = File(..., description="Colonoscopy image in JPEG or PNG format"),
    threshold: float = Form(0.5, ge=0.0, le=1.0),
):
    if image.content_type not in {"image/jpeg", "image/png"}:
        raise HTTPException(status_code=415, detail="仅支持 JPG、JPEG 和 PNG 图像")
    content = await image.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="图像文件不能超过 20 MB")
    try:
        decoded = Image.open(io.BytesIO(content))
        decoded.load()
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise HTTPException(status_code=400, detail="无法读取上传的图像") from error
    if decoded.width * decoded.height > MAX_IMAGE_PIXELS:
        raise HTTPException(status_code=413, detail="图像像素数量不能超过 2500 万")
    if not service.ready:
        raise HTTPException(status_code=503, detail="服务器未找到可用模型权重")
    try:
        return await run_in_threadpool(service.predict, decoded, threshold)
    except (RuntimeError, ValueError) as error:
        raise HTTPException(status_code=500, detail=f"模型推理失败：{error}") from error
