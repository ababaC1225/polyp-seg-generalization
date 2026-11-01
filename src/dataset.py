"""RGB [0,1], binary masks thresholded BEFORE nearest-neighbor resizing."""
import csv
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from src.augmentation import appearance_transform

ROOT = Path(__file__).resolve().parents[1]


class PolypDataset(Dataset):
    def __init__(self, manifest, size=(256, 256), augment=False, limit=None, appearance=None):
        with open(manifest) as file:
            self.rows = list(csv.DictReader(file))
        if not self.rows or len({r["id"] for r in self.rows}) != len(self.rows):
            raise ValueError("Manifest must be nonempty with unique IDs")
        if limit is not None:
            self.rows = self.rows[:limit]
        self.size = tuple(size)  # PIL (width, height).
        self.augment = augment
        self.appearance = appearance if augment else None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(ROOT / row["image"]) as image:
            original_size = image.size
            x = np.array(image.convert("RGB").resize(self.size, Image.Resampling.BILINEAR),
                         dtype=np.float32) / 255.0
        with Image.open(ROOT / row["mask"]) as image:
            if image.size != original_size:
                raise ValueError(f"Image/mask size mismatch: {row['id']}")
            mask = image.convert("L").point(lambda value: 255 if value >= 128 else 0)
            y = (np.array(mask.resize(self.size, Image.Resampling.NEAREST)) >= 128).astype(np.float32)
        x = torch.from_numpy(x.transpose(2, 0, 1).copy())
        y = torch.from_numpy(y[None].copy())
        if self.augment:
            for dimension in (1, 2):
                if torch.rand(()) < 0.5:
                    x, y = x.flip(dimension), y.flip(dimension)
        if self.appearance and self.appearance.get("enabled", False):
            x = appearance_transform(x, self.appearance)
        return {"image": x, "mask": y, "id": row["id"],
                "mask_path": str(ROOT / row["mask"]), "image_path": str(ROOT / row["image"])}
