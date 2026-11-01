"""Numerical edge cases, paired transforms and complete optimizer/RNG restoration."""
import csv
import json
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.dataset import PolypDataset
from src.experiment import load_model, restore_rng, save_checkpoint, seed_everything, verify_lock
from src.metrics import boundary, boundary_f1, overlap, segmentation_loss, size_group
from src.model import UNet


class PipelineTests(unittest.TestCase):
    def setUp(self):
        seed_everything(73, threads=2)

    def test_overlap_empty_perfect_disjoint_and_partial(self):
        empty = np.zeros((16, 16), dtype=bool)
        a, b = empty.copy(), empty.copy()
        a[2:6, 2:6], b[2:6, 4:8] = True, True
        self.assertEqual(overlap(empty, empty), {"dice": 1, "iou": 1})
        self.assertEqual(overlap(a, a), {"dice": 1, "iou": 1})
        self.assertEqual(overlap(a, empty), {"dice": 0, "iou": 0})
        self.assertEqual(overlap(empty, a), {"dice": 0, "iou": 0})
        self.assertAlmostEqual(overlap(a, b)["dice"], 0.5)
        self.assertAlmostEqual(overlap(a, b)["iou"], 1 / 3)
        self.assertEqual(overlap(a, np.roll(a, 8, axis=0))["dice"], 0)

    def test_boundary_and_border_convention(self):
        empty = np.zeros((32, 32), dtype=bool)
        mask = empty.copy()
        mask[10:20, 10:20] = True
        self.assertEqual(boundary_f1(empty, empty), 1)
        self.assertEqual(boundary_f1(mask, empty), 0)
        self.assertEqual(boundary_f1(empty, mask), 0)
        self.assertEqual(boundary_f1(mask, mask), 1)
        self.assertEqual(boundary_f1(mask, np.roll(mask, 1, axis=1)), 1)
        self.assertLess(boundary_f1(mask, np.roll(mask, 8, axis=1)), 0.5)
        self.assertEqual(boundary(np.ones((4, 4), dtype=bool)).sum(), 12)
        self.assertEqual(boundary_f1(np.ones_like(mask), np.ones_like(mask)), 1)

    def test_shape_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            overlap(np.zeros((2, 2)), np.zeros((3, 3)))

    def test_tertile_boundary_assignments(self):
        self.assertEqual(size_group(0.099, [0.1, 0.2]), "small")
        self.assertEqual(size_group(0.1, [0.1, 0.2]), "medium")
        self.assertEqual(size_group(0.2, [0.1, 0.2]), "large")

    def test_loss_extreme_logits_finite_gradients(self):
        logits = torch.tensor([[[[-100.0, 100.0], [0.0, 1.0]]]], requires_grad=True)
        loss = segmentation_loss(logits, torch.ones_like(logits))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(logits.grad.abs().sum()), 0)

    def test_unet_odd_non_square_output_and_backward(self):
        model = UNet(2)
        result = model(torch.randn(2, 3, 33, 49))
        self.assertEqual(result.shape, (2, 1, 33, 49))
        segmentation_loss(result, torch.zeros_like(result)).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in model.parameters()))

    def test_mask_threshold_and_synchronized_augmentation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mask = np.zeros((32, 48), dtype=np.uint8)
            mask[4:16, 20:30] = 255
            mask[0, 0], mask[0, 1] = 127, 128
            rgb = np.repeat((mask >= 128).astype(np.uint8)[..., None] * 255, 3, axis=2)
            Image.fromarray(rgb).save(root / "image.png")
            Image.fromarray(mask).save(root / "mask.png")
            with (root / "manifest.csv").open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=["id", "image", "mask"])
                writer.writeheader()
                writer.writerow({"id": "synthetic", "image": root / "image.png", "mask": root / "mask.png"})
            data = PolypDataset(root / "manifest.csv", size=(48, 32), augment=True)
            for _ in range(12):
                batch = data[0]
                self.assertTrue(torch.equal(batch["image"][:1], batch["mask"]))
                self.assertEqual(set(batch["mask"].unique().tolist()), {0, 1})
            resized = PolypDataset(root / "manifest.csv", size=(19, 17))[0]
            self.assertEqual(set(resized["mask"].unique().tolist()), {0, 1})

    def test_checkpoint_restores_optimizer_rng_and_next_step(self):
        config = {"base_channels": 2}
        model = UNet(2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        generator = torch.Generator().manual_seed(42)

        def update(network, optim):
            x = torch.randn(2, 3, 32, 32)
            target = (torch.rand(2, 1, 32, 32) > 0.5).float()
            optim.zero_grad()
            loss = segmentation_loss(network(x), target)
            loss.backward()
            optim.step()
            return loss.detach().clone()

        update(model, optimizer)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "last.pt"
            save_checkpoint(checkpoint, model, optimizer, scaler, generator, config, {}, 1, 0.2, [])
            expected_random = (random.random(), float(np.random.rand()), torch.randperm(8, generator=generator))
            expected_loss = update(model, optimizer)
            restored, saved = load_model(checkpoint, torch.device("cpu"))
            restored.train()
            restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.001)
            restored_optimizer.load_state_dict(saved["optimizer"])
            restore_rng(saved["rng"], generator)
            actual_random = (random.random(), float(np.random.rand()), torch.randperm(8, generator=generator))
            self.assertEqual(expected_random[:2], actual_random[:2])
            self.assertTrue(torch.equal(expected_random[2], actual_random[2]))
            self.assertTrue(torch.equal(expected_loss, update(restored, restored_optimizer)))
            for actual, expected in zip(restored.parameters(), model.parameters()):
                self.assertTrue(torch.equal(actual, expected))

    def test_split_lock_and_disjoint_groups(self):
        config = json.loads(Path("configs/unet_pilot.json").read_text())
        lock = verify_lock(config)
        seen, groups = set(), {}
        for split, record in lock["manifests"].items():
            with open(record["path"]) as file:
                rows = list(csv.DictReader(file))
            for row in rows:
                self.assertNotIn(row["id"], seen)
                seen.add(row["id"])
                self.assertEqual(groups.setdefault(row["group"], split), split)
        self.assertEqual(len(seen), 1000)


if __name__ == "__main__":
    unittest.main()
