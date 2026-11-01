"""Per-image metrics and explicitly defined pixel boundary tolerance."""
import math

import numpy as np
import torch
from PIL import Image, ImageFilter
from torch.nn import functional as F


def segmentation_loss(logits, target):
    logits = logits.float()
    probabilities = logits.sigmoid()
    dimensions = (1, 2, 3)
    dice = (2 * (probabilities * target).sum(dimensions) + 1) / (
        probabilities.sum(dimensions) + target.sum(dimensions) + 1
    )
    return F.binary_cross_entropy_with_logits(logits, target) + 1 - dice.mean()


def overlap(prediction, target):
    prediction, target = np.asarray(prediction, dtype=bool), np.asarray(target, dtype=bool)
    if prediction.shape != target.shape:
        raise ValueError("Metric masks must have the same shape")
    intersection = int(np.logical_and(prediction, target).sum())
    total = int(prediction.sum()) + int(target.sum())
    union = int(np.logical_or(prediction, target).sum())
    return {"dice": 2 * intersection / total if total else 1.0,
            "iou": intersection / union if union else 1.0}


def boundary(mask):
    mask = np.asarray(mask, dtype=bool)
    height, width = mask.shape
    padded = np.pad(mask, 1, constant_values=False)
    eroded = np.ones_like(mask)
    for dy in range(3):
        for dx in range(3):
            eroded &= padded[dy:dy + height, dx:dx + width]
    return mask & ~eroded


def boundary_f1(prediction, target, tolerance_fraction=0.005):
    """Inner 8-connected contours; square/Chebyshev tolerance at native resolution."""
    pb, tb = boundary(prediction), boundary(target)
    if not pb.any() or not tb.any():
        return float(not pb.any() and not tb.any())
    radius = max(1, int(math.floor(tolerance_fraction * math.hypot(*tb.shape) + 0.5)))

    def dilate(array):
        return np.asarray(Image.fromarray(array.astype(np.uint8) * 255).filter(
            ImageFilter.MaxFilter(2 * radius + 1))) > 0

    precision = (pb & dilate(tb)).sum() / pb.sum()
    recall = (tb & dilate(pb)).sum() / tb.sum()
    return float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0


def size_group(fraction, thresholds):
    return "small" if fraction < thresholds[0] else "medium" if fraction < thresholds[1] else "large"


def boundary_target(mask, radius=2):
    """Inner contour of a binary BCHW mask; image exterior is background."""
    if not isinstance(radius, int) or radius < 1:
        raise ValueError('Boundary radius must be a positive integer')
    inverse = F.pad(1 - mask.float(), (radius,) * 4, value=1)
    erosion = 1 - F.max_pool2d(inverse, 2 * radius + 1, stride=1)
    return (mask.float() - erosion).clamp(0, 1)


def training_objective(outputs, masks, config=None):
    """Segmentation BCE+Dice, optionally plus weighted boundary BCE+Dice."""
    config = config or {}
    if not config.get('enabled', False):
        segmentation = segmentation_loss(outputs, masks)
        return segmentation, {'segmentation': segmentation, 'boundary': segmentation.new_zeros(())}
    if config['weight'] <= 0 or config['positive_weight'] <= 0:
        raise ValueError('Boundary loss weights must be positive')
    logits, edges = outputs
    segmentation = segmentation_loss(logits, masks)
    target = boundary_target(masks, config['radius'])
    edges = edges.float()
    probabilities = edges.sigmoid()
    dims = (1, 2, 3)
    dice = (2 * (probabilities * target).sum(dims) + 1) / (
        probabilities.sum(dims) + target.sum(dims) + 1)
    boundary_loss = F.binary_cross_entropy_with_logits(
        edges, target, pos_weight=edges.new_tensor(config['positive_weight'])) + 1 - dice.mean()
    return segmentation + config['weight'] * boundary_loss, {
        'segmentation': segmentation, 'boundary': boundary_loss}
