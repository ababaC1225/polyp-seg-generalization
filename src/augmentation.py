"""Training-only photometric perturbations with an isolated reproducible RNG."""
import hashlib
from io import BytesIO

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import torch


def appearance_transform(image, config):
    """Keep geometry RNG unchanged, including the next sample's flip decisions.

Hash the current CPU RNG state into a local generator. Worker seeds change each
loader epoch, and epoch-boundary RNG restoration reproduces this for 0+ workers.
No external images or statistics are used to select these perturbations.
"""
    digest = hashlib.blake2b(torch.get_rng_state().numpy().tobytes(), digest_size=8,
                             person=b'polyp-A-v1').digest()
    generator = torch.Generator().manual_seed(int.from_bytes(digest, 'little') % (2**63))
    def uniform(bounds):
        return bounds[0] + (bounds[1] - bounds[0]) * torch.rand((), generator=generator).item()
    def choose(probability):
        return torch.rand((), generator=generator).item() < probability
    array = (image.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    pil = Image.fromarray(array)
    if choose(config['color_probability']):
        for enhancer, key in ((ImageEnhance.Brightness, 'brightness'),
                              (ImageEnhance.Contrast, 'contrast'),
                              (ImageEnhance.Color, 'saturation')):
            pil = enhancer(pil).enhance(uniform(config[key]))
    if choose(config['blur_probability']):
        pil = pil.filter(ImageFilter.GaussianBlur(uniform(config['blur_radius'])))
    if choose(config['jpeg_probability']):
        low, high = config['jpeg_quality']
        quality = int(torch.randint(low, high + 1, (), generator=generator))
        with BytesIO() as buffer:
            pil.save(buffer, format='JPEG', quality=quality, subsampling=2)
            buffer.seek(0)
            with Image.open(buffer) as compressed:
                pil = compressed.convert('RGB').copy()
    result = np.array(pil, dtype=np.float32) / 255.0
    return torch.from_numpy(result.transpose(2, 0, 1).copy())
