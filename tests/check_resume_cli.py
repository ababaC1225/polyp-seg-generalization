"""CPU CLI integration check using synthetic images; run from the project root."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image
import torch

root = Path.cwd()
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=Path, default=root / 'configs/unet_pilot.json')
parser.add_argument('--output', type=Path, default=root / 'results/recovery_checks')
args = parser.parse_args()
output = args.output
output.mkdir(parents=True, exist_ok=True)
(output / 'cli.log').write_text('')
(output / 'epoch_resume.json').unlink(missing_ok=True)
with tempfile.TemporaryDirectory(prefix='polyp_resume_check_') as directory:
    work = Path(directory)
    rng = np.random.default_rng(73)
    records = []
    for index in range(6):
        rgb = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[5:20, 7+index:22+index] = 255
        image, target = work / f'{index}.png', work / f'{index}_mask.png'
        Image.fromarray(rgb).save(image)
        Image.fromarray(mask).save(target)
        records.append(dict(id=str(index), image=str(image), mask=str(target)))
    manifests = {}
    for split, rows in [('train', records[:4]), ('val', records[4:])]:
        path = work / f'{split}.csv'
        with path.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=['id','image','mask'])
            writer.writeheader()
            writer.writerows(rows)
        manifests[split] = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), count=len(rows))
    lock = work / 'freeze.json'
    lock.write_text(json.dumps(dict(manifests=manifests, size_thresholds=[0.1,0.3])))
    config = json.loads(args.config.read_text())
    config.update(base_channels=2, image_size=[32,32], batch_size=2, epochs=2,
                  num_workers=2, cpu_threads=2, amp=False,
                  split_lock=str(lock), train_manifest=manifests['train']['path'],
                  val_manifest=manifests['val']['path'], purpose='synthetic_resume_check')
    config_path = work / 'config.json'
    config_path.write_text(json.dumps(config))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
    def run(name, *extra):
        command = [sys.executable, 'scripts/train.py', '--config', str(config_path),
                   '--output', str(work / name), '--device', 'cpu', *extra]
        result = subprocess.run(command, env=env, text=True, capture_output=True)
        with (output / 'cli.log').open('a') as handle:
            handle.write(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError(result.stderr)
    run('continuous')
    run('resumed', '--epochs', '1')
    run('resumed', '--resume', str(work / 'resumed/last.pt'))
    left = torch.load(work / 'continuous/last.pt', weights_only=True)
    right = torch.load(work / 'resumed/last.pt', weights_only=True)
    def equal(a,b):
        if isinstance(a, torch.Tensor): return torch.equal(a,b)
        if isinstance(a, dict): return a.keys() == b.keys() and all(equal(a[k],b[k]) for k in a)
        if isinstance(a, (list,tuple)): return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
        return a == b
    comparisons = {key: equal(left[key], right[key]) for key in ['model','optimizer','scaler','rng','best_val_dice']}
    metric_keys = ['train_loss','val_loss','val_dice','val_iou','val_boundary_f1']
    comparisons['epoch_metrics'] = (
        len(left['history']) == len(right['history']) == 2
        and all(a[k] == b[k] for a,b in zip(left['history'],right['history']) for k in metric_keys)
    )
    report = dict(passed=all(comparisons.values()), comparisons=comparisons,
                  device='cpu', num_workers=2, augmentation=True,
                  appearance_augmentation=config.get('appearance_augmentation'),
                  boundary_supervision=config.get('boundary_supervision'),
                  data='6 synthetic images only, no held-out real data',
                  procedure='2 uninterrupted epochs versus 1 epoch plus CLI --resume to epoch 2',
                  limitation='CPU epoch-boundary recovery; not GPU equivalence or mid-write crash recovery')
    (output / 'epoch_resume.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)
    assert report['passed']
