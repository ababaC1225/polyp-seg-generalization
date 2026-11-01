"""Audit completed A/B pilots and the real CLI prediction before formal freeze."""
import csv
from datetime import datetime, timezone
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_baselines import read, sha, verify_files, write
from src.metrics import boundary_f1, overlap
from src.model import build_model


def rows(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    torch.set_num_threads(2)
    run = ROOT / 'results/unet_ablation_pilot_v1'
    output = ROOT / 'results/ablation_preflight'
    protocol, summary = read(run / 'protocol.json'), read(run / 'summary.json')
    snapshot = run / 'source'
    verify_files(snapshot, protocol['files_sha256'])
    verify_files(ROOT, protocol['files_sha256'])
    require(summary['status'] == 'complete' and not summary['held_out_evaluated'], 'Incomplete pilots')
    require(len(summary['runs']) == len(protocol['runs']) == 3, 'Expected three pilots')
    require({r['condition'] for r in protocol['runs']} == {'A', 'B', 'AB'}, 'Missing condition')
    lock = read(ROOT / 'data/splits/freeze.json')
    inventory_path = ROOT / 'results/kvasir_seg/inventory.csv'
    require(sha(inventory_path) == lock['inventory_sha256'], 'Inventory differs')
    inventory = {r['id']: r for r in rows(inventory_path)}
    raw_files_verified = 0
    for split in ('train', 'val'):
        for row in rows(ROOT / lock['manifests'][split]['path']):
            for kind in ('image', 'mask'):
                require(sha(ROOT / row[kind]) == inventory[row['id']][kind + '_sha256'],
                        f'Raw file changed: {row[kind]}')
                raw_files_verified += 1
    records = []
    for entry in protocol['runs']:
        directory = run / entry['name']
        config = read(snapshot / entry['config'])
        lock = read(snapshot / config['split_lock'])
        history = read(directory / 'history.json')
        require([r['epoch'] for r in history] == list(range(1, 6)), 'Incomplete history')
        boundary = config['boundary_supervision']['enabled']
        for row in history:
            require(all(math.isfinite(v) for v in row.values()), 'Nonfinite history')
            expected = row['train_segmentation_loss'] + (0.2 * row['train_boundary_loss'] if boundary else 0)
            require(math.isclose(row['train_loss'], expected, abs_tol=1e-6), 'Loss decomposition differs')
            require(row['train_boundary_loss'] > 0 if boundary else row['train_boundary_loss'] == 0,
                    'Auxiliary loss enabled incorrectly')
        best = max(history, key=lambda row: row['val_dice'])
        metrics = read(directory / 'validation/metrics.json')
        provenance = read(directory / 'validation/provenance.json')
        table = rows(directory / 'validation/per_image.csv')
        ids = {r['id'] for r in rows(snapshot / config['val_manifest'])}
        require(len(table) == len(ids) == metrics['count'] == 100, 'Validation count differs')
        require({r['id'] for r in table} == ids, 'Validation IDs differ')
        record = next(r for r in summary['runs'] if (r['condition'], r['seed']) ==
                      (entry['condition'], entry['seed']))
        require(provenance['epoch'] == best['epoch'] == record['best_epoch'], 'Best epoch differs')
        require(provenance['manifest_sha256'] == protocol['validation_manifest_sha256'], 'Manifest differs')
        require(provenance['checkpoint_sha256'] == sha(directory / 'best.pt') == record['checkpoint_sha256'],
                'Checkpoint hash differs')
        require(metrics['threshold'] == config['threshold'] == 0.5, 'Threshold differs')
        for key in ('dice', 'iou', 'boundary_f1'):
            values = np.array([float(r[key]) for r in table])
            require(np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all(), 'Invalid metric')
            require(math.isclose(values.mean(), metrics[key], abs_tol=1e-12), 'Macro average differs')
            require(metrics[key] == best['val_' + key] == record[key], 'Metric provenance differs')
            for name, group in metrics['groups'].items():
                selected = [float(r[key]) for r in table if r['size_group'] == name]
                require(len(selected) == group['count'] and
                        math.isclose(np.mean(selected), group[key], abs_tol=1e-12), 'Group average differs')
        for filename, epoch in [('best.pt', best['epoch']), ('last.pt', 5)]:
            saved = torch.load(directory / filename, map_location='cpu', weights_only=True)
            require(saved['epoch'] == epoch and saved['config'] == config and saved['split_lock'] == lock,
                    'Checkpoint metadata differs')
            require(saved['history'] == history[:epoch], 'Checkpoint history differs')
            verify_files(snapshot, saved['source_sha256'])
            require(all(torch.isfinite(v).all() for v in saved['model'].values()), 'Nonfinite weights')
            require(any(k.startswith('boundary_head.') for k in saved['model']) == boundary, 'Head differs')
            model = build_model(config)
            model.load_state_dict(saved['model'], strict=True)
            del model, saved
        records.append({**record, 'epochs': 5,
                        'last_checkpoint_sha256': sha(directory / 'last.pt'),
                        'history_sha256': sha(directory / 'history.json'),
                        'mean_last3_epoch_seconds': np.mean([r['epoch_seconds'] for r in history[-3:]]),
                        'peak_reserved_mib': max(r['peak_reserved_mib'] for r in history),
                        'amp_skipped_steps': sum(r['amp_skipped_steps'] for r in history)})

    prediction = output / 'prediction_AB'
    report = read(prediction / 'prediction.json')
    config = read(snapshot / 'configs/ablation_pilot/AB_seed20261020.json')
    example = rows(snapshot / config['val_manifest'])[0]
    require(Path(report['image']).resolve() == (ROOT / example['image']).resolve(), 'Prediction image differs')
    require(report['checkpoint_sha256'] == sha(run / 'AB_seed20261020/best.pt'), 'Prediction weights differ')
    probability = np.load(prediction / 'probability.npy')
    with Image.open(ROOT / example['image']) as image:
        shape = (image.height, image.width)
    with Image.open(ROOT / example['mask']) as image:
        truth = np.asarray(image.convert('L')) >= 128
    with Image.open(prediction / 'mask.png') as image:
        binary = np.asarray(image)
    require(probability.shape == binary.shape == truth.shape == shape, 'Native shape differs')
    require(np.isfinite(probability).all() and ((probability >= 0) & (probability <= 1)).all(), 'Invalid probabilities')
    require(report['threshold'] == config['threshold'], 'Prediction threshold differs')
    require(np.array_equal(binary, (probability >= config['threshold']).astype(np.uint8) * 255), 'Mask differs')
    expected = {**overlap(binary > 0, truth),
                'boundary_f1': boundary_f1(binary > 0, truth, config['boundary_tolerance_fraction'])}
    require(expected == report['metrics'], 'Prediction metrics differ')
    with Image.open(prediction / 'probability.png') as image:
        require(np.array_equal(np.asarray(image), (probability * 65535).round().astype(np.uint16)),
                'Probability PNG differs')

    evidence = ['overfit_B/report.json', 'resume_AB/epoch_resume.json']
    for name in evidence:
        require(read(output / name)['passed'] is True, f'Failed preflight: {name}')
    unit_log = (output / 'unit_tests.log').read_text()
    require('Ran 15 tests' in unit_log and unit_log.rstrip().endswith('OK'), 'Unit checks not passed')
    evidence += ['unit_tests.log', 'prediction_AB/prediction.json', 'prediction_AB/probability.npy',
                 'prediction_AB/probability.png', 'prediction_AB/mask.png']
    acceptance = {'passed': True, 'audited_at_utc': datetime.now(timezone.utc).isoformat(),
                  'pilot_protocol_sha256': sha(run / 'protocol.json'),
                  'pilot_summary_sha256': sha(run / 'summary.json'),
                  'snapshot_and_live_frozen_files_verified': True, 'checkpoint_count': 6,
                  'train_validation_raw_files_verified': raw_files_verified,
                  'validation_images_per_condition': 100, 'native_cli_prediction_verified': True,
                  'evidence_sha256': {name: sha(output / name) for name in evidence},
                  'held_out_evaluated': False, 'runs': records,
                  'note': 'Five-epoch scores are pipeline checks, not method selection or generalization evidence.'}
    write(output / 'acceptance.json', acceptance)
    print(f'PASS: 3 pilots, 6 checkpoints, validation exports and native CLI prediction; {output / "acceptance.json"}')


if __name__ == '__main__':
    main()
