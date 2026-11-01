"""Audit frozen baseline artifacts and export validation-only tables and figures."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_baselines import read, verify_files, write


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def csv_rows(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def csv_write(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, default=ROOT / 'results/unet_baseline_v1')
    args = parser.parse_args()
    run = args.run.resolve()
    protocol, summary = read(run / 'protocol.json'), read(run / 'summary.json')
    snapshot = run / 'source'
    verify_files(snapshot, protocol['files_sha256'])
    assert summary['status'] == 'complete' and not summary['held_out_evaluated']
    output = run / 'analysis'; output.mkdir(exist_ok=True)
    torch.set_num_threads(2)
    records, histories, tables, groups, ledger = [], {}, {}, {}, []
    for entry, summary_row in zip(protocol['runs'], summary['runs']):
        directory = run / entry['name']
        config = read(snapshot / entry['config'])
        lock = read(snapshot / config['split_lock'])
        expected_ids = {row['id'] for row in csv_rows(snapshot / config['val_manifest'])}
        history = read(directory / 'history.json')
        assert [row['epoch'] for row in history] == list(range(1, entry['epochs'] + 1))
        best_row = max(history, key=lambda row: row['val_dice'])
        metrics = read(directory / 'validation/metrics.json')
        provenance = read(directory / 'validation/provenance.json')
        rows = csv_rows(directory / 'validation/per_image.csv')
        assert len(rows) == len(expected_ids) == len({row['id'] for row in rows}) == metrics['count']
        assert {row['id'] for row in rows} == expected_ids
        assert provenance['epoch'] == best_row['epoch'] == summary_row['best_epoch']
        assert entry['seed'] == summary_row['seed']
        assert provenance['manifest_sha256'] == protocol['validation_manifest_sha256']
        assert provenance['checkpoint_sha256'] == sha(directory / 'best.pt') == summary_row['checkpoint_sha256']
        for key in ('dice', 'iou', 'boundary_f1'):
            values = np.array([float(row[key]) for row in rows])
            assert np.isfinite(values).all() and np.all((values >= 0) & (values <= 1))
            assert np.isclose(values.mean(), metrics[key], rtol=0, atol=1e-12)
            assert metrics[key] == best_row['val_' + key] == summary_row[key]
        for name, group in metrics['groups'].items():
            selected = [row for row in rows if row['size_group'] == name]
            assert group['count'] == len(selected)
            for key in ('dice', 'iou', 'boundary_f1'):
                assert np.isclose(np.mean([float(row[key]) for row in selected]), group[key])
        for filename, expected_epoch in [('best.pt', best_row['epoch']), ('last.pt', entry['epochs'])]:
            saved = torch.load(directory / filename, map_location='cpu', weights_only=True)
            assert saved['epoch'] == expected_epoch and saved['config'] == config and saved['split_lock'] == lock
            assert saved['history'] == history[:expected_epoch]
            verify_files(snapshot, saved['source_sha256'])
            assert all(torch.isfinite(value).all() for value in saved['model'].values())
            del saved
        record = {
            'seed': entry['seed'], 'epochs': len(history), 'best_epoch': best_row['epoch'],
            **{key: metrics[key] for key in ('dice', 'iou', 'boundary_f1')},
            'train_hours': sum(row['train_seconds'] for row in history) / 3600,
            'train_val_hours': sum(row['epoch_seconds'] for row in history) / 3600,
            'peak_allocated_mib': max(row['peak_allocated_mib'] for row in history),
            'peak_reserved_mib': max(row['peak_reserved_mib'] for row in history),
            'amp_skipped_steps': sum(row['amp_skipped_steps'] for row in history),
        }
        records.append(record)
        histories[entry['seed']] = history
        tables[entry['seed']] = {row['id']: row for row in rows}
        groups[entry['seed']] = metrics['groups']
        ledger.append({'seed': entry['seed'], 'best_epoch': best_row['epoch'],
                       'checkpoint': str((directory / 'best.pt').relative_to(ROOT)),
                       'checkpoint_sha256': sha(directory / 'best.pt'),
                       'last_checkpoint_sha256': sha(directory / 'last.pt'),
                       'history_sha256': sha(directory / 'history.json'),
                       'validation_metrics_sha256': sha(directory / 'validation/metrics.json'),
                       'validation_per_image_sha256': sha(directory / 'validation/per_image.csv')})
    assert len(records) == len(summary['runs']) == len(protocol['runs'])
    for key in ('dice', 'iou', 'boundary_f1'):
        assert np.isclose(statistics.mean(row[key] for row in records), summary['metrics'][key]['mean'])
        assert np.isclose(statistics.stdev(row[key] for row in records), summary['metrics'][key]['sample_std'])
    grouped = []
    for group_name in ('small', 'medium', 'large'):
        counts = {groups[seed][group_name]['count'] for seed in groups}
        assert len(counts) == 1
        row = {'group': group_name, 'count': counts.pop()}
        for key in ('dice', 'iou', 'boundary_f1'):
            values = [groups[seed][group_name][key] for seed in groups]
            row[key + '_mean'] = statistics.mean(values)
            row[key + '_seed_std'] = statistics.stdev(values)
        grouped.append(row)
    per_image = []
    for identifier in sorted(next(iter(tables.values()))):
        items = [table[identifier] for table in tables.values()]
        assert len({item['size_group'] for item in items}) == 1
        per_image.append({'id': identifier, 'size_group': items[0]['size_group'],
                          'mean_seed_dice': statistics.mean(float(item['dice']) for item in items),
                          'seed_dice_std': statistics.stdev(float(item['dice']) for item in items),
                          'mean_seed_iou': statistics.mean(float(item['iou']) for item in items)})
    ranked = sorted(per_image, key=lambda row: (row['mean_seed_dice'], row['id']))
    csv_write(output / 'seed_results.csv', records)
    csv_write(output / 'size_groups.csv', grouped)
    csv_write(output / 'per_image_seed_summary.csv', per_image)
    csv_write(output / 'lowest10_validation.csv', ranked[:10])
    csv_write(output / 'highest10_validation.csv', ranked[-10:])
    write(output / 'checkpoint_ledger.json', {'protocol_id': protocol['protocol_id'],
          'protocol_sha256': sha(run / 'protocol.json'), 'held_out_evaluated': False, 'runs': ledger})

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for seed, history in histories.items():
        for axis, key, title in zip(axes, ('train_loss', 'val_dice', 'val_iou'),
                                   ('Training BCE + Dice loss', 'Validation Dice', 'Validation IoU')):
            axis.plot([row['epoch'] for row in history], [row[key] for row in history], label=str(seed))
            axis.set(xlabel='Epoch', title=title); axis.grid(alpha=0.2)
        best = max(history, key=lambda row: row['val_dice'])
        axes[1].scatter([best['epoch']], [best['val_dice']], s=24)
    for axis in axes[1:]: axis.set_ylim(0,1)
    axes[0].legend(title='Seed', fontsize=8)
    figure.suptitle('U-Net baseline | Kvasir validation only | 3 seeds x 100 epochs')
    figure.savefig(output / 'learning_curves.png', dpi=180)
    figure.savefig(output / 'learning_curves.svg'); plt.close(figure)
    figure, axis = plt.subplots(figsize=(7,4), constrained_layout=True)
    for index, key in enumerate(('dice','iou','boundary_f1')):
        positions = np.arange(3) + (index-1)*0.23
        axis.bar(positions, [row[key+'_mean'] for row in grouped], 0.23,
                 yerr=[row[key+'_seed_std'] for row in grouped], capsize=3, label=key)
    axis.set(xticks=np.arange(3), xticklabels=[f"{r['group']} (n={r['count']})" for r in grouped],
             ylim=(0,1), ylabel='Metric', title='Validation area groups | mean +/- seed SD')
    axis.legend(); axis.grid(axis='y', alpha=0.2)
    figure.savefig(output / 'size_groups.png', dpi=180)
    figure.savefig(output / 'size_groups.svg'); plt.close(figure)
    report = {'passed': True, 'audited_at_utc': datetime.now(timezone.utc).isoformat(),
              'checkpoint_count': len(records)*2, 'epochs_per_seed': [r['epochs'] for r in records],
              'validation_images_per_seed': len(per_image), 'source_snapshot_verified': True,
              'best_selection_and_checkpoint_history_verified': True,
              'per_image_group_and_seed_aggregates_verified': True,
              'held_out_evaluated': False, 'runs': records, 'size_groups': grouped,
              'notes': ['Seed SD is not a sample confidence interval.',
                        'Averaging per-seed image scores is not evaluating a prediction ensemble.',
                        'Validation is used for checkpoint selection; this is not held-out test performance.']}
    write(output / 'audit.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
