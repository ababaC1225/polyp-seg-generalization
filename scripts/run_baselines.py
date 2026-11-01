"""Run a frozen multi-seed training queue from an isolated source snapshot.

Training and validation only. Never evaluates the internal or external test sets.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def safe_relative(value):
    path = Path(value)
    if path.is_absolute() or '..' in path.parts or not path.parts:
        raise ValueError(f'Expected a project-relative path: {value}')
    return path


def verify_files(root, files):
    for name, digest in files.items():
        path = root / safe_relative(name)
        if not path.is_file() or sha(path) != digest:
            raise ValueError(f'Frozen file changed or missing: {path}')


def prepare_snapshot(protocol, output):
    snapshot = output / 'source'
    marker = output / 'protocol.json'
    if marker.exists():
        if read(marker) != protocol:
            raise ValueError('Output belongs to another frozen protocol')
        verify_files(snapshot, protocol['files_sha256'])
        if read(snapshot / 'frozen_protocol.json') != protocol:
            raise ValueError('Snapshot protocol changed')
        if not (snapshot / 'data/raw').is_symlink():
            raise ValueError('Snapshot raw-data link missing')
        return snapshot
    if snapshot.exists():
        raise ValueError('Incomplete snapshot; inspect it before using a new output directory')
    verify_files(ROOT, protocol['files_sha256'])
    snapshot.mkdir(parents=True)
    for name in protocol['files_sha256']:
        relative = safe_relative(name)
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    (snapshot / 'data').mkdir(exist_ok=True)
    (snapshot / 'data/raw').symlink_to((ROOT / 'data/raw').resolve(), target_is_directory=True)
    verify_files(snapshot, protocol['files_sha256'])
    write(snapshot / 'frozen_protocol.json', protocol)
    write(marker, protocol)
    return snapshot


def progress(protocol, output):
    runs = []
    for entry in protocol['runs']:
        directory = output / entry['name']
        history = []
        if (directory / 'history.json').exists():
            try:
                history = read(directory / 'history.json')
            except json.JSONDecodeError:
                pass  # Training may be replacing a history file; retry next heartbeat.
        runs.append({
            'name': entry['name'], 'seed': entry['seed'],
            'condition': entry.get('condition', 'baseline'),
            'completed_epochs': history[-1]['epoch'] if history else 0,
            'epoch_limit': entry['epochs'],
            'best_val_dice': max(row['val_dice'] for row in history) if history else None,
            'validation_exported': (directory / 'validation/provenance.json').exists(),
        })
    return runs


def validate_protocol(protocol):
    if protocol.get('status') != 'training_frozen' or protocol.get('evaluate_test') is not False:
        raise ValueError('Expected a frozen training-only protocol')
    runs = protocol['runs']
    if protocol.get('baseline_reference'):
        safe_relative(protocol['baseline_reference'])
        if protocol['baseline_reference'] not in protocol['files_sha256']:
            raise ValueError('Baseline reference must be frozen in the snapshot')
    if not runs or len({(entry.get('condition', 'baseline'), entry['seed']) for entry in runs}) != len(runs):
        raise ValueError('Condition/seed pairs must be unique and nonempty')
    if len({entry['name'] for entry in runs}) != len(runs):
        raise ValueError('Run names must be unique')
    for entry in runs:
        if len(safe_relative(entry['name']).parts) != 1 or entry['epochs'] < 1:
            raise ValueError('Invalid run name or epoch count')
        safe_relative(entry['config'])
        if entry['config'] not in protocol['files_sha256']:
            raise ValueError('Every run config must be hashed')


def aggregate(protocol, output):
    records = []
    for entry in protocol['runs']:
        directory = output / entry['name']
        metrics = read(directory / 'validation/metrics.json')
        provenance = read(directory / 'validation/provenance.json')
        if provenance['checkpoint_sha256'] != sha(directory / 'best.pt'):
            raise ValueError('Validation output does not match the current best checkpoint')
        if provenance['manifest_sha256'] != protocol['validation_manifest_sha256']:
            raise ValueError('Validation manifest does not match the frozen protocol')
        records.append({
            'seed': entry['seed'], 'best_epoch': provenance['epoch'],
            'condition': entry.get('condition', 'baseline'),
            'checkpoint_sha256': provenance['checkpoint_sha256'],
            **{key: metrics[key] for key in ('count', 'dice', 'iou', 'boundary_f1')},
        })
    by_condition = {}
    for condition in sorted({row['condition'] for row in records}):
        selected = [row for row in records if row['condition'] == condition]
        by_condition[condition] = {
            'seed_count': len(selected),
            'metrics': {key: {
                'mean': statistics.mean(row[key] for row in selected),
                'sample_std': statistics.stdev(row[key] for row in selected) if len(selected) > 1 else None,
            } for key in ('dice', 'iou', 'boundary_f1')},
        }
    summary = {
        'status': 'complete', 'purpose': 'validation_only_not_cross_dataset_results',
        'protocol_id': protocol['protocol_id'], 'runs': records,
        'by_condition': by_condition,
        'held_out_evaluated': False,
    }
    if len(by_condition) == 1:
        summary['metrics'] = next(iter(by_condition.values()))['metrics']
    if protocol.get('baseline_reference'):
        baseline = read(output / 'source' / protocol['baseline_reference'])
        control = {row['seed']: row for row in baseline['runs']}
        variants = {(row['condition'], row['seed']): row for row in records}
        effects = {}
        for label, coefficients in {
            'A_minus_baseline': {'A': 1, 'baseline': -1},
            'B_minus_baseline': {'B': 1, 'baseline': -1},
            'AB_minus_baseline': {'AB': 1, 'baseline': -1},
            'interaction_AB_minus_A_minus_B_plus_baseline': {'AB': 1, 'A': -1, 'B': -1, 'baseline': 1},
        }.items():
            rows = []
            for seed in sorted(control):
                row = {'seed': seed}
                for key in ('dice', 'iou', 'boundary_f1'):
                    row[key] = sum(weight * (control[seed] if condition == 'baseline'
                                   else variants[(condition, seed)])[key]
                                   for condition, weight in coefficients.items())
                rows.append(row)
            effects[label] = {'paired_seeds': rows, 'metrics': {key: {
                'mean': statistics.mean(row[key] for row in rows),
                'sample_std': statistics.stdev(row[key] for row in rows) if len(rows) > 1 else None,
            } for key in ('dice', 'iou', 'boundary_f1')}}
        summary['validation_factorial_effects'] = effects
        summary['comparison_note'] = 'Paired seed deltas on the validation set; not significance tests or held-out generalization.'
    write(output / 'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=Path, default=ROOT / 'configs/unet_baseline_protocol.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/unet_baseline_v1')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--status', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    protocol = read(args.protocol)
    validate_protocol(protocol)
    if args.status:
        print(json.dumps({'queue': read(output / 'status.json') if (output / 'status.json').exists() else None,
                          'runs': progress(protocol, output)}, indent=2, ensure_ascii=False))
        return
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.queue.lock').open('a+') as queue_lock:
        try:
            fcntl.flock(queue_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('A supervisor or training child already holds this queue lock')
        snapshot = prepare_snapshot(protocol, output)
        for entry in protocol['runs']:
            config = read(snapshot / entry['config'])
            if config['seed'] != entry['seed'] or config['epochs'] != entry['epochs']:
                raise ValueError('Run metadata and config disagree')
            if config['purpose'] != protocol['purpose']:
                raise ValueError('Run purpose differs from protocol')
            if config.get('condition', 'baseline') != entry.get('condition', 'baseline'):
                raise ValueError('Run condition differs from config')
        active = {'run': None, 'phase': None, 'child_pid': None}
        def status(state, **extra):
            write(output / 'status.json', {
                'state': state, 'updated_at_utc': datetime.now(timezone.utc).isoformat(),
                'supervisor_pid': os.getpid(), 'device': args.device,
                'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                'protocol_id': protocol['protocol_id'], **active,
                'runs': progress(protocol, output), **extra,
            })
        if args.prepare_only:
            status('prepared')
            print(f'Prepared frozen source snapshot: {snapshot}', flush=True)
            return
        def interrupted(signum, frame):
            raise KeyboardInterrupt(f'Received signal {signum}')
        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGINT, interrupted)
        def run_command(command, log_path):
            verify_files(snapshot, protocol['files_sha256'])
            with log_path.open('a') as log:
                log.write('\nSTART ' + datetime.now(timezone.utc).isoformat() + '\n')
                log.flush()
                # Child retains the lock even if the supervisor is killed unexpectedly.
                child = subprocess.Popen(command, cwd=snapshot, stdout=log, stderr=subprocess.STDOUT,
                                         pass_fds=(queue_lock.fileno(),))
                active['child_pid'] = child.pid
                status('running')
                try:
                    while child.poll() is None:
                        try:
                            child.wait(timeout=20)
                        except subprocess.TimeoutExpired:
                            status('running')
                except BaseException:
                    child.terminate()
                    try:
                        child.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait()
                    raise
                if child.returncode:
                    raise RuntimeError(f'{active["phase"]} failed with exit {child.returncode}; see {log_path}')
                active['child_pid'] = None
                verify_files(snapshot, protocol['files_sha256'])
        try:
            for entry in protocol['runs']:
                directory = output / entry['name']
                history = read(directory / 'history.json') if (directory / 'history.json').exists() else []
                epoch = history[-1]['epoch'] if history else 0
                if epoch > entry['epochs']:
                    raise ValueError('Existing run exceeded the frozen epoch budget')
                active.update(run=entry['name'], phase='training', child_pid=None)
                if epoch < entry['epochs']:
                    command = [sys.executable, str(snapshot / 'scripts/train.py'), '--config',
                               str(snapshot / entry['config']), '--output', str(directory), '--device', args.device]
                    if (directory / 'last.pt').exists():
                        command.extend(['--resume', str(directory / 'last.pt')])
                    run_command(command, output / f'{entry["name"]}_train.log')
                if not (directory / 'last.pt').is_file() or not (directory / 'best.pt').is_file():
                    raise ValueError('Completed history is missing checkpoints')
                active['phase'] = 'validation'
                if not (directory / 'validation/provenance.json').exists():
                    validation = directory / 'validation'
                    if validation.exists() and any(validation.iterdir()):
                        # Preserve a partial export, then rerun validation only.
                        validation.rename(directory / f'validation_incomplete_{time.time_ns()}')
                    run_command([sys.executable, str(snapshot / 'scripts/evaluate.py'), '--checkpoint',
                                 str(directory / 'best.pt'), '--output', str(validation), '--device', args.device],
                                output / f'{entry["name"]}_validation.log')
                status('running')
            aggregate(protocol, output)
            active.update(run=None, phase=None, child_pid=None)
            status('complete')
        except KeyboardInterrupt as error:
            status('interrupted', error=str(error))
            raise SystemExit(130)
        except Exception as error:
            status('failed', error=str(error))
            raise


if __name__ == '__main__':
    main()
