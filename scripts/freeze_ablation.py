"""Freeze matched A/B/AB configurations, source and baseline references."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--pilot',action='store_true')
    args=parser.parse_args()
    phase='pilot' if args.pilot else 'formal'
    destination=ROOT/('configs/unet_ablation_pilot_protocol.json' if args.pilot else 'configs/unet_ablation_protocol.json')
    if destination.exists(): raise SystemExit('Frozen protocol exists; create a new version instead of overwriting')
    folder=ROOT/('configs/ablation_pilot' if args.pilot else 'configs/ablation')
    epochs=5 if args.pilot else 100
    purpose='ablation_pilot_not_formal' if args.pilot else 'formal_unet_ablation_v1'
    seeds=[20261020] if args.pilot else [20261020,20261021,20261022]
    runs=[]; configurations=[]
    for condition in ['A','B','AB']:
        for seed in seeds:
            config_path=folder/f'{condition}_seed{seed}.json'
            config=read(config_path)
            control=read(ROOT/f'configs/formal/unet_seed{seed}.json')
            expected={**control,'epochs':epochs,'purpose':purpose,'condition':condition,
                      'appearance_augmentation':config['appearance_augmentation'],
                      'boundary_supervision':config['boundary_supervision']}
            if config != expected: raise ValueError(f'Unplanned change relative to baseline: {config_path}')
            if config['appearance_augmentation']['enabled'] != ('A' in condition): raise ValueError('Factor A mismatch')
            if config['boundary_supervision']['enabled'] != ('B' in condition): raise ValueError('Factor B mismatch')
            configurations.append(config)
            runs.append({'name':f'{condition}_seed{seed}','condition':condition,'seed':seed,'epochs':epochs,
                         'config':str(config_path.relative_to(ROOT))})
    for key in ['appearance_augmentation','boundary_supervision']:
        enabled=[c[key] for c in configurations if c[key]['enabled']]
        if any(c != enabled[0] for c in enabled): raise ValueError(f'Factor settings differ across conditions: {key}')
    files=sorted((ROOT/'src').glob('*.py'))+sorted((ROOT/'scripts').glob('*.py'))
    files+=sorted(folder.glob('*.json'))+sorted((ROOT/'data/splits').glob('*.csv'))+sorted((ROOT/'data/splits').glob('*.json'))
    files+=[ROOT/'requirements.txt']
    protocol={'protocol_id':f'unet_ablation_{phase}_v1_20261021','status':'training_frozen',
              'frozen_at_utc':datetime.now(timezone.utc).isoformat(),'purpose':purpose,'evaluate_test':False,
              'runs':runs,'validation_manifest_sha256':read(ROOT/'data/splits/freeze.json')['manifests']['val']['sha256'],
              'selection':'Maximum native-resolution validation Dice; earliest tie; no early stopping',
              'budget':f'{epochs} epochs per run, from scratch; only A/B factors differ from baseline',
              'initialization':'Shared backbone/segmentation initialization and CPU RNG match baseline; auxiliary head uses forked RNG',
              'appearance_rng':'Blake2b of current post-flip CPU RNG state seeds local generator; global RNG unchanged',
              'boundary_definition':'Inner radius-2 contour at training resolution; outside-image background',
              'validation_loss':'Segmentation BCE+Dice only; training total adds 0.2 times boundary BCE(pos_weight=5)+Dice',
              'held_out_evaluated':False}
    if not args.pilot:
        reference=ROOT/'results/unet_baseline_v1/summary.json'
        baseline=read(reference)
        if baseline['status']!='complete' or baseline['held_out_evaluated']: raise ValueError('Baseline is not eligible')
        if {r['seed'] for r in baseline['runs']}!=set(seeds): raise ValueError('Baseline seeds differ')
        pilot=read(ROOT/'results/unet_ablation_pilot_v1/summary.json')
        if pilot['status']!='complete' or {r['condition'] for r in pilot['runs']}!={'A','B','AB'}:
            raise ValueError('All three pilots must complete before formal freeze')
        protocol['baseline_reference']=str(reference.relative_to(ROOT))
        files.append(reference)
        checks=['results/ablation_preflight/acceptance.json',
                'results/ablation_preflight/overfit_B/report.json',
                'results/ablation_preflight/resume_AB/epoch_resume.json']
        for name in checks:
            if read(ROOT/name).get('passed') is not True: raise ValueError(f'Preflight not passed: {name}')
        protocol['preflight_sha256']={name:sha(ROOT/name) for name in checks}
        protocol['pilot_summary_sha256']=sha(ROOT/'results/unet_ablation_pilot_v1/summary.json')
    protocol['files_sha256']={str(p.relative_to(ROOT)):sha(p) for p in files}
    destination.write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n')
    print(f'Frozen {phase}: {len(runs)} runs, {len(files)} files, {destination}')


if __name__=='__main__': main()
