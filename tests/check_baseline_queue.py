"""Exercise the actual frozen queue, duplicate lock, exports and restart on CPU."""
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np
from PIL import Image

ROOT = Path.cwd()
REPORT = ROOT / 'results/formal_preflight'
REPORT.mkdir(parents=True, exist_ok=True)

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

with tempfile.TemporaryDirectory(prefix='queue_check_', dir=ROOT / '.local') as temporary:
    work = Path(temporary)
    rows = []
    rng = np.random.default_rng(7)
    for index in range(6):
        rgb = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
        mask = np.zeros((32,32), dtype=np.uint8)
        mask[8:24, 8:24] = 255
        image, truth = work / f'image{index}.png', work / f'mask{index}.png'
        Image.fromarray(rgb).save(image)
        Image.fromarray(mask).save(truth)
        rows.append({'id':str(index),'image':str(image),'mask':str(truth)})
    manifests = {}
    fixture_files = []
    for name, records in [('train',rows[:4]),('val',rows[4:])]:
        path=work/f'{name}.csv'
        with path.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=['id','image','mask'])
            writer.writeheader(); writer.writerows(records)
        manifests[name]={'path':str(path.relative_to(ROOT)),'sha256':digest(path),'count':len(records)}
        fixture_files.append(path)
    lock=work/'freeze.json'
    lock.write_text(json.dumps({'manifests':manifests,'size_thresholds':[0.1,0.3]}))
    fixture_files.append(lock)
    runs=[]
    for seed in [71,72]:
        config=json.loads((ROOT/'configs/unet_pilot.json').read_text())
        config.update(base_channels=2,image_size=[32,32],batch_size=2,epochs=2,num_workers=0,
                      cpu_threads=2,amp=False,seed=seed,purpose='queue_integration_test',
                      split_lock=str(lock.relative_to(ROOT)),train_manifest=manifests['train']['path'],
                      val_manifest=manifests['val']['path'])
        path=work/f'config{seed}.json'; path.write_text(json.dumps(config))
        fixture_files.append(path)
        runs.append({'name':f'seed{seed}','seed':seed,'epochs':2,'config':str(path.relative_to(ROOT))})
    files=list((ROOT/'src').glob('*.py'))+list((ROOT/'scripts').glob('*.py'))+fixture_files
    protocol={'protocol_id':'queue_integration_test','status':'training_frozen','evaluate_test':False,
              'purpose':'queue_integration_test','runs':runs,
              'validation_manifest_sha256':manifests['val']['sha256'],
              'files_sha256':{str(p.relative_to(ROOT)):digest(p) for p in files}}
    protocol_path=work/'protocol.json'; protocol_path.write_text(json.dumps(protocol))
    output=work/'output'
    command=[sys.executable,'scripts/run_baselines.py','--protocol',str(protocol_path),'--output',str(output),'--device','cpu']
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='')
    with (REPORT/'queue_integration.log').open('w') as log:
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env)
        try:
            deadline=time.monotonic()+60
            while not (output/'status.json').exists() and child.poll() is None and time.monotonic()<deadline:
                time.sleep(0.1)
            assert child.poll() is None, 'Queue ended before duplicate-lock check'
            duplicate=subprocess.run(command,capture_output=True,text=True,env=env)
            assert duplicate.returncode!=0 and 'already holds this queue lock' in duplicate.stderr
            assert child.wait(timeout=180)==0, (REPORT/'queue_integration.log').read_text()
        finally:
            if child.poll() is None:
                child.terminate(); child.wait(timeout=30)
    state=json.loads((output/'status.json').read_text())
    summary=json.loads((output/'summary.json').read_text())
    assert state['state']=='complete' and len(summary['runs'])==2
    assert all(r['completed_epochs']==2 and r['validation_exported'] for r in state['runs'])
    before={str(p):digest(p) for p in output.glob('seed*/*.pt')}
    assert subprocess.run(command,env=env,capture_output=True,text=True).returncode==0
    assert before=={str(p):digest(p) for p in output.glob('seed*/*.pt')}
    target=output/'source/src/model.py'
    target.write_text(target.read_text()+'\n# tamper probe\n')
    tampered=subprocess.run(command,capture_output=True,text=True,env=env)
    assert tampered.returncode!=0 and 'Frozen file changed' in tampered.stderr
    report={'passed':True,'data':'6 synthetic images only','seeds':2,'epochs_per_seed':2,
            'training_and_validation_completed':True,'duplicate_supervisor_blocked':True,
            'completed_restart_did_not_retrain':True,'source_tampering_blocked':True,
            'held_out_evaluated':False}
    (REPORT/'queue_integration.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
