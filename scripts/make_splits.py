"""Deterministic split; optional publisher-verified patient/video map (id,group)."""
import argparse,csv,hashlib,json,random
from collections import defaultdict
from pathlib import Path

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--groups',type=Path)
    ap.add_argument('--similarity-pairs',type=Path,help='Reviewed candidate CSV with id_a,id_b; kept together conservatively, not patient IDs')
    a=ap.parse_args()
    cfg=json.loads(Path('configs/data.json').read_text());out=Path('data/splits');out.mkdir(exist_ok=True)
    if (out/'freeze.json').exists():
        raise SystemExit('Splits are frozen. Create a separately versioned protocol before changing them.')
    inv=Path('results/kvasir_seg/inventory.csv');rows=list(csv.DictReader(inv.open()))
    assert rows and len({r['id'] for r in rows})==len(rows)
    summary=json.loads(Path('results/kvasir_seg/summary.json').read_text())
    assert not any(summary[k] for k in ['read_errors','missing_masks','missing_images','duplicate_image_ids','duplicate_mask_ids','size_mismatch'])
    groups={}
    if a.groups:
        mapping=list(csv.DictReader(a.groups.open()));groups={r['id']:r['group'] for r in mapping}
        assert len(groups)==len(mapping) and set(groups)=={r['id'] for r in rows} and all(groups.values())
    # Union both known identity groups and exact image duplicates.
    parent={r['id']:r['id'] for r in rows}
    def root(k):
        while parent[k]!=k: parent[k]=parent[parent[k]];k=parent[k]
        return k
    seen={}
    for r in rows:
        keys=[('pixels',r['image_pixel_sha256'])]
        if groups:keys.append(('identity',groups[r['id']]))
        for key in keys:
            if key in seen:parent[root(r['id'])]=root(seen[key])
            else:seen[key]=r['id']
    similarity_pairs=[]
    if a.similarity_pairs:
        similarity_pairs=list(csv.DictReader(a.similarity_pairs.open()))
        for pair in similarity_pairs:
            if pair['id_a'] not in parent or pair['id_b'] not in parent:
                raise ValueError('Unknown ID in similarity pairs')
            parent[root(pair['id_a'])]=root(pair['id_b'])
    buckets=defaultdict(list)
    for r in rows:buckets[root(r['id'])].append(r)
    blocks=sorted(buckets.values(),key=lambda b:min(r['id'] for r in b));random.Random(cfg['seed']).shuffle(blocks)
    names=['train','val','test'];target=[int(len(rows)*x) for x in cfg['ratios']];target[-1]=len(rows)-sum(target[:-1]);parts=[[],[],[]]
    for b in blocks:
        dest=max(range(3),key=lambda i:target[i]-len(parts[i]));parts[dest].extend(b)
    assignment={r['id']:i for i,p in enumerate(parts) for r in p}
    for b in buckets.values():assert len({assignment[r['id']] for r in b})==1
    assert sum(map(len,parts))==len(rows)
    outputs={}
    for name,part in zip(names,parts):
        p=out/f'kvasir_{name}.csv'
        with p.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['id','image','mask','group']);w.writeheader()
            for r in sorted(part,key=lambda r:r['id']):w.writerow({**{k:r[k] for k in ['id','image','mask']},'group':root(r['id'])})
        outputs[p.name]={'count':len(part),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    meta={'seed':cfg['seed'],'requested_ratios':cfg['ratios'],'outputs':outputs,'inventory_sha256':hashlib.sha256(inv.read_bytes()).hexdigest(),
        'identity_grouping':bool(groups),'group_file_sha256':hashlib.sha256(a.groups.read_bytes()).hexdigest() if a.groups else None,
        'similarity_pairs':len(similarity_pairs),
        'similarity_pairs_sha256':hashlib.sha256(a.similarity_pairs.read_bytes()).hexdigest() if a.similarity_pairs else None,
        'method':'seeded shuffle of sorted union-find groups, allocate to largest remaining target deficit',
        'limitation':'No verified patient/video mapping supplied; heuristic similarity grouping cannot exclude all related frames or patient leakage.' if not groups else '',
        'external':'CVC-ClinicDB not downloaded; no external split fabricated; all future samples evaluation-only'}
    (out/'split_metadata.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta,indent=2))
if __name__=='__main__':main()
