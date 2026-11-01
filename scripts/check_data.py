"""Read-only inventory, exact duplicate checks and deterministic QC overlays."""
import argparse, csv, hashlib, json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

EXT = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def scan(folder):
    rows, errors = [], []
    for p in sorted(folder.rglob('*')):
        if not p.is_file(): continue
        if p.suffix.lower() not in EXT:
            errors.append({'path': str(p), 'issue': 'unexpected file type'}); continue
        try:
            with Image.open(p) as im: im.verify()
            with Image.open(p) as im:
                im.load(); arr = np.array(im.convert('RGB'))
                rows.append({'id':p.stem,'path':str(p),'width':im.width,'height':im.height,
                    'format':im.format,'mode':im.mode,'sha256':sha(p),
                    'pixel_sha256':hashlib.sha256(str(arr.shape).encode()+arr.tobytes()).hexdigest()})
        except Exception as e: errors.append({'path':str(p),'issue':str(e)})
    return rows,errors

def duplicates(rows, key):
    groups=defaultdict(list)
    for row in rows: groups[row[key]].append(row['path'])
    return [v for v in groups.values() if len(v)>1]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--images',type=Path,required=True);ap.add_argument('--masks',type=Path,required=True);ap.add_argument('--name',required=True);a=ap.parse_args()
    if not a.images.is_dir() or not a.masks.is_dir(): raise SystemExit('Missing dataset directories')
    out=Path('results')/a.name;out.mkdir(parents=True,exist_ok=True)
    ir,ie=scan(a.images);mr,me=scan(a.masks)
    ids_i=Counter(r['id'] for r in ir);ids_m=Counter(r['id'] for r in mr)
    ii={r['id']:r for r in ir};mm={r['id']:r for r in mr}
    mismatch=[];empty=[];nonbinary=[];pairs=[]
    for key in sorted(ii.keys() & mm.keys()):
        i,m=ii[key],mm[key]
        if (i['width'],i['height'])!=(m['width'],m['height']): mismatch.append(key);continue
        with Image.open(m['path']) as im: raw=np.array(im.convert('L'))
        mask=raw>=128
        if not mask.any():empty.append(key)
        if not np.isin(raw,[0,255]).all():nonbinary.append(key)
        pairs.append({'id':key,'image':i['path'],'mask':m['path'],'width':i['width'],'height':i['height'],
                      'foreground_fraction':float(mask.mean()),'image_sha256':i['sha256'],'image_pixel_sha256':i['pixel_sha256'],
                      'mask_sha256':m['sha256']})
    summary={'dataset':a.name,'images':len(ir),'masks':len(mr),'paired':len(pairs),
       'missing_masks':sorted(ii.keys()-mm.keys()),'missing_images':sorted(mm.keys()-ii.keys()),
       'duplicate_image_ids':[k for k,v in ids_i.items() if v>1], 'duplicate_mask_ids':[k for k,v in ids_m.items() if v>1],
       'read_errors':ie+me,'size_mismatch':mismatch,'empty_masks_at_128':empty,'nonbinary_masks':len(nonbinary),
       'mask_policy':'grayscale >=128; applied in memory only; JPEG masks may contain compression levels',
       'formats_images':dict(Counter(r['format'] for r in ir)), 'formats_masks':dict(Counter(r['format'] for r in mr)),
       'image_sizes':dict(Counter(f"{r['width']}x{r['height']}" for r in ir)),
       'mask_sizes':dict(Counter(f"{r['width']}x{r['height']}" for r in mr)),
       'duplicates':{label:{key:duplicates(rows,key) for key in ['sha256','pixel_sha256']} for label,rows in [('images',ir),('masks',mr)]}}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    (out/'files.json').write_text(json.dumps({'images':ir,'masks':mr},indent=2)+'\n')
    if pairs:
        with (out/'inventory.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(pairs[0]));w.writeheader();w.writerows(pairs)
        # Fixed evenly spaced IDs, no subjective selection or use for tuning.
        selected=[pairs[int(i)] for i in np.linspace(0,len(pairs)-1,min(6,len(pairs)))]
        canvas=Image.new('RGB',(768,280*len(selected)),'white');draw=ImageDraw.Draw(canvas)
        for n,r in enumerate(selected):
            with Image.open(r['image']) as im: rgb=im.convert('RGB');rgb.thumbnail((256,256))
            with Image.open(r['mask']) as im: mask=im.convert('L').point(lambda x:255 if x>=128 else 0).resize(rgb.size,Image.Resampling.NEAREST)
            overlay=np.array(rgb).copy();sel=np.array(mask)>0
            overlay[sel]=(0.55*overlay[sel]+0.45*np.array([255,0,0])).astype('uint8')
            for j,im in enumerate([rgb,mask.convert('RGB'),Image.fromarray(overlay)]):canvas.paste(im,(j*256,n*280+24))
            draw.text((4,n*280+4),r['id'],fill='black')
        canvas.save(out/'qc_overlays.png')
        (out/'qc_ids.json').write_text(json.dumps([r['id'] for r in selected],indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ['image_sizes','mask_sizes','duplicates']},ensure_ascii=False))
    if ie or me or mismatch or summary['missing_masks'] or summary['missing_images'] or summary['duplicate_image_ids'] or summary['duplicate_mask_ids']:raise SystemExit('Integrity check failed; inspect summary')
if __name__=='__main__':main()
