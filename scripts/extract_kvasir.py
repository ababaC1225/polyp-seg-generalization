"""Validate ZIP CRC and safe paths; never overwrite an existing extraction."""
import datetime,hashlib,json,stat,zipfile
from pathlib import Path
p=Path('data/raw/kvasir-seg.zip');dest=Path('data/raw/kvasir_seg')
if dest.exists():raise SystemExit('Extraction already exists; refusing to overwrite raw data')
with zipfile.ZipFile(p) as z:
    for info in z.infolist():
        target=(dest/info.filename).resolve()
        if not target.is_relative_to(dest.resolve()) or stat.S_ISLNK(info.external_attr>>16):raise SystemExit('Unsafe archive member')
    assert z.testzip() is None,'ZIP CRC validation failed'
    names=z.namelist();z.extractall(dest)
r={'checked_at':datetime.datetime.now().astimezone().isoformat(),'kvasir_seg':{'url':'https://datasets.simula.no/downloads/kvasir-seg.zip','archive':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'zip_crc':'passed','member_count':len(names),'tls':'verified using system roots plus verified Sectigo intermediate; project-local bundle'},'cvc_clinicdb':{'status':'not_downloaded','reason':'Official current page has no download target; legacy URL HTTP 404'}}
Path('docs/downloads.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(r,ensure_ascii=False,indent=2));print('Non-image entries:',[n for n in names if not n.lower().endswith(('.jpg','.jpeg','.png','/'))])
