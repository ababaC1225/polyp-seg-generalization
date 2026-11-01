"""One CPU batch only; no model, optimizer, training or external data."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from torch.utils.data import DataLoader
from src.dataset import PolypDataset
cfg=json.loads(Path('configs/data.json').read_text())
batch=next(iter(DataLoader(PolypDataset('data/splits/kvasir_train.csv',cfg['smoke_size']),batch_size=4,shuffle=False,num_workers=0)))
x,y=batch['image'],batch['mask']
assert x.shape==(4,3,256,256) and y.shape==(4,1,256,256)
assert torch.isfinite(x).all() and torch.isfinite(y).all()
assert x.dtype==y.dtype==torch.float32 and 0<=x.min()<=x.max()<=1
assert set(y.unique().tolist())<={0.,1.} and (y.sum((1,2,3))>0).all()
report={'status':'passed','batches_read':1,'device':str(x.device),'image_shape':list(x.shape),'mask_shape':list(y.shape),'dtype':str(x.dtype),'image_range':[float(x.min()),float(x.max())],'mask_values':y.unique().tolist(),'ids':batch['id'],'training_started':False}
Path('results/smoke_test.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
