"""Shared native model/data setup for matched P1 arms; no test split access."""
import hashlib,json,pathlib,random
import numpy as np
import torch,h5py,hdf5plugin
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from data import build_hdf5_dataset,Compose,get_column_normalizer_from_artifact
from hjepa_utils import hjepa_forward
from src.upstream_adapter import build_common_baseline


def sha(path):
 h=hashlib.sha256()
 with pathlib.Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()


class Runtime:
 def __init__(self,root):
  self.root=root;self.project=root.parents[1]
  if self.project.name!='26bc4d31-99a3-433c-9035-1a5a7fcd5847':raise RuntimeError('Wrong project')
  if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('One governed GPU required')
  torch.set_num_threads(2);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
  self.settings=json.loads((root/'configs/p1-training-settings.json').read_text())
  self.inputs=json.loads((root/'configs/p1-arm-inputs.json').read_text())
  for f in self.inputs['files']:
   if sha(root/f['path'])!=f['sha256']:raise RuntimeError('Bad input checksum '+f['path'])
  random.seed(11);np.random.seed(11);torch.manual_seed(11);torch.cuda.manual_seed_all(11)
  ckpt=torch.load(root/'materials/checkpoints/shared-warmup.pt',map_location='cpu',weights_only=False)
  self.cfg=OmegaConf.create(ckpt['config']);self.normalizer=ckpt['normalizer']
  self.model,losses,self.cfg=build_common_baseline(self.cfg,self.settings['common_idm_coefficient'],self.settings['common_idm_hidden'])
  self.model.load_state_dict(ckpt['model'],strict=True);self.model.cuda()
  for k,v in losses.items():setattr(self,k,v.cuda())
  self.optim=torch.optim.AdamW([{'params':self.model.get_level(l).parameters(),'lr':float(self.cfg[f'level{l}'].lr),'name':f'level{l}'} for l in (1,2)],weight_decay=float(self.cfg.optimizer.weight_decay))
  self.optim.load_state_dict(ckpt['optimizer'])
  self.datasets={}
  dc=OmegaConf.to_container(self.cfg.data.dataset,resolve=True);dc.pop('val_total_transitions',None)
  transform=Compose(get_column_normalizer_from_artifact(self.normalizer,'action','action'))
  for name in ['train','dev']:
   dc['name']='p1_'+name;ds=build_hdf5_dataset(dc,cache_dir=root/self.settings['dataset'])
   ds.clip_indices=[(ep,t) for ep,t in ds.clip_indices if t+ds.last_level['span']<int(ds.lengths[ep])]
   ds.transform=transform;self.datasets[name]=ds
  self.mean=torch.tensor([.485,.456,.406],device='cuda').reshape(1,1,3,1,1);self.std=torch.tensor([.229,.224,.225],device='cuda').reshape(1,1,3,1,1)
  self.action_mean=torch.as_tensor(self.normalizer['stats']['action']['mean'],device='cuda').reshape(1,1,2)
  self.action_std=torch.as_tensor(self.normalizer['stats']['action']['std'],device='cuda').reshape(1,1,2)
  with h5py.File(root/'data/p1-branches-v1/branches.h5','r') as f:self.branches={k:f[k][:] for k in ['pixels_u','pixels_v','actions_u','actions_v','ancestor_seed']}
 def log_dict(self,*a,**kw):pass
 def pixels(self,x):
  if x.ndim==4:return self.pixels(x[:,None])[:,0]
  if x.shape[-1]==3:x=x.permute(0,1,4,2,3)
  return (x.cuda().float()/255-self.mean)/self.std
 def batch(self,b):return {'pixels_level1':self.pixels(b['pixels_level1']),'action_level1':b['action_level1'].cuda()}
 def forward(self,b,stage='train'):return hjepa_forward(self,b,stage,self.cfg,normalize_batch=lambda _:None)
 def loader(self,split,seed=4011,shuffle=True):return DataLoader(self.datasets[split],batch_size=8,shuffle=shuffle,drop_last=True,num_workers=0,generator=torch.Generator().manual_seed(seed))
 def branch_batch(self,indices):
  out={}
  for side in ['u','v']:
   pix=self.pixels(torch.from_numpy(self.branches['pixels_'+side][indices]))
   raw=torch.from_numpy(self.branches['actions_'+side][indices]).cuda();actions=((raw.reshape(-1,35,2)-self.action_mean)/self.action_std).reshape(-1,7,10)
   out[side]={'pixels_level1':pix,'action_level1':actions}
  return out
 def perturb(self,normalized,seed):
  generator=torch.Generator(device='cuda').manual_seed(seed)
  raw=normalized.reshape(-1,35,2)*self.action_std+self.action_mean
  proposal=raw[:,20:30]+torch.randn(raw[:,20:30].shape,device='cuda',generator=generator)*self.action_std*.1
  changed=raw.clone();changed[:,20:30]=proposal.clamp(-1,1)
  # Preserve exact common prefix bit patterns rather than roundtrip normalization.
  result=normalized.clone();result[:,4:6]=((changed[:,20:30]-self.action_mean)/self.action_std).reshape(-1,2,10)
  return result,float((proposal.abs()>1).float().mean())
