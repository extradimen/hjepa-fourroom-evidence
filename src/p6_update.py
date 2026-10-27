"""P5 update arithmetic reused for P6; schedule selects the active arm."""
import torch,numpy as np
from src.upstream_adapter import m1_auxiliary,m2_auxiliary
from src.mechanisms import project_upper_gradient
from src.p5_mechanisms import m3_replacement

def update(r,b,o,head,reference,coef,arm,seed,step):
 base=o['loss'];aux=torch.zeros((),device='cuda');clip=0.;conflict=None
 if arm in ['M2','B-same-branch-data']:
  base_value=.5*float(base.detach());(.5*base).backward();ids=(np.arange(8)+(step-1)*8)%64;branches=r.branch_batch(ids)
  for side in ['u','v']:
   branchloss=r.forward(branches[side])['loss'];base_value+=.25*float(branchloss.detach());(.25*branchloss).backward()
  if arm=='M2':
   aux=m2_auxiliary(r.model,head,reference,branches['u']['pixels_level1'][:,4],branches['u']['pixels_level1'][:,6],branches['v']['pixels_level1'][:,6],branches['u']['action_level1'][:,4:6],branches['v']['action_level1'][:,4:6],1.)
   (coef*aux).backward()
 elif arm=='M4':
  params=list(r.model.parameters());shared={id(x) for x in r.model.get_level(1).encoder.parameters()};lo=torch.autograd.grad(o['loss_level1'],params,retain_graph=True,allow_unused=True);hi=torch.autograd.grad(o['loss_level2'],params,allow_unused=True);indices=[i for i,x in enumerate(params) if id(x) in shared];ga=[lo[i] if lo[i] is not None else torch.zeros_like(params[i]) for i in indices];gb=[hi[i] if hi[i] is not None else torch.zeros_like(params[i]) for i in indices];dot=sum((a*c).sum() for a,c in zip(ga,gb));conflict=bool(dot<0);projected=dict(zip(indices,project_upper_gradient(ga,gb)))
  for i,param in enumerate(params):param.grad=projected[i].detach() if i in projected else (None if lo[i] is None and hi[i] is None else ((lo[i] if lo[i] is not None else torch.zeros_like(param))+(hi[i] if hi[i] is not None else torch.zeros_like(param))).detach())
  base_value=float(base.detach());del lo,hi,ga,gb,projected
 else:
  if arm=='M1':
   changed,clip=r.perturb(b['action_level1'],seed+5000+step);aux=m1_auxiliary(r.model,b['pixels_level1'],b['action_level1'],changed,1.)
  elif arm=='M3-R':aux=m3_replacement(r.model,o,head)
  (base+coef*aux).backward();base_value=float(base.detach())
 norm=torch.nn.utils.clip_grad_norm_(r.model.parameters(),1.,error_if_nonfinite=True);headnorm=float(torch.nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True)) if head else None
 assert np.isfinite(base_value) and torch.isfinite(aux);r.optim.step()
 return {'base_loss':base_value,'raw_aux_loss':float(aux.detach()),'weighted_aux_loss':coef*float(aux.detach()),'grad_norm':float(norm),'head_grad_norm':headnorm,'m4_conflict':conflict,'clipped_action_fraction':clip}
