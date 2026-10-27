"""Fresh P7 no-IDM B prefix; requires a pinned job and governed launch."""
import pathlib,sys,json,argparse,time,random,hashlib,datetime,os
ROOT=pathlib.Path(__file__).resolve().parents[1];UPSTREAM=next(ROOT.glob('vendor/H-JEPA-*'));sys.path[:0]=[str(ROOT),str(UPSTREAM/'h_jepa'),str(UPSTREAM)]
import torch,numpy as np
from omegaconf import OmegaConf
from torch.utils.data import DataLoader,Subset
from src.p1_runtime import Runtime,sha
from src.upstream_adapter import build_common_baseline
p=argparse.ArgumentParser();p.add_argument('--job',required=True);args=p.parse_args()
job_path=(ROOT/args.job).resolve();assert job_path.is_relative_to(ROOT)
job=json.loads(job_path.read_text());assert job['status']=='frozen' and job['stage']=='p7_b_prefix' and job['test_access'] is False
assert job['project_id']=='26bc4d31-99a3-433c-9035-1a5a7fcd5847' and job['arm']=='B' and job['steps']==3000
assert sha(ROOT/job['protocol_path'])==job['protocol_sha256'] and sha(ROOT/job['budget_authorization_path'])==job['budget_authorization_sha256']
protocol=json.loads((ROOT/job['protocol_path']).read_text());assert job['seed'] in protocol['training']['seeds']
auth=json.loads((ROOT/job['budget_authorization_path']).read_text());assert auth['status']=='user_approved' and auth['cap_seconds']==job['required_total_cap_seconds']
assert datetime.datetime.now(datetime.timezone.utc)<datetime.datetime.fromisoformat(auth['deadline_utc'])
recipe=json.loads((ROOT/'configs/p5-common-training.json').read_text())
assert sha(ROOT/'configs/p5-common-training.json')==job['common_recipe_sha256'] and recipe['steps']==3000
for item in job['materials']:
 path=(ROOT/item['path']).resolve();assert path.is_relative_to(ROOT) and sha(path)==item['sha256']
r=Runtime(ROOT);seed=job['seed'];name=job['run_name'];out=r.project/'runs'/name;assert out.parent==r.project/'runs'
marker=r.project/'budget-reservations'/f"{job['reservation_id']}.json"
guard=json.loads(marker.read_text());assert guard['status']=='consumed_before_launch' and guard['pid']==os.getppid()
out.mkdir(parents=True,exist_ok=False)
# Runtime provides immutable native config/data/normalizer; discard its P1 model and optimizer entirely.
del r.model,r.optim
r.settings={**r.settings,'scope':'P7 fresh neutral B prefix no-IDM','seed':seed,'batch_size':recipe['batch_size'],'warmup_steps':recipe['steps']}
r.cfg.seed=seed
r.settings['common_idm_coefficient']=0.0
random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
r.model,losses,r.cfg=build_common_baseline(r.cfg,0.,r.settings['common_idm_hidden']);r.model.cuda()
for k,v in losses.items():setattr(r,k,v.cuda())
r.optim=torch.optim.AdamW([{'params':r.model.get_level(l).parameters(),'lr':float(r.cfg[f'level{l}'].lr),'name':f'level{l}'} for l in [1,2]],weight_decay=float(r.cfg.optimizer.weight_decay))
generator=torch.Generator().manual_seed(seed+4000);loader=DataLoader(r.datasets['train'],batch_size=recipe['batch_size'],shuffle=True,drop_last=True,num_workers=0,generator=generator)
it=iter(loader);started=time.monotonic();r.model.train()
def weights_sha():
 h=hashlib.sha256()
 for n,p in r.model.named_parameters():h.update(n.encode());h.update(p.detach().cpu().numpy().tobytes())
 return h.hexdigest()
initial_sha=weights_sha()
def save(step,filename):
 tmp=out/(filename+'.tmp');torch.save({'model':r.model.state_dict(),'optimizer':r.optim.state_dict(),'step':step,'normalizer':r.normalizer,'config':OmegaConf.to_container(r.cfg,resolve=True),'settings':r.settings,'phase_recipe':recipe,'p7_job':job,'p7_job_sha256':sha(job_path),'seed':seed,'initialization':'fresh','initial_parameters_sha256':initial_sha,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'loader_rng':generator.get_state(),'data_inputs':r.inputs,'parent_checkpoint':None},tmp);tmp.replace(out/filename)
for step in range(1,recipe['steps']+1):
 try:batch=next(it)
 except StopIteration:it=iter(loader);batch=next(it)
 torch.manual_seed(seed*100000+step);torch.cuda.manual_seed_all(seed*100000+step)
 r.optim.zero_grad(set_to_none=True);o=r.forward(r.batch(batch),'train');assert not any(k.startswith('idm_loss') for k in o);o['loss'].backward();norm=torch.nn.utils.clip_grad_norm_(r.model.parameters(),recipe['gradient_clip_norm'],error_if_nonfinite=True)
 assert torch.isfinite(o['loss']);r.optim.step()
 row={'step':step,'seconds':time.monotonic()-started,'grad_norm':float(norm),**{k:float(v.detach()) for k,v in o.items() if 'loss' in k and v.numel()==1}}
 with (out/'train.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
 if step==1 or step%100==0:print(json.dumps({'seed':seed,**row}),flush=True)
 if step%recipe['checkpoint_every_steps']==0:save(step,'latest.pt')
save(recipe['steps'],'shared-warmup.pt');final_sha=weights_sha()
ds=r.datasets['dev'];lookup={tuple(x):i for i,x in enumerate(ds.clip_indices)};clipids=[(e,t) for e in range(32) for t in [40,240]]
validation_loader=DataLoader(Subset(ds,[lookup[x] for x in clipids]),batch_size=8,shuffle=False,num_workers=0)
def evaluate():
 r.model.eval();rows=[]
 with torch.no_grad():
  for k,batch in enumerate(validation_loader):
   torch.manual_seed(9011+k);torch.cuda.manual_seed_all(9011+k);o=r.forward(r.batch(batch),'dev');rows.append({k:float(v) for k,v in o.items() if 'loss' in k and v.numel()==1})
 return {k:float(np.mean([x[k] for x in rows])) for k in rows[0]}
raw_dev=evaluate();r.model.requires_grad_(False);r.model.eval();norms=[m for m in r.model.modules() if isinstance(m,torch.nn.modules.batchnorm._BatchNorm)]
for m in norms:m.reset_running_stats();m.momentum=None;m.train()
with torch.no_grad():
 for k,batch in enumerate(r.loader('train',seed=50131)):
  if k>=128:break
  torch.manual_seed(11013+k);torch.cuda.manual_seed_all(11013+k);r.forward(r.batch(batch),'bn_recalibration')
r.model.eval();assert weights_sha()==final_sha
buffers={k:v.cpu().clone() for k,v in r.model.state_dict().items() if k.endswith(('running_mean','running_var','num_batches_tracked'))};torch.save(buffers,out/'bn-buffers.pt');cal_dev=evaluate()
summary={'status':'complete','seed':seed,'steps':recipe['steps'],'batch_size':recipe['batch_size'],'initial_parameters_sha256':initial_sha,'final_parameters_sha256':final_sha,'elapsed_seconds':time.monotonic()-started,'raw_dev':raw_dev,'bn_recalibrated_dev':cal_dev,'bn_recalibration_batches':128,'world_model_parameters_unchanged_during_bn_recalibration':True,'shared_checkpoint_sha256':sha(out/'shared-warmup.pt'),'bn_buffers_sha256':sha(out/'bn-buffers.pt'),'recipe_sha256':sha(ROOT/'configs/p5-common-training.json'),'p7_job_sha256':sha(job_path),'p7_protocol_sha256':job['protocol_sha256'],'test_access':False,'closed_loop_evaluated':False,'idm_coefficient':0.,'parent_checkpoint':None,'optimizer_initialization':'fresh','performance_superiority_claim':False,'peak_allocated_bytes':torch.cuda.max_memory_allocated()}
with (out/'summary.json').open('x') as f:json.dump(summary,f,indent=2,allow_nan=False)
print(json.dumps(summary),flush=True)
