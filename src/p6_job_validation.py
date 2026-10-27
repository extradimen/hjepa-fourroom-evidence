"""Fail-closed project-local input binding for proposed P6 workers."""
import hashlib,json,pathlib

def validate_inputs(root,job):
 root=pathlib.Path(root).resolve()
 required={'coefficient_source','neutral_checkpoint','neutral_bn','timing_dag'}
 if job['stage']=='tail':required|={'parent_checkpoint','parent_bn'}
 if job['stage']=='tail' and job.get('parent_kind')=='p6_prefix':required.add('parent_audit')
 material={}
 for x in job['materials']:
  rel=x['path'];p=(root/rel).resolve()
  if pathlib.Path(rel).is_absolute() or not p.is_relative_to(root):raise ValueError('Input outside deployment')
  if rel in material:raise ValueError('Duplicate material binding')
  if hashlib.sha256(p.read_bytes()).hexdigest()!=x['sha256']:raise ValueError('Input SHA mismatch')
  material[rel]=x['sha256']
 for name in required:
  if job.get(name) not in material:raise ValueError('Used input omitted from material binding: '+name)
 validate_training_job(job,json.loads((root/job['timing_dag']).read_text()))
 if 'parent_audit' in required:
  audit=json.loads((root/job['parent_audit']).read_text())
  validate_parent_audit(audit,job,material)
 return material


def select_calibration(document, *, seed, checkpoint_sha256, bn_sha256, mechanism):
 import math
 rows=[x for x in document['results'] if x.get('seed')==seed]
 if len(rows)!=1:raise ValueError('Calibration seed must have exactly one record')
 cal=rows[0]
 if cal.get('state_unchanged') is not True or cal.get('checkpoint_sha256')!=checkpoint_sha256 or cal.get('bn_sha256')!=bn_sha256:
  raise ValueError('Calibration does not match the neutral checkpoint and BN')
 if cal.get('initial_head_seed')!=2011:raise ValueError('Unexpected head initialization seed')
 if mechanism in ('M1','M2','M3-R'):
  entry=cal['coefficients'][mechanism];value=entry['coefficient']
  if entry.get('status')!='ready' or not math.isfinite(value) or not 1e-6<=value<=1e6:
   raise ValueError('Invalid fixed calibration coefficient')
  return value
 return 0.


def validate_parent_audit(audit,job,material):
 from src.p6_schedule import TimingCell
 from src.p6_receipt_audit import PROJECT
 if audit.get('project_id')!=PROJECT or audit.get('stage')!='prefix' or audit.get('status')!='files_os_and_standard_transfer_audited_payload_pending':
  raise ValueError('Parent lacks a prefix OS/transfer audit')
 rows=[x for x in audit.get('audits',[]) if x.get('id')==job.get('parent_id')]
 if len(rows)!=1:raise ValueError('Parent run audit missing or ambiguous')
 parent_sha=rows[0]['evidence_references']['job']['sha256']
 matches=[x for x in audit.get('checkpoints',[]) if x.get('job_sha256')==parent_sha]
 if len(matches)!=1:raise ValueError('Parent checkpoint binding missing or ambiguous')
 binding=matches[0]
 if binding.get('source_kind')!='p6_prefix' or binding.get('prefix_key')!=list(TimingCell(**job['cell']).prefix_key()):
  raise ValueError('Parent timing prefix mismatch')
 for field,sha_field in [('parent_checkpoint','checkpoint_sha256'),('parent_bn','bn_sha256')]:
  if material[job[field]]!=binding.get(sha_field):raise ValueError('Parent artifact differs from audited prefix')
 # Optimizer/head state is separately checked by restore().
 return binding


def validate_parent_payload(payload, audit, job, material):
 binding=validate_parent_audit(audit,job,material)
 parent=payload.get('job',{})
 if payload.get('job_sha256')!=binding['job_sha256']:
  raise ValueError('Parent payload job differs from audited job')
 if parent.get('status')!='frozen' or parent.get('stage')!='prefix' or parent.get('test_access') is not False:
  raise ValueError('Parent payload is not a frozen prefix')
 if parent.get('run_name')!=job['parent_id'] or parent.get('cell')!=binding['cell']:
  raise ValueError('Parent payload cell or run differs from audited prefix')
 if payload.get('step')!=3000 or list(payload.get('prefix_key',[]))!=binding['prefix_key']:
  raise ValueError('Parent payload boundary or timing differs from audit')


def validate_training_job(job,graph):
 """A frozen job must implement exactly one planned prefix or tail."""
 from src.p6_schedule import TimingCell
 stage=job.get('stage')
 if stage not in ('prefix','tail'):raise ValueError('Not a training job')
 rows=graph['prefixes' if stage=='prefix' else 'tails']
 matches=[x for x in rows if x['id']==job.get('run_name')]
 if len(matches)!=1:raise ValueError('Job absent or duplicated in timing graph')
 planned=matches[0]
 if planned['stage']!=stage or job.get('cell')!=planned['cell']:
  raise ValueError('Job timing differs from planned condition')
 cell=TimingCell(**job['cell'])
 if stage=='prefix':
  if planned['steps']!=3000 or 3001 in (cell.aux_start,cell.mechanism_start,cell.data_start):
   raise ValueError('Invalid canonical prefix')
 else:
  if planned['steps']!=1000 or any(job.get(k)!=planned[k] for k in ('parent_kind','parent_id')):
   raise ValueError('Tail ancestry differs from timing graph')
  if job['parent_kind']=='p6_prefix':
   parents=[x for x in graph['prefixes'] if x['id']==job['parent_id']]
   if len(parents)!=1 or TimingCell(**parents[0]['cell']).prefix_key()!=cell.prefix_key():
    raise ValueError('Tail references incompatible prefix')
  elif job['parent_kind']!='p5_neutral' or cell.prefix_key()!=(cell.seed,0,None,False) or job['parent_id']!=f'p5-common-seed{cell.seed}-v1':
   raise ValueError('Invalid neutral parent')
 return planned
