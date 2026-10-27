"""Validate a proposed dispatch and build argv; never reserve or launch work."""
import datetime as dt,math,pathlib,re
from src.p6_receipt_audit import PROJECT
WORKERS={'prefix':'scripts/train_p6_timing.py','tail':'scripts/train_p6_timing.py',
         'planning':'scripts/evaluate_p6_planning_item.py','position':'scripts/evaluate_p6_position.py',
         'parity':'scripts/validate_p6_model_parity.py','parity_diagnostic':'scripts/validate_p6_model_parity.py'}
LIMITS={'prefix':3150,'tail':1050,'planning':225,'position':1050,'parity':600,'parity_diagnostic':120}
DEADLINE='2026-10-15T16:30:34.028750+00:00'

def dispatch_arguments(*,project_path,job_path,job,authorization,ledger,historical_ledger,reservation,material_manifest,environment_manifest,now):
    if job.get('status')!='frozen' or job.get('stage') not in WORKERS or job.get('test_access') is not False:
        raise ValueError('Job is not frozen within P6 scope')
    if authorization.get('status')!='user_approved' or authorization.get('project_id')!=PROJECT:
        raise ValueError('Explicit project budget approval required')
    if ledger.get('project_id')!=PROJECT or reservation.get('project_id')!=PROJECT:
        raise ValueError('Cross-project budget identity')
    deadline_text=authorization['deadline_utc']
    if ledger['aggregate_deadline_utc']!=deadline_text or reservation['deadline_utc']!=deadline_text:
        raise ValueError('Deadline differs across approval, ledger, and reservation')
    if deadline_text!=DEADLINE:
        direct_extension=(authorization.get('extension_of_deadline_utc')==DEADLINE
                and authorization.get('previous_authorization_path')=='configs/p6-budget-211620s-approved.json')
        budget_after_extension=(authorization.get('previous_authorization_path')=='configs/p6-deadline-extension-20261022-approved.json'
                and authorization.get('previous_gpu_seconds_cap')==211620
                and authorization.get('cap_seconds')==243495
                and authorization.get('additional_gpu_seconds')==31875)
        recovery_budget_extension=(authorization.get('previous_authorization_path')=='configs/p6-budget-243495s-approved.json'
                and authorization.get('previous_gpu_seconds_cap')==243495
                and authorization.get('cap_seconds')==244995
                and authorization.get('additional_gpu_seconds')==1500)
        if (not (direct_extension or budget_after_extension or recovery_budget_extension)
                or not re.fullmatch(r'[0-9a-f]{64}',authorization.get('previous_authorization_sha256',''))):
            raise ValueError('Deadline changed without a separate protocol revision')
    deadline=dt.datetime.fromisoformat(deadline_text)
    if deadline.tzinfo is None or now.tzinfo is None or now>=deadline:raise ValueError('Expired or ambiguous time')
    cap=authorization['cap_seconds']
    if cap!=ledger['aggregate_cap_seconds'] or cap<job['required_total_cap_seconds']:
        raise ValueError('Authorized cap and retained ledger disagree')
    entries=ledger['entries'];ids=[e['reservation_id'] for e in entries]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate reservation IDs')
    if ledger.get('historical_p1_reservations_retained') is not True or ledger.get('prior_p1_conservative_seconds')!=900:
        raise ValueError('Historical P1 reservation missing')
    old_entries=historical_ledger['entries']
    if historical_ledger.get('project_id')!=PROJECT or historical_ledger.get('prior_p1_conservative_seconds')!=900 or 900+sum(e['reserved_gpu_seconds'] for e in old_entries)!=85200:
        raise ValueError('Unexpected historical baseline ledger')
    current={e['reservation_id']:e for e in entries}
    if any(current.get(e['reservation_id'])!=e for e in old_entries):raise ValueError('Historical reservation changed or removed')
    if any(e['project_id']!=PROJECT or type(e['reserved_gpu_seconds']) is not int or e['reserved_gpu_seconds']<=0 for e in entries):raise ValueError('Invalid reservation accounting')
    total=900+sum(e['reserved_gpu_seconds'] for e in entries)
    if total<85200 or total>cap:raise ValueError('Retained total outside approved bounds')
    rid=job['reservation_id']
    if not re.fullmatch(r'[A-Za-z0-9_-]+',rid):raise ValueError('Invalid reservation ID')
    rows=[e for e in entries if e['reservation_id']==rid]
    if len(rows)!=1 or rows[0]!=reservation:raise ValueError('Reservation not exactly present in ledger')
    allowed_seconds={LIMITS[job['stage']]}
    if job['stage']=='planning' and rid.startswith('p6-fresh-planning-cell'):
        allowed_seconds.add(255)
    if job['stage']=='planning' and rid=='p6-fresh-planning-cell26-recovery1':
        allowed_seconds.add(300)
    if reservation['status']!='authorized_ready' or reservation['counted_gpus']!=1 or reservation['reserved_gpu_seconds'] not in allowed_seconds:
        raise ValueError('Stage reservation mismatch')
    expires=dt.datetime.fromisoformat(reservation['reservation_expires_utc'])
    if (expires-now).total_seconds()<=60 or expires>deadline:raise ValueError('Reservation has no useful execution window')
    for value in (job_path,material_manifest,environment_manifest):
        p=pathlib.PurePosixPath(value)
        if p.is_absolute() or '..' in p.parts:raise ValueError('Project-relative configuration required')
    rp='configs/reservations/'+rid+'.json'
    common=['--project',str(project_path),'--experiment-id',rid,'--accelerator','cuda','--gpu-count','1',
            '--min-gpu-memory-gib','16','--estimated-minutes',str(math.ceil(reservation['reserved_gpu_seconds']/60)),
            '--config',job_path,'--material-manifest',material_manifest,'--environment-manifest',environment_manifest]
    child=['python','scripts/run_with_budget.py','--reservation',rp,'--','python',WORKERS[job['stage']],'--job',job_path]
    return {'preflight_arguments':common,'launch_arguments':common+['--']+child,
            'retained_gpu_seconds':total,'requires_verified_materialization_and_environment':True,
            'requires_live_global_resource_preflight':True,'launch_performed':False}


def bound_dispatch_arguments(root,spec,*,now):
    """Resolve pinned configuration bytes before building argv. No I/O dispatch.

    The caller must supply approval captured from the user, then perform live
    resource, deployment, materialization and environment checks separately.
    """
    import hashlib,json
    root=pathlib.Path(root).resolve()
    if spec.get('project_id')!=PROJECT or spec.get('status')!='dispatch_inputs_pinned':
        raise ValueError('Expected project-bound dispatch inputs')
    def read(ref,*,parse=True):
        rel=pathlib.PurePosixPath(ref['path']);path=(root/rel).resolve()
        if rel.is_absolute() or '..' in rel.parts or not path.is_relative_to(root):
            raise ValueError('Dispatch input outside project')
        data=path.read_bytes()
        if hashlib.sha256(data).hexdigest()!=ref['sha256']:raise ValueError('Dispatch input SHA mismatch')
        return json.loads(data) if parse else data
    names=('job','authorization','ledger','historical_ledger','reservation','material_manifest','environment_manifest')
    docs={name:read(spec[name]) for name in names}
    job=docs['job'];auth=spec['authorization'];rid=job['reservation_id']
    approval=docs['authorization']
    if approval['deadline_utc']!=DEADLINE:
        appendix=docs['ledger'].get('budget_authorization_appendix',{})
        if appendix.get('path')!=auth['path'] or appendix.get('sha256')!=auth['sha256']:
            raise ValueError('Extended approval differs from retained ledger appendix')
    if approval['deadline_utc']!=DEADLINE:
        previous={'path':approval['previous_authorization_path'],
                  'sha256':approval['previous_authorization_sha256']}
        old=read(previous)
        if old.get('status')!='user_approved' or old.get('project_id')!=PROJECT:
            raise ValueError('Invalid prior approval for deadline extension')
        if approval.get('previous_authorization_path')=='configs/p6-deadline-extension-20261022-approved.json':
            if (old.get('deadline_utc')!=approval['deadline_utc']
                    or old.get('cap_seconds')!=approval.get('previous_gpu_seconds_cap')
                    or approval.get('cap_seconds')-old['cap_seconds']!=approval.get('additional_gpu_seconds')):
                raise ValueError('Invalid GPU extension after deadline extension')
            original=read({'path':old['previous_authorization_path'],
                           'sha256':old['previous_authorization_sha256']})
            if (original.get('status')!='user_approved' or original.get('project_id')!=PROJECT
                    or original.get('deadline_utc')!=DEADLINE
                    or original.get('cap_seconds')!=old['cap_seconds']):
                raise ValueError('Invalid original approval for deadline extension')
        elif approval.get('previous_authorization_path')=='configs/p6-budget-243495s-approved.json':
            if (old.get('deadline_utc')!=approval['deadline_utc']
                    or old.get('cap_seconds')!=approval.get('previous_gpu_seconds_cap')
                    or approval.get('cap_seconds')-old['cap_seconds']!=approval.get('additional_gpu_seconds')
                    or old.get('previous_authorization_path')!='configs/p6-deadline-extension-20261022-approved.json'):
                raise ValueError('Invalid recovery budget extension chain')
            deadline_approval=read({'path':old['previous_authorization_path'],
                                    'sha256':old['previous_authorization_sha256']})
            if (deadline_approval.get('status')!='user_approved'
                    or deadline_approval.get('deadline_utc')!=approval['deadline_utc']
                    or deadline_approval.get('cap_seconds')!=old['previous_gpu_seconds_cap']):
                raise ValueError('Invalid underlying deadline extension')
        elif old.get('deadline_utc')!=DEADLINE or old.get('cap_seconds')!=approval.get('cap_seconds'):
            raise ValueError('Invalid prior approval for deadline extension')
    if job.get('budget_authorization_path')!=auth['path'] or job.get('budget_authorization_sha256')!=auth['sha256']:
        raise ValueError('Worker approval binding differs from dispatch approval')
    if spec['reservation']['path']!='configs/reservations/'+rid+'.json':
        raise ValueError('Dispatch reservation is not the worker reservation')
    manifest=docs['material_manifest'];env=docs['environment_manifest']
    if manifest.get('project_id')!=PROJECT or manifest.get('schema_version')!=2:
        raise ValueError('Expected same-project version2 material manifest')
    if env.get('project_id')!=PROJECT or env.get('schema_version')!=1 or env.get('kind')!='python-venv':
        raise ValueError('Expected same-project Python environment')
    if spec['environment_lock']['path']!=env['lock_file']:raise ValueError('Environment lock path mismatch')
    read(spec['environment_lock'],parse=False)
    material={}
    for item in manifest['materials']:
        dest=pathlib.PurePosixPath(item['destination'])
        if dest.is_absolute() or '..' in dest.parts or str(dest) in material:
            raise ValueError('Invalid or duplicate material destination')
        if not re.fullmatch('[0-9a-f]{64}',item['sha256']):raise ValueError('Unpinned material')
        material[str(dest)]=item['sha256']
    for item in job['materials']:
        if material.get(item['path'])!=item['sha256']:
            raise ValueError('Worker input not covered by dispatch material manifest')
    if job['stage'] in ('prefix','tail'):
        from src.p6_job_validation import validate_training_job
        if spec['timing_dag']['path']!=job.get('timing_dag') or material.get(job['timing_dag'])!=spec['timing_dag']['sha256']:
            raise ValueError('Timing graph not bound as worker material')
        validate_training_job(job,read(spec['timing_dag']))
    elif job['stage'] in ('parity','parity_diagnostic'):
        from src.p6_parity_job import validate_parity_job
        protocol=read({'path':job['protocol_path'],'sha256':job['protocol_sha256']})
        inputs=read({'path':job['inputs_path'],'sha256':material[job['inputs_path']]})
        validate_parity_job(job,protocol,inputs,material)
    else:
        from src.p6_evaluation_identity import validate_training_binding
        required=evaluation_material_references(job,material)
        training_audit=read(required[0])
        for item in ([job['input']] if job['stage']=='planning' else job['inputs']):
            validate_training_binding(training_audit,item)
    result=dispatch_arguments(project_path=root,job_path=spec['job']['path'],job=job,
        authorization=docs['authorization'],ledger=docs['ledger'],historical_ledger=docs['historical_ledger'],
        reservation=docs['reservation'],material_manifest=spec['material_manifest']['path'],
        environment_manifest=spec['environment_manifest']['path'],now=now)
    result['evidence_references']={name:spec[name] for name in (*names,'environment_lock')}
    if job['stage'] in ('prefix','tail'):result['evidence_references']['timing_dag']=spec['timing_dag']
    if job['stage'] in ('planning','position'):result['evaluation_material_references']=required
    result['status']='configuration_bound_live_checks_pending'
    return result


def evaluation_material_references(job,material):
    """Enumerate files actually read by evaluators, not just a caller's list."""
    from src.p6_schedule import TimingCell
    stage=job['stage']
    recipes={'planning':'configs/p5-planning-evaluation.json',
             'position':'configs/p5-position-staging-repair-v2-evaluation.json'}
    if stage not in recipes:raise ValueError('Not an evaluation stage')
    required=[{'path':job['training_audit_path'],'sha256':job['training_audit_sha256']},
              {'path':recipes[stage],'sha256':job['p5_recipe_sha256']}]
    inputs=[job['input']] if stage=='planning' else job['inputs']
    if not inputs:raise ValueError('Empty evaluation inputs')
    seen=set()
    for item in inputs:
        cell=TimingCell(**item['cell'])
        if cell in seen or item.get('source_kind')!='p6_tail':raise ValueError('Duplicate or non-tail evaluation input')
        seen.add(cell)
        required.extend([{'path':item['path'],'sha256':item['checkpoint_sha256']},
                         {'path':item['bn_path'],'sha256':item['bn_sha256']}])
    for ref in required:
        path=pathlib.PurePosixPath(ref['path'])
        if path.is_absolute() or '..' in path.parts or not ref['sha256'] or material.get(ref['path'])!=ref['sha256']:
            raise ValueError('Actual evaluator input missing or mismatched in material manifest')
    return required
