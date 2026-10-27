"""Check native per-task arithmetic; OS, transfer and tensor audits are separate."""
import math
from src.p6_schedule import TimingCell


def planning_outcomes(summary,chunks,*,item,recipe,recipe_sha256,job_sha256,task_sha256):
    """Chunks are in frozen task order, never filtered by success or runtime."""
    TimingCell(**item['cell'])
    expected={'status':'complete','cell':item['cell'],'source_kind':'p6_tail',
        'checkpoint_sha256':item['checkpoint_sha256'],'bn_sha256':item['bn_sha256'],
        'training_job_sha256':item['job_sha256'],'job_sha256':job_sha256,
        'recipe_sha256':recipe_sha256,'task_sha256':task_sha256,
        'task_count':32,'test_access':False,'native_success_rate_unit':'percent'}
    if any(summary.get(k)!=v for k,v in expected.items()):raise ValueError('Planning summary identity mismatch')
    if item.get('source_kind')!='p6_tail' or recipe.get('status')!='frozen' or recipe.get('test_access') is not False:
        raise ValueError('Unexpected planning source or recipe')
    if recipe['episodes']!=list(range(32)) or len(chunks)!=32:
        raise ValueError('All 32 fixed single-task chunks are required')
    successes=[];resets=[]
    for episode,chunk in zip(recipe['episodes'],chunks):
        values=chunk['episode_successes'];seeds=chunk['seeds']
        if len(values)!=1 or type(values[0]) is not bool or seeds!=[recipe['reset_seed_start']+episode]:
            raise ValueError('Wrong task order, reset identity or success type')
        if chunk['loaded_eval_original_lengths']!=[recipe['physical_goal_offset']+1]:
            raise ValueError('Wrong original task length')
        expected_rate=100. if values[0] else 0.
        if not math.isclose(chunk['success_rate'],expected_rate,abs_tol=1e-9):raise ValueError('Chunk success arithmetic mismatch')
        steps=chunk['steps_to_success'];wall=chunk['wall_clock_to_success']
        if len(steps)!=1 or len(wall)!=1:raise ValueError('Incomplete per-task timing')
        if values[0]:
            if not math.isfinite(steps[0]) or not 0<=steps[0]<=recipe['eval_budget'] or not math.isfinite(wall[0]) or wall[0]<0:
                raise ValueError('Invalid successful task timing')
        elif not math.isnan(steps[0]) or not math.isnan(wall[0]):
            raise ValueError('Failed tasks must retain native missing success timing')
        successes.extend(values);resets.extend(seeds)
    rate=100*sum(successes)/32
    if not math.isclose(summary['native_success_rate'],rate,abs_tol=1e-9):raise ValueError('Summary success arithmetic mismatch')
    return {'cell':item['cell'],'episode_successes':successes,'reset_seeds':resets,
        'successes':sum(successes),'task_count':32,'success_rate_percent':rate,
        'evidence_status':'native_task_arithmetic_checked_pending_provenance_and_payload',
        'remaining':['SHA and semantic equality of task tensors','OS and budget child waits','complete governed return coverage']}


def audit_planning_files(root,spec):
    """Inspect SHA-pinned, same-project files; does not certify OS/transfer."""
    import hashlib,io,json,pathlib,torch
    from src.p6_receipt_audit import PROJECT
    from src.p6_task_identity import validate_tasks
    root=pathlib.Path(root).resolve()
    if spec.get('project_id')!=PROJECT:raise ValueError('Wrong planning project')
    def data(ref):
        path=(root/ref['path']).resolve()
        if pathlib.Path(ref['path']).is_absolute() or not path.is_relative_to(root):raise ValueError('Planning artifact outside project')
        raw=path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=ref['sha256']:raise ValueError('Planning artifact SHA mismatch')
        return raw
    job=json.loads(data(spec['job']));recipe=json.loads(data(spec['recipe']))
    if job.get('status')!='frozen' or job.get('stage')!='planning' or job.get('test_access') is not False or job.get('p5_recipe_sha256')!=spec['recipe']['sha256']:
        raise ValueError('Planning job or recipe mismatch')
    from src.p6_evaluation_identity import validate_training_binding
    if job.get('training_audit_path')!=spec['training_audit']['path'] or job.get('training_audit_sha256')!=spec['training_audit']['sha256']:
        raise ValueError('Planning job training audit binding mismatch')
    validate_training_binding(json.loads(data(spec['training_audit'])),job['input'])
    directory=pathlib.PurePosixPath(spec['directory'])
    expected=[str(directory/'native-results'/'chunks'/f'tasks_{i:03d}-{i+1:03d}.json') for i in range(32)]
    if [r['path'] for r in spec['chunks']]!=expected or spec['summary']['path']!=str(directory/'summary.json') or spec['tasks']['path']!=str(directory/'tasks.pt'):
        raise ValueError('Unexpected planning file layout or task coverage')
    summary=json.loads(data(spec['summary']));chunks=[json.loads(data(r)) for r in spec['chunks']]
    from src.p6_reference_source import validate_reference_source
    validate_reference_source(root,spec['reference_source'],tasks=spec['reference_tasks'],recipe=spec['recipe'])
    # Hash verification precedes deserialization. Inputs must be trusted project artifacts.
    payload=torch.load(io.BytesIO(data(spec['tasks'])),map_location='cpu',weights_only=False)
    reference=torch.load(io.BytesIO(data(spec['reference_tasks'])),map_location='cpu',weights_only=False)
    validate_tasks(payload,reference,recipe=recipe,recipe_sha256=spec['recipe']['sha256'])
    row=planning_outcomes(summary,chunks,item=job['input'],recipe=recipe,
        recipe_sha256=spec['recipe']['sha256'],job_sha256=spec['job']['sha256'],task_sha256=spec['tasks']['sha256'])
    row['evidence_status']='files_tasks_and_arithmetic_checked_pending_os_and_transfer'
    row['remaining']=['OS and budget child waits','complete governed return coverage','training checkpoint provenance']
    return {'row':row,'artifacts':[spec['summary'],spec['tasks'],*spec['chunks']],
            'references':{k:spec[k] for k in ('job','recipe','reference_tasks','reference_source','training_audit')},'launch_authorized':False}


def audit_planning_return(root,spec):
    """Join file/task checks to pinned two-layer OS and standard transfer proofs."""
    import pathlib
    from scripts.audit_p6_endpoint_inputs import read_bound
    from src.p6_receipt_audit import audit_evaluation_process,audit_return_coverage
    job,state,run_id=audit_evaluation_process(root,spec,stage='planning')
    result=audit_planning_files(root,spec)
    relative={a['path']:str(pathlib.PurePosixPath(a['path']).relative_to(spec['directory'])) for a in result['artifacts']}
    audit_return_coverage(read_bound(root,spec['return_manifest']),read_bound(root,spec['return_receipt']),
        manifest_sha256=spec['return_manifest']['sha256'],node_id=state['node_id'],run_name=job['run_name'],
        artifacts=result['artifacts'],source_relative_paths=relative)
    result['run_id']=run_id
    result['row']['evidence_status']='planning_return_audited_pending_training_ancestry'
    result['row']['remaining']=['training checkpoint payload and ancestry provenance']
    result['references'].update({k:spec[k] for k in ('expected_launch','launch','node_state','reservation','budget_result','return_manifest','return_receipt')})
    return result
