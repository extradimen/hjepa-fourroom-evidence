"""Cross-check standard controller receipts; no process actions or lease mutation."""
import pathlib

PROJECT='26bc4d31-99a3-433c-9035-1a5a7fcd5847'

def audit_os_exit(launch,state,*,expected_cwd,expected_command,expected_config_sha256,launch_observation=None):
    identity=launch['identity'];child=state['child_process']
    # The node can snapshot /proc before exec has populated argv. In that case
    # require a separate, SHA-bound live observation of the *same* PID/start.
    command_observed=child['command']==expected_command
    if not command_observed and child['command']==[] and launch_observation is not None:
        from datetime import datetime
        if not isinstance(launch_observation,list) or len(launch_observation)!=1:
            raise ValueError('Expected one launch-time status observation')
        observation=launch_observation[0]
        lease=observation['lease'];early=observation['node_state']
        live=early['child_process_now']
        command_observed=all([
            observation.get('query_error') is None,
            lease['run_id']==launch['run_id']==early['run_id']==state['run_id'],
            lease['fingerprint']==launch['fingerprint']==early['fingerprint'],
            lease['identity']['command']==expected_command,
            lease['identity']['config_sha256']==expected_config_sha256,
            lease['receipt']['worker_pid']==state['worker_pid'],
            early['status']=='running',live['alive'] is True,
            live['identity_complete'] is True,
            live['pid']==child['pid'],live['ppid']==child['ppid'],
            live['start_ticks']==child['start_ticks'],
            live['cwd']==child['cwd']==expected_cwd,
            live['command']==expected_command,
            early['node_boot_at_launch']['boot_id']==state['node_boot_at_launch']['boot_id'],
            early['node_boot_now']['boot_id']==state['node_boot_now']['boot_id'],
            datetime.fromisoformat(early['node_boot_now']['observed_at']) < datetime.fromisoformat(state['completed_at'])])
    checks=[launch['project_id']==PROJECT,identity['project_id']==PROJECT,
            launch['receipt']['accepted'] is True,
            launch['run_id']==state['run_id']==launch['receipt']['run_id'],
            launch['node_id']==state['node_id'],launch['fingerprint']==state['fingerprint'],
            identity['command']==expected_command,command_observed,
            identity['config_sha256']==expected_config_sha256,
            child['cwd']==expected_cwd,child['identity_complete'] is True,
            child['pid']==state['child_pid']==state['child_process_now']['pid'],
            child['ppid']==state['worker_pid']==launch['receipt']['worker_pid'],
            isinstance(child['start_ticks'],int) and child['start_ticks']>0,
            state['node_boot_at_launch']['boot_id']==state['node_boot_now']['boot_id'],
            bool(state['node_boot_at_launch']['boot_id']),
            state['termination_evidence']=='os_child_wait_status',state['return_code']==0,
            state['status']=='completed',state['child_process_now']['alive'] is False,
            state['worker_process_now']['alive'] is False,
            state['worker_process_now']['pid']==state['worker_pid'],
            len(state['gpu_ids'])==1 and state['gpu_ids']==launch['gpu_ids'],
            bool(state.get('completed_at'))]
    if not all(checks):raise ValueError('OS receipt identity or successful child wait mismatch')
    return state['run_id']


def audit_return_coverage(manifest,receipt,*,manifest_sha256,node_id,run_name,artifacts,source_relative_paths=None):
    # Composite recovery receipts require a separate explicit expansion/audit.
    if receipt.get('receipt_kind') or receipt.get('status')!='complete' or receipt.get('manifest_sha256')!=manifest_sha256:
        raise ValueError('Expected completed standard materialization receipt')
    if manifest.get('project_id')!=PROJECT:raise ValueError('Wrong return project')
    declared={x['destination']:x for x in manifest['materials']}
    if len(declared)!=len(manifest['materials']):raise ValueError('Duplicate manifest destination')
    events={}
    for event in receipt['materials']:
        if event.get('event')=='materialized':events.setdefault(event['destination'],[]).append(event)
    for artifact in artifacts:
        dest=artifact['path'];digest=artifact['sha256'];entry=declared.get(dest)
        if not entry or entry['sha256']!=digest:raise ValueError('Artifact absent from return manifest')
        source=entry['source'];path=pathlib.PurePosixPath(source['path'])
        run_root=pathlib.PurePosixPath('runs')/run_name
        if source_relative_paths is None:
            matches_path=path.parent==run_root
        else:
            relative=pathlib.PurePosixPath(source_relative_paths[dest])
            matches_path=not relative.is_absolute() and '..' not in relative.parts and path==run_root/relative
        if source.get('kind')!='project' or source.get('node')!=node_id or not matches_path or '..' in path.parts:
            raise ValueError('Return source belongs to another run or artifact')
        matches=[e for e in events.get(dest,[]) if e.get('project_id')==PROJECT and e.get('node_id')=='local-mac' and e.get('sha256')==digest and e.get('source')==source and e.get('manifest_sha256')==manifest_sha256]
        if len(matches)!=1:raise ValueError('Missing or ambiguous governed materialization event')
    return {'status':'standard_receipt_coverage_verified','artifacts':len(artifacts)}


def audit_guard_result(reservation,result,*,reservation_id,expected_command,expected_wrapper=None):
    import math
    checks=[reservation['project_id']==PROJECT,reservation['reservation_id']==reservation_id,
            reservation['status']=='authorized_ready',reservation['counted_gpus']==1,
            result['reservation_id']==reservation_id,result['termination_evidence']=='owned_child_wait_status',
            type(result['return_code']) is int and result['return_code']==0,
            result['timed_out'] is False,result['forwarded_signal'] is None,
            result['command']==expected_command,len(result['devices'])==1,
            type(result['child_pid']) is int and result['child_pid']>0,
            str(result['child_start_ticks']).isdigit() and int(result['child_start_ticks'])>0,
            result['reserved_gpu_seconds']==reservation['reserved_gpu_seconds'],
            math.isfinite(result['elapsed_seconds']) and 0<=result['elapsed_seconds']<=reservation['reserved_gpu_seconds']]
    if expected_wrapper is not None:
        checks.extend([result.get('wrapper_pid')==expected_wrapper['pid'],
                       str(result.get('wrapper_start_ticks'))==str(expected_wrapper['start_ticks'])])
    if not all(checks):raise ValueError('Budget guard child wait or reservation mismatch')
    return result['child_pid']


def audit_evaluation_process(root,spec,*,stage):
    from scripts.audit_p6_endpoint_inputs import read_bound
    workers={'planning':'scripts/evaluate_p6_planning_item.py','position':'scripts/evaluate_p6_position.py'}
    if stage not in workers:raise ValueError('Unknown evaluation stage')
    job=read_bound(root,spec['job']);expected=read_bound(root,spec['expected_launch'])
    if spec.get('project_id')!=PROJECT or job.get('status')!='frozen' or job.get('stage')!=stage or job.get('test_access') is not False:
        raise ValueError('Wrong evaluation project/job')
    if expected['job_sha256']!=spec['job']['sha256'] or expected['config_sha256']!=spec['job']['sha256']:
        raise ValueError('Evaluation dispatch config/job mismatch')
    command=['python','scripts/run_with_budget.py','--reservation',spec['reservation']['path'],
             '--','python',workers[stage],'--job',spec['job']['path']]
    if expected['command']!=command:raise ValueError('Unexpected evaluation guard/worker command')
    launch=read_bound(root,spec['launch']);state=read_bound(root,spec['node_state'])
    run_id=audit_os_exit(launch,state,expected_cwd=expected['cwd'],expected_command=command,
                         expected_config_sha256=spec['job']['sha256'])
    audit_guard_result(read_bound(root,spec['reservation']),read_bound(root,spec['budget_result']),
        reservation_id=job['reservation_id'],expected_command=command[5:],expected_wrapper=state['child_process'])
    return job,state,run_id
