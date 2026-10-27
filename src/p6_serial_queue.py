"""Pure serial continuation decisions; never launches or declares OS exit.

The controller adapter must populate these observations from SHA-bound audits.
A next candidate is not launch permission: live governance checks are mandatory.
"""


def next_serial_action(jobs, attempts):
    ids=[j['job_id'] for j in jobs]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate queue job')
    known=set()
    for job in jobs:
        if any(parent not in known for parent in job.get('dependencies', [])):
            raise ValueError('Queue dependencies must precede their child')
        known.add(job['job_id'])
    seen=set();accepted=set()
    for index, attempt in enumerate(attempts):
        job_id=attempt['job_id']
        if job_id in seen or job_id not in known:raise ValueError('Duplicate or unknown attempt; no automatic retry')
        if index>=len(ids) or ids[index]!=job_id:raise ValueError('Attempt order differs from frozen queue')
        seen.add(job_id)
        phase=attempt.get('phase')
        if phase=='accepted':
            # Acceptance requires the adapter's complete audit, not merely exit0.
            if not attempt.get('audit_reference') or attempt.get('audit_status')!='fully_audited':
                raise ValueError('Queue acceptance requires full audit reference')
            if any(p not in accepted for p in jobs[index].get('dependencies', [])):
                raise ValueError('Accepted job has unaccepted dependency')
            accepted.add(job_id);continue
        if index!=len(attempts)-1:raise ValueError('Later job attempted before prior acceptance')
        base={'job_id':job_id,'launch_authorized':False,'accepted_jobs':len(accepted)}
        if phase in ('failed','rejected'):
            return {**base,'action':'stop','reason':'Failure retained; independent repair and budget review required'}
        if phase in ('intent_recorded','submitted','running','unknown'):
            return {**base,'action':'audit_process','reason':'Process identity, global lease and both owned waits required; never resubmit'}
        if phase=='process_exited':
            waits=attempt.get('waits',{})
            if (waits.get('global_status')!='os_child_wait_status' or waits.get('guard_status')!='owned_child_wait_status'
                    or not attempt.get('process_audit_reference')):
                return {**base,'action':'audit_process','reason':'Incomplete exit evidence'}
            codes=[waits.get('global_return_code'),waits.get('guard_return_code')]
            if any(type(code) is not int for code in codes):
                return {**base,'action':'audit_process','reason':'Exit status unknown'}
            if codes!=[0,0]:return {**base,'action':'stop','reason':'Nonzero owned exit; retain full reservation'}
            return {**base,'action':'collect_return','reason':'Collect through governed materialization before acceptance'}
        if phase=='returned':
            if not attempt.get('return_reference'):
                raise ValueError('Returned phase lacks governed return reference')
            return {**base,'action':'audit_result','reason':'File, payload, ancestry and transfer acceptance still required'}
        raise ValueError('Unknown queue phase')
    if len(accepted)==len(jobs):
        return {'action':'queue_complete','accepted_jobs':len(accepted),'launch_authorized':False}
    job=jobs[len(accepted)]
    return {'action':'prepare_candidate','job_id':job['job_id'],'accepted_jobs':len(accepted),
            'launch_authorized':False,'reason':'Revalidate budget, source/deployment, live resources, material and environment before controller launch'}
