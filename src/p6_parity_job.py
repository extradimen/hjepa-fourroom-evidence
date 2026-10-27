"""Fixed local-update diagnostic scope; does not authorize model reuse."""
from src.p6_schedule import all_cells


def validate_parity_job(job, protocol, inputs, materials):
    expected=[c.p5_reuse_source() for c in all_cells() if c.p5_reuse_source()]
    diagnostic=job.get('stage')=='parity_diagnostic'
    backend=protocol.get('verification_backend')
    if backend is not None:
        if (backend!='deterministic_diagnostic_only' or diagnostic
            or protocol.get('comparison_paths')!=['P5','P6','P5-repeat']
            or protocol.get('production_backend_changes') is not False
            or protocol.get('acceptance_kind')!='exact_with_same_path_control'
            or protocol.get('backend_settings')!={'CUBLAS_WORKSPACE_CONFIG':':4096:8',
                'deterministic_algorithms':True,'warn_only':False,
                'cudnn_deterministic':True,'cudnn_benchmark':False}):
            raise ValueError('Unsupported verification backend or acceptance scope')
    if diagnostic:expected=['p5-seed11-idm0-B-v1']
    if (job.get('status')!='frozen' or job.get('stage')!='parity'
        or job.get('steps')!=2 or job.get('test_access') is not False
        or job.get('required_total_cap_seconds') not in (209700,210300,210900,211620)):
        if not (diagnostic and job.get('status')=='frozen' and job.get('steps')==1
                and job.get('test_access') is False and job.get('required_total_cap_seconds')==211020):
            raise ValueError('Invalid parity job scope')
    if diagnostic and protocol.get('comparison_paths')!=['P5','P6','P5-repeat']:
        raise ValueError('Diagnostic requires fixed same-path repeat control')
    if (protocol.get('status')!='frozen' or protocol.get('steps_per_candidate')!=(1 if diagnostic else 2)
        or protocol.get('candidates')!=expected or protocol.get('max_gpu_seconds')!=(120 if diagnostic else 600)
        or protocol.get('model_reuse_authorized') is not False
        or protocol.get('test_access') is not False
        or protocol.get('automatic_retry') is not False):
        raise ValueError('Parity protocol differs from fixed diagnostic scope')
    required={job['protocol_path']:job['protocol_sha256'],
              'scripts/train_p5_arm.py':job['p5_worker_sha256'],
              job['inputs_path']:job['inputs_sha256'],
              job['coefficient_source']:job['coefficient_sha256']}
    rows=inputs['checkpoints']
    if [x['seed'] for x in rows]!=[11,23] or inputs.get('parent_idm_exposure') is not False:
        raise ValueError('Parity requires both neutral parent seeds')
    for row in rows:
        required[row['path']]=row['sha256'];required[row['bn_path']]=row['bn_sha256']
    if any(materials.get(p)!=h for p,h in required.items()):
        raise ValueError('Parity input absent from material binding')
