"""Validate timing provenance before loading a model into an evaluator.

This adapter neither evaluates nor grants permission to reuse P5 evidence.
Tensor keys are checked without importing torch, so its rejection paths can be
covered by ordinary unit tests before any governed GPU run.
"""
from dataclasses import asdict
from src.p6_schedule import TimingCell


def validate_checkpoint(payload, item, *, reuse_receipt=None):
    cell = TimingCell(**item['cell'])
    if not {'model', 'normalizer'}.issubset(payload):
        raise ValueError('Missing world model or train normalizer')
    if item['source_kind'] == 'p6_tail':
        job = payload.get('job', {})
        if payload.get('step') != 4000 or job.get('stage') != 'tail':
            raise ValueError('Only the 4000-step endpoint may be evaluated')
        if job.get('cell') != asdict(cell) or job.get('test_access') is not False:
            raise ValueError('Timing cell or split provenance mismatch')
        if not item.get('job_sha256') or payload.get('job_sha256') != item['job_sha256']:
            raise ValueError('Frozen training job mismatch')
    elif item['source_kind'] == 'p5_reuse':
        source = cell.p5_reuse_source()
        if source is None or item.get('run_name') != source:
            raise ValueError('P5 ancestry cannot represent this timing cell')
        arm = cell.mechanism or ('B-same-branch-data' if cell.data_start else 'B')
        expected = {'step': 1000, 'seed': cell.seed, 'idm': cell.at(4000)['idm'], 'arm': arm}
        if any(payload.get(k) != v for k, v in expected.items()):
            raise ValueError('P5 payload identity mismatch')
        # The caller must separately SHA-verify this receipt as a material input.
        if not reuse_receipt or reuse_receipt.get('status') != 'execution_equivalence_verified':
            raise ValueError('Conditional P5 reuse has not been verified')
        matching = [x for x in reuse_receipt.get('cells', []) if x.get('cell') == asdict(cell)]
        if len(matching) != 1 or any(matching[0].get(k) != item.get(k) for k in ('checkpoint_sha256', 'bn_sha256')):
            raise ValueError('Reuse proof must bind this exact checkpoint and BN')
    else:
        raise ValueError('Unknown checkpoint source')
    return {'cell': asdict(cell), 'idm': cell.at(4000)['idm'], 'total_training_steps': 4000}


def validate_bn_keys(model_keys, buffers):
    expected = {k for k in model_keys if k.endswith(('running_mean', 'running_var', 'num_batches_tracked'))}
    if set(buffers) != expected:
        raise ValueError('BN artifact must contain exactly the model BN buffers')


def validate_training_binding(audit,item):
    """Bind evaluator input to the independently audited returned training run.

    This does not upgrade the training audit's pending checkpoint-payload status.
    """
    from src.p6_receipt_audit import PROJECT
    if audit.get('project_id')!=PROJECT or audit.get('stage')!='tail' or audit.get('status')!='files_os_and_standard_transfer_audited_payload_pending':
        raise ValueError('Expected returned tail training audit')
    matches=[x for x in audit.get('checkpoints',[]) if x.get('cell')==item.get('cell')]
    if len(matches)!=1:raise ValueError('Training cell audit missing or ambiguous')
    bound=matches[0]
    for key in ('source_kind','checkpoint_sha256','bn_sha256','job_sha256'):
        if not item.get(key) or item[key]!=bound.get(key):raise ValueError('Evaluation input differs from audited training artifact')
    if item['source_kind']!='p6_tail':raise ValueError('Training binding applies only to new tails')
    runs=[x for x in audit.get('audits',[]) if x.get('evidence_references',{}).get('job',{}).get('sha256')==item['job_sha256']]
    if len(runs)!=1:raise ValueError('Training OS/transfer run audit missing or ambiguous')
    artifacts={x['path']:x['sha256'] for x in runs[0]['artifacts']}
    if artifacts.get(bound['path'])!=item['checkpoint_sha256'] or artifacts.get(bound['bn_path'])!=item['bn_sha256']:
        raise ValueError('Training artifact absent from run audit')
    return runs[0]['run_id']


def validate_state_values(state):
    """Reject corrupt tensors before loading weights or recalibrated BN.

    Shape/key compatibility remains enforced by strict model loading. This is
    not an ancestry or training-equivalence proof.
    """
    import torch
    if not isinstance(state, dict) or not state:
        raise ValueError('Expected a nonempty tensor state dictionary')
    for name, value in state.items():
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise ValueError('State entries must be named tensors')
        if not torch.isfinite(value).all().item():
            raise ValueError(f'Nonfinite state tensor: {name}')
        if name.endswith(('running_var', 'num_batches_tracked')) and (value < 0).any().item():
            raise ValueError(f'Invalid negative BN state: {name}')
