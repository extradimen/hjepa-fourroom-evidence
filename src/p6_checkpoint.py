"""Checkpoint identity and optimizer preservation for timing continuations."""
import copy
import torch


def parameter_groups(model, head, optimizer):
    names={id(p):'model.'+n for n,p in model.named_parameters()}
    if head is not None:names.update({id(p):'head.'+n for n,p in head.named_parameters()})
    groups=[];seen=set()
    for group in optimizer.param_groups:
        order=[]
        for p in group['params']:
            if id(p) not in names or id(p) in seen:raise ValueError('Unknown or duplicate optimizer parameter')
            seen.add(id(p));order.append(names[id(p)])
        groups.append({'name':group.get('name'),'parameters':order})
    if seen != set(names):raise ValueError('Model/head parameters missing from optimizer')
    return groups


def snapshot(model, head, optimizer, *, phase_step, prefix_key):
    if phase_step != 3000:raise ValueError('Only fixed 3000-step prefix can branch')
    return copy.deepcopy({'schema_version':1,'step':phase_step,'prefix_key':tuple(prefix_key),
        'model':model.state_dict(),'head':None if head is None else head.state_dict(),
        'optimizer':optimizer.state_dict(),'optimizer_parameter_names':parameter_groups(model,head,optimizer)})


def restore(payload, model, head, optimizer, *, expected_prefix):
    if payload['step']!=3000 or tuple(payload['prefix_key'])!=tuple(expected_prefix):
        raise ValueError('Wrong ancestry or branch boundary')
    if (head is None)!=(payload['head'] is None):raise ValueError('Early head must be reconstructed before optimizer restore')
    if parameter_groups(model,head,optimizer)!=payload['optimizer_parameter_names']:
        raise ValueError('Optimizer parameter order differs from checkpoint')
    validate_parent_state(payload)
    model.load_state_dict(payload['model'])
    if head is not None:head.load_state_dict(payload['head'])
    optimizer.load_state_dict(payload['optimizer'])


def initialize_head(factory, *, seed, cuda_devices=()):
    # New heads must not consume the world-model/dropout stream.
    with torch.random.fork_rng(devices=list(cuda_devices)):
        torch.manual_seed(seed)
        if cuda_devices:torch.cuda.manual_seed_all(seed)
        return factory()


def recalibrate_bn(runtime, *, batches=128):
    # Same P5 training-only loader/seeds; optimizer and early head are untouched.
    if type(batches) is not int or batches<=0:raise ValueError('Positive BN batch count required')
    processed=0
    model=runtime.model;flags=[(p,p.requires_grad) for p in model.parameters()]
    model.eval();model.requires_grad_(False)
    try:
        for m in model.modules():
            if isinstance(m,torch.nn.modules.batchnorm._BatchNorm):
                m.reset_running_stats();m.momentum=None;m.train()
        with torch.no_grad():
            for k,batch in enumerate(runtime.loader('train',seed=50131)):
                if k>=batches:break
                torch.manual_seed(11013+k)
                if next(model.parameters()).is_cuda:torch.cuda.manual_seed_all(11013+k)
                runtime.forward(runtime.batch(batch),'bn_recalibration')
                processed+=1
        if processed!=batches:raise ValueError('Incomplete train-only BN recalibration')
    finally:
        model.eval()
        for p,flag in flags:p.requires_grad_(flag)
    return processed


def validate_parent_state(payload):
    """Check numerical integrity before mutating a continuation's live state."""
    import math
    from src.p6_evaluation_identity import validate_state_values
    validate_state_values(payload['model'])
    if payload.get('head') is not None:validate_state_values(payload['head'])
    def finite(value):
        if isinstance(value, torch.Tensor):
            if not torch.isfinite(value).all().item():raise ValueError('Nonfinite optimizer tensor')
        elif isinstance(value, float):
            if not math.isfinite(value):raise ValueError('Nonfinite optimizer scalar')
        elif isinstance(value, dict):
            for child in value.values():finite(child)
        elif isinstance(value, (list, tuple)):
            for child in value:finite(child)
    optimizer=payload['optimizer']
    if not isinstance(optimizer, dict) or not {'state','param_groups'}.issubset(optimizer):
        raise ValueError('Missing optimizer state structure')
    finite(optimizer)
