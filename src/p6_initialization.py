"""Production P6 model setup shared by training and governed parity checks."""


def initialize_world(runtime, seed):
    import random
    import numpy as np
    import torch
    from src.upstream_adapter import build_common_baseline
    if seed not in (11,23):raise ValueError('Unsupported P6 seed')
    for level in (1,2):runtime.cfg[f'level{level}'].idm_coeff=0.
    del runtime.model,runtime.optim
    runtime.cfg.seed=seed
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    runtime.model,losses,runtime.cfg=build_common_baseline(runtime.cfg,0.,runtime.settings['common_idm_hidden'])
    runtime.model.cuda()
    for key,value in losses.items():setattr(runtime,key,value.cuda())
    runtime.optim=torch.optim.AdamW([
        {'params':runtime.model.get_level(level).parameters(),
         'lr':float(runtime.cfg[f'level{level}'].lr),'name':f'level{level}'} for level in (1,2)
    ],weight_decay=float(runtime.cfg.optimizer.weight_decay))


def load_neutral_parent(runtime, payload, *, seed):
    from src.p6_task_identity import equal_payload
    from src.p6_checkpoint import validate_parent_state
    if (payload.get('step')!=3000 or payload.get('seed')!=seed
            or payload.get('parent_checkpoint') is not None
            or payload.get('settings',{}).get('common_idm_coefficient')!=0):
        raise ValueError('Expected seed-matched 3000-step hierarchy-only parent')
    equal_payload(runtime.normalizer,payload['normalizer'],path='normalizer')
    validate_parent_state(payload)
    runtime.model.load_state_dict(payload['model'],strict=True)
    runtime.optim.load_state_dict(payload['optimizer'])
    runtime.normalizer=payload['normalizer']
