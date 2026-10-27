"""Exact task-payload comparison after callers verify returned file hashes.

No inference or simulation. A matching payload alone is not full provenance.
"""
import math


def equal_payload(left,right,path='tasks'):
    import numpy as np
    import torch
    if type(left) is not type(right):raise ValueError(f'Task type mismatch: {path}')
    if isinstance(left,torch.Tensor):
        if left.dtype!=right.dtype or left.shape!=right.shape:raise ValueError(f'Task tensor layout mismatch: {path}')
        same=left==right
        if left.is_floating_point() or left.is_complex():same=same | (torch.isnan(left)&torch.isnan(right))
        if not bool(same.all()):raise ValueError(f'Task tensor values differ: {path}')
    elif isinstance(left,np.ndarray):
        if left.dtype!=right.dtype or left.shape!=right.shape or not np.array_equal(left,right,equal_nan=True):
            raise ValueError(f'Task array differs: {path}')
    elif isinstance(left,dict):
        if left.keys()!=right.keys():raise ValueError(f'Task keys differ: {path}')
        for key in left:equal_payload(left[key],right[key],path+'.'+str(key))
    elif isinstance(left,(list,tuple)):
        if len(left)!=len(right):raise ValueError(f'Task sequence length differs: {path}')
        for i,(a,b) in enumerate(zip(left,right)):equal_payload(a,b,path+f'[{i}]')
    elif isinstance(left,float) and math.isnan(left) and math.isnan(right):pass
    elif left!=right:raise ValueError(f'Task value differs: {path}')


def validate_tasks(payload,reference,*,recipe,recipe_sha256):
    import torch
    expected={'split':'dev','episodes_idx':recipe['episodes'],
              'start_steps':[recipe['start_step']]*32,
              'goal_offset_steps':recipe['physical_goal_offset'],'eval_budget':recipe['eval_budget'],
              'reset_seed_start':recipe['reset_seed_start'],'recipe_sha256':recipe_sha256,'process':{}}
    if recipe['episodes']!=list(range(32)):raise ValueError('Expected fixed32dev tasks')
    for obj in (payload,reference):
        if any(obj.get(k)!=v for k,v in expected.items()):raise ValueError('Task payload recipe identity mismatch')
        if len(obj['data'])!=32 or 'seed' not in obj['columns']:raise ValueError('Task payload coverage mismatch')
        for episode,chunk in enumerate(obj['data']):
            seed=chunk['seed'];length=recipe['physical_goal_offset']+1
            if not isinstance(seed,torch.Tensor) or seed.dtype!=torch.int64 or tuple(seed.shape)!=(length,) or not bool((seed==recipe['reset_seed_start']+episode).all()):
                raise ValueError('Task tensor reset seeds mismatch')
            if len(chunk['pixels'])!=length:raise ValueError('Task tensor observation length mismatch')
    equal_payload(payload,reference)
    return {'status':'task_payload_semantically_equal','tasks':32,'provenance_complete':False}
