"""Full in-memory branch snapshots for FourRoom feasibility checks.

No partial position-only restore. Replay verification is mandatory before data use.
"""
import copy
import random
import numpy as np
import torch


def capture(env):
    return {'environment': copy.deepcopy(env),
            'python_rng': random.getstate(),
            'numpy_rng': copy.deepcopy(np.random.get_state()),
            'torch_rng': torch.random.get_rng_state().clone()}


def restore(snapshot):
    env = copy.deepcopy(snapshot['environment'])
    random.setstate(snapshot['python_rng'])
    np.random.set_state(copy.deepcopy(snapshot['numpy_rng']))
    torch.random.set_rng_state(snapshot['torch_rng'].clone())
    return env


def replay(snapshot, actions):
    env = restore(snapshot)
    outputs = []
    for action in actions:
        observation, reward, terminated, truncated, info = env.step(action.copy())
        outputs.append({'observation':copy.deepcopy(observation),
                        'pixels':copy.deepcopy(env.render()),
                        'reward':reward, 'terminated':terminated,
                        'truncated':truncated, 'info':copy.deepcopy(info)})
        if terminated or truncated:
            raise RuntimeError('Branch ended early; no fixed-horizon pair can be emitted')
    return outputs


def assert_identical(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_identical(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x,y in zip(a,b):
            assert_identical(x,y)
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a,b,equal_nan=True)
    elif torch.is_tensor(a):
        assert torch.equal(a,b)
    else:
        assert a == b


def verified_pair(snapshot, actions_u, actions_v):
    if len(actions_u) != len(actions_v) or not len(actions_u):
        raise ValueError('Matching nonempty branch horizons required')
    branch_u = replay(snapshot, actions_u)
    assert_identical(branch_u, replay(snapshot, actions_u))
    branch_v = replay(snapshot, actions_v)
    assert_identical(branch_v, replay(snapshot, actions_v))
    return branch_u, branch_v
