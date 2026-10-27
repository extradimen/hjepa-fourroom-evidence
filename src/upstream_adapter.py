"""FourRoom-only adapter; native source remains unchanged.

This file is preparation code until a governed upstream integration check passes.
"""
from copy import deepcopy
import torch
from .mechanisms import predicted_idm_loss, project_upper_gradient, effect_consistency
from contextlib import contextmanager


@contextmanager
def deterministic_auxiliary(model):
    """Suppress independent dropout noise in a paired effect comparison."""
    states = [(m, m.training) for m in model.modules()]
    try:
        model.eval()
        yield
    finally:
        for module, training in states:
            module.training = training


def m1_auxiliary(model, normalized_pixels, normalized_actions, perturbed_actions,
                 scale_squared, return_details=False):
    """Explicit L1 timeline 0..6, L2 anchors 0,2,4 -> 6; q remains five.

    Bypasses independent random crops for this extra branch. The caller perturbs
    only the final ten physical actions (L1 action blocks 4 and 5), clips in raw
    action space, then normalizes using the frozen training normalizer.
    This adapter is preparation code until the native integration check passes.
    """
    bottom, upper = model.get_level(1), model.get_level(2)
    if upper.temporal_stride != 2 or upper.temporal_window_size != 1:
        raise ValueError('M1 initial adapter requires stride 2, window 1')
    if normalized_actions.shape != perturbed_actions.shape:
        raise ValueError('Matched action sequences required')
    if normalized_actions.shape[1] < 6 or normalized_pixels.shape[1] < 5:
        raise ValueError('Need five observed start/history states and six action blocks')
    if not torch.equal(normalized_actions[:, :4], perturbed_actions[:, :4]):
        raise ValueError('Perturbation must preserve the common action history')
    with deterministic_auxiliary(model):
        with torch.no_grad():
            dense = bottom.encode({'pixels': normalized_pixels[:, :5]}, key='pixels')['embed_0']
            upper_history = upper.encoder(dense[:, [0, 2, 4]].flatten(0, 1)).reshape(
                dense.size(0), 3, -1)
        low, high = [], []
        for raw in (normalized_actions, perturbed_actions):
            with torch.no_grad():
                action = bottom.action_embed(bottom.action_encoder(raw[:, :6]))
                roll = dense[:, 2:5]
                for step in range(2):
                    pred = bottom.predict(roll[:, -3:], action[:, 2+step:5+step])[:, -1:]
                    roll = torch.cat((roll, pred), dim=1)
                low.append(upper.encoder(roll[:, -1]))
            macro = upper.action_encoder(action.reshape(action.size(0), 3, 2, -1))
            conditioning = upper.action_embed(macro)
            high.append(upper.predict(upper_history.detach(), conditioning)[:, -1])
        loss = effect_consistency(high[0], high[1], low[0], low[1], scale_squared)
        if return_details:
            return loss, {'low_effect_mse': (low[1]-low[0]).square().mean().detach(),
                          'high_effect_mse': (high[1]-high[0]).square().mean().detach()}
        return loss


def build_common_baseline(cfg, idm_weight=1.0, idm_hidden=256):
    from hjepa_utils import create_world_model
    from loss import InverseDynamicsLoss, InverseDynamicsModel
    if int(cfg.num_levels) != 2:
        raise ValueError('Initial protocol fixes two levels')
    construction = deepcopy(cfg)
    for level in (1, 2):
        construction[f'level{level}']['idm_coeff'] = 0.0
    model, losses = create_world_model(construction)
    runtime = deepcopy(cfg)
    for level in (1, 2):
        lc = runtime[f'level{level}']
        if lc.predictor.get('type') == 'causal_transformer':
            raise ValueError('This adapter is restricted to native FourRoom AR predictors')
        # encode_hierarchical exposes action AFTER the learned action embedding.
        # Its width is embed_dim (192), not raw action width (10) or macro width (4).
        width = int(lc.wm.embed_dim)
        model.get_level(level).idm_loss_fn = InverseDynamicsLoss(
            InverseDynamicsModel(width, idm_hidden, width))
        lc['idm_coeff'] = float(idm_weight)
    return model, losses, runtime


def m3_auxiliary(model, output, history_size):
    bottom = model.get_level(1)
    start = output['embed_1'][:, :history_size].detach()
    action = output['action_1'][:, :history_size].detach()
    predicted = bottom.predict(start, action)
    return predicted_idm_loss(bottom.idm_loss_fn.idm,
                              start.flatten(0, 1), predicted.flatten(0, 1),
                              action.flatten(0, 1))


def m2_auxiliary(model, head, frozen_reference, observation_start,
                 actual_end_u, actual_end_v, actions_u, actions_v, scale_squared):
    """One verified simulator branch pair; actions are two normalized q=5 blocks.

    Reference pretraining and full-state/ancestor split verification belong to
    the data protocol and must pass before this function is used for training.
    """
    bottom, upper = model.get_level(1), model.get_level(2)
    if actions_u.shape != actions_v.shape or actions_u.shape[1:] != (2, 10):
        raise ValueError('Initial FourRoom branch is ten two-dimensional physical actions')
    if any(p.requires_grad for p in frozen_reference.parameters()):
        raise ValueError('Reference encoder must be frozen before M2')
    with deterministic_auxiliary(frozen_reference), deterministic_auxiliary(model):
        with torch.no_grad():
            state = frozen_reference(observation_start)
            end_u = frozen_reference(actual_end_u)
            end_v = frozen_reference(actual_end_v)
            low_u = bottom.action_embed(bottom.action_encoder(actions_u))
            low_v = bottom.action_embed(bottom.action_encoder(actions_v))
        macro_u = upper.action_encoder(low_u.unsqueeze(1)).squeeze(1)
        macro_v = upper.action_encoder(low_v.unsqueeze(1)).squeeze(1)
        return head.loss(state, macro_u, macro_v, end_u, end_v, scale_squared)


def assign_m4_gradients(model, bottom_loss, upper_loss, shared_parameters):
    """Single-device, float32 only. Caller clips once, then steps optimizer.

    Real IDM and regularization must already be included in per-level losses.
    Never combine this with Lightning's automatic backward/AMP scaler.
    """
    if torch.distributed.is_initialized():
        raise RuntimeError('Distributed M4 is not validated')
    parameters = [p for p in model.parameters() if p.requires_grad]
    if any(p.dtype != torch.float32 for p in parameters):
        raise ValueError('Initial M4 implementation requires float32')
    shared_ids = {id(p) for p in shared_parameters}
    if not shared_ids or not shared_ids.issubset({id(p) for p in parameters}):
        raise ValueError('Explicit nonempty shared encoder group required')
    lower = torch.autograd.grad(bottom_loss, parameters, retain_graph=True, allow_unused=True)
    upper = torch.autograd.grad(upper_loss, parameters, allow_unused=True)
    indices = [i for i,p in enumerate(parameters) if id(p) in shared_ids]
    ga = [lower[i] if lower[i] is not None else torch.zeros_like(parameters[i]) for i in indices]
    gb = [upper[i] if upper[i] is not None else torch.zeros_like(parameters[i]) for i in indices]
    projected = dict(zip(indices, project_upper_gradient(ga, gb)))
    for i,p in enumerate(parameters):
        if i in projected:
            p.grad = projected[i].detach()
        elif lower[i] is None and upper[i] is None:
            p.grad = None
        else:
            p.grad = ((lower[i] if lower[i] is not None else torch.zeros_like(p)) +
                      (upper[i] if upper[i] is not None else torch.zeros_like(p))).detach()
