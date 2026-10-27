"""Protocol v2 auxiliary mechanisms. Upstream integration not yet validated."""
import torch
from torch import nn
from torch.nn import functional as F

def effect_consistency(high_u, high_v, low_u, low_v, scale_squared):
    """M1: caller must detach auxiliary inputs according to protocol."""
    if not (high_u.shape == high_v.shape == low_u.shape == low_v.shape):
        raise ValueError('Aligned upper-space predictions required')
    scale = torch.as_tensor(scale_squared, device=high_u.device, dtype=high_u.dtype)
    if not torch.all(scale > 0):
        raise ValueError('Frozen positive train calibration scale required')
    return (((high_v-high_u)-(low_v-low_u).detach()).square()/scale).mean()

class MacroEffectHead(nn.Module):
    """M2: actual paired outcomes required; no generated pseudo-targets."""
    def __init__(self, state_dim, macro_dim, hidden_dim):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(state_dim+2*macro_dim,hidden_dim),nn.ReLU(),nn.Linear(hidden_dim,state_dim))
    def forward(self, state, macro_u, macro_v):
        return self.net(torch.cat((state.detach(),macro_u,macro_v),dim=-1))
    def loss(self, state, macro_u, macro_v, actual_end_u, actual_end_v, scale_squared):
        target=(actual_end_v-actual_end_u).detach()
        scale=torch.as_tensor(scale_squared,device=target.device,dtype=target.dtype)
        if not torch.all(scale>0): raise ValueError('Positive frozen scale required')
        return ((self(state,macro_u,macro_v)-target).square()/scale).mean()

def predicted_idm_loss(idm, start, predicted_end, action_target):
    """M3: fixed auxiliary head parameters, differentiable predicted input."""
    from torch.func import functional_call
    state={k:v.detach() for k,v in idm.named_parameters()}
    state.update({k:v.detach().clone() for k,v in idm.named_buffers()})
    estimate=functional_call(idm,state,(start.detach(),predicted_end))
    if estimate.shape != action_target.shape: raise ValueError('Action target shape mismatch')
    return F.mse_loss(estimate,action_target.detach())

def project_upper_gradient(bottom, upper, eps=1e-12):
    """M4: unscaled, globally reduced gradients on one shared parameter group.
    None gradients must be replaced with correctly shaped zeros by caller.
    Optimizer integration and clipping are deliberately left to upstream adapter.
    """
    if len(bottom)!=len(upper) or not bottom: raise ValueError('Matching shared gradients required')
    if any(a.shape!=b.shape for a,b in zip(bottom,upper)): raise ValueError('Gradient shape mismatch')
    dot=sum((a.float()*b.float()).sum() for a,b in zip(bottom,upper))
    norm=sum(a.float().square().sum() for a in bottom)
    if not torch.isfinite(dot) or not torch.isfinite(norm): raise ValueError('Nonfinite gradient')
    coefficient=torch.where((dot<0)&(norm>eps),dot/(norm+eps),torch.zeros_like(dot))
    return [a+b-coefficient.to(a.dtype)*a for a,b in zip(bottom,upper)]
