import torch
from torch.nn import functional as F
def m3_replacement(model, output, head):
 start=output['embed_1'][:,:3].detach();action=output['action_1'][:,:3].detach()
 predicted=model.get_level(1).predict(start,action)
 return F.mse_loss(head(start.flatten(0,1),predicted.flatten(0,1)),action.flatten(0,1))
