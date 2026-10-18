"""Reproducible interventions; interventions are not a proof of intelligence."""
from __future__ import annotations
import torch
from .model import NeuralSociety
from .features import encode
from .tasks import Task

@torch.no_grad()
def communication_intervention(model:NeuralSociety,task:Task)->dict:
    p=next(model.parameters())
    x=encode(task.public(),device=p.device,dtype=p.dtype)
    model.eval()
    baseline=model(x)
    without=model(x,force_no_messages=True)
    pa=torch.softmax(baseline.logits.float(),dim=-1)
    pb=torch.softmax(without.logits.float(),dim=-1)
    return {'policy_total_variation':float((pa-pb).abs().sum()/2),
            'value_delta':float((baseline.value-without.value).item()),
            'communication_links':int((baseline.routing>0).sum()),
            'max_agent_influence':float(baseline.influence.max()),
            'influence_entropy':float(-(baseline.influence*baseline.influence.clamp_min(1e-9).log()).sum()),
            'interpretation':'Intervention; alone does not prove conceptual specialization'}

@torch.no_grad()
def agent_lesion(model:NeuralSociety,task:Task,which=0)->dict:
    """Approximate targeted lesion by temporarily zeroing agent adapters."""
    if not 0<=which<model.agents:raise ValueError('agent index outside network')
    p=next(model.parameters());f=encode(task.public(),device=p.device,dtype=p.dtype)
    original=model(f)
    names=('agent_gain','agent_bias','agent_routing_bias')
    old={k:getattr(model,k)[which].detach().clone() for k in names}
    try:
        model.agent_gain[which].zero_();model.agent_bias[which].zero_()
        model.agent_routing_bias[which].zero_()
        ablated=model(f)
    finally:
        for name,value in old.items():getattr(model,name)[which].copy_(value)
    return {'agent':which,'policy_change_l1':float(torch.softmax(original.logits.float(),-1).sub(
              torch.softmax(ablated.logits.float(),-1)).abs().sum()),
            'value_change':float(original.value-ablated.value)}
