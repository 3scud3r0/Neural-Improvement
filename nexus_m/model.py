"""100-agent scalable recurrent neural society with learnable sparse messaging.

Width = actual activation count PER agent; parameters use factorized/shared
projections plus agent-specific adapters (not N dense HxH matrices).
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import torch
from torch import nn
import torch.nn.functional as F
from .features import FEATURE_DIM
from .grammar import TOKENS

@dataclass
class CollectiveOutput:
    logits: torch.Tensor
    value: torch.Tensor
    states: torch.Tensor
    messages: torch.Tensor
    routing: torch.Tensor
    influence: torch.Tensor


class NeuralSociety(nn.Module):
    def __init__(self, agents=100, width=100, message_dim=16, top_k=4,
                 recurrence=2, communicate=True):
        super().__init__()
        if agents<2 or width<4 or message_dim<2 or not 1<=top_k<agents:
            raise ValueError('Bad architecture dimensions')
        self.agents=agents
        self.width=width
        self.message_dim=message_dim
        self.top_k=top_k
        self.recurrence=recurrence
        self.communicate=communicate
        # Agent-specific weights and biases; O(agents*width), not O(agents*width²).
        self.agent_bias=nn.Parameter(torch.randn(agents,width)*0.025)
        self.agent_gain=nn.Parameter(torch.ones(agents,width)+torch.randn(agents,width)*0.01)
        self.agent_recur=nn.Parameter(torch.full((agents,width),0.20))
        self.agent_routing_bias=nn.Parameter(torch.randn(agents,message_dim)*0.05)
        self.encoder=nn.Linear(FEATURE_DIM,width)
        self.message_encoder=nn.Linear(width,message_dim)
        self.message_decoder=nn.Linear(message_dim,width,bias=False)
        self.query_proj=nn.Linear(message_dim,message_dim,bias=False)
        self.key_proj=nn.Linear(message_dim,message_dim,bias=False)
        self.action_head=nn.Linear(width,len(TOKENS))
        self.value_head=nn.Linear(width,1)
        self.influence_head=nn.Linear(width,1)
        self.halt_gate=nn.Parameter(torch.tensor(-0.5))

    def forward(self, features: torch.Tensor, prior_states=None,
                force_no_messages=False, diagnostics=False) -> CollectiveOutput:
        if features.ndim!=1 or features.shape[0]!=FEATURE_DIM:
            raise ValueError(f'Expected ({FEATURE_DIM},) input')
        base=torch.tanh(self.encoder(features))[None,:]
        states=(torch.zeros_like(self.agent_bias) if prior_states is None else prior_states)
        last_routing=None
        last_msg=None
        for _ in range(max(1,self.recurrence)):
            normalized=F.layer_norm(states,(self.width,))
            # Independent per-agent hidden state with trainable adapters.
            candidate=torch.tanh(base*self.agent_gain+self.agent_bias +
                                 self.agent_recur*normalized)
            msg=torch.tanh(self.message_encoder(candidate)+self.agent_routing_bias)
            if self.communicate and not force_no_messages:
                q=F.normalize(self.query_proj(msg),dim=-1,eps=1e-6)
                k=F.normalize(self.key_proj(msg),dim=-1,eps=1e-6)
                scores=(q @ k.T)/math.sqrt(self.message_dim)
                scores=scores.masked_fill(torch.eye(self.agents,dtype=torch.bool,device=scores.device),-1e6)
                chosen=scores.topk(self.top_k,dim=-1)
                sparse_w=torch.softmax(chosen.values,dim=-1)
                incoming=(sparse_w.unsqueeze(-1)*msg[chosen.indices]).sum(dim=1)
                # Non-selected links are exactly zero; top-k routing is data dependent.
                last_routing=torch.zeros_like(scores).scatter(1,chosen.indices,sparse_w)
                states=torch.tanh(candidate+0.65*self.message_decoder(incoming))
            else:
                last_routing=torch.zeros((self.agents,self.agents),device=msg.device,dtype=msg.dtype)
                states=candidate
            last_msg=msg
        importance=torch.softmax(self.influence_head(states).squeeze(-1),dim=0)
        logits=(importance.unsqueeze(-1)*self.action_head(states)).sum(dim=0)
        value=torch.sigmoid((importance.unsqueeze(-1)*self.value_head(states)).sum())
        return CollectiveOutput(logits,value,states,last_msg,last_routing,importance)

    @property
    def neuron_count(self): return self.agents*self.width

    def report(self):
        return {'agents':self.agents,'neurons_per_agent':self.width,
                'total_neurons':self.neuron_count,
                'trainable_parameters':sum(p.numel() for p in self.parameters()),
                'top_k':self.top_k,'message_dim':self.message_dim,
                'recurrent_rounds':self.recurrence,'communicate':self.communicate}
