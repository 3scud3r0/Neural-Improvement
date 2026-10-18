"""Bounded topology search and neural plasticity for NeuralSociety.

Allows growth/pruning/widening, preserving overlapping weights. Does not
provision cloud resources, invent new layers automatically or preserve an
optimizer's moments after structural change (optimizer must be recreated).
"""
from __future__ import annotations
from dataclasses import dataclass,asdict
import time
import torch
from .model import NeuralSociety

@dataclass(frozen=True)
class TopologyLimits:
    max_agents:int=1000
    max_width:int=10000
    max_parameters:int=25_000_000
    max_training_memory_mb:float=2048.0
    min_agents:int=2

@dataclass
class TopologyEvent:
    action:str
    old:dict
    new:dict
    preserved_parameters:int
    timestamp:float


class DynamicTopology:
    AGENT_PARAMS=('agent_bias','agent_gain','agent_recur','agent_routing_bias')
    def __init__(self,model:NeuralSociety,limits:TopologyLimits|None=None):
        self.model=model
        self.limits=limits or TopologyLimits()
        self.history:list[TopologyEvent]=[]
        self._check(model)

    def _check(self,model):
        params=sum(p.numel() for p in model.parameters())
        estimated_adam_mb=params*16/(1024**2) # parameters, grads + two optimizer moments
        if (model.agents<self.limits.min_agents or model.agents>self.limits.max_agents or
            model.width>self.limits.max_width or params>self.limits.max_parameters or
            estimated_adam_mb>self.limits.max_training_memory_mb):
            raise ValueError('topology exceeds declared capacity limits')

    def _transform(self,agents:int,width:int,keep:tuple[int,...]|None=None,action='reshape'):
        current=self.model
        proposal=NeuralSociety(agents=agents,width=width,
                               message_dim=current.message_dim,
                               top_k=min(current.top_k,agents-1),
                               recurrence=current.recurrence,
                               communicate=current.communicate)
        self._check(proposal)
        proposal=proposal.to(device=next(current.parameters()).device,
                             dtype=next(current.parameters()).dtype)
        copied=0
        with torch.no_grad():
            src=current.state_dict();target=proposal.state_dict()
            for name,dest in target.items():
                origin=src.get(name)
                if origin is None:continue
                if keep is not None and name in self.AGENT_PARAMS:
                    origin=origin[list(keep)]
                overlap=tuple(slice(0,min(a,b)) for a,b in zip(dest.shape,origin.shape))
                if len(overlap)!=dest.ndim:continue
                dest[overlap].copy_(origin[overlap]);copied+=dest[overlap].numel()
            proposal.load_state_dict(target)
        event=TopologyEvent(action,current.report(),proposal.report(),copied,time.time())
        self.model=proposal
        self.history.append(event)
        return event

    def recruit(self,count:int=1):
        if not 1<=count<=1000:raise ValueError('invalid recruit count')
        return self._transform(self.model.agents+count,self.model.width,action=f'recruit_{count}')

    def widen(self,additional_neurons:int):
        if not 1<=additional_neurons<=10000:raise ValueError('invalid width increase')
        return self._transform(self.model.agents,self.model.width+additional_neurons,
                               action=f'widen_{additional_neurons}')

    def prune(self,indices:list[int]):
        original=self.model.agents
        removed=set(indices)
        if any(i<0 or i>=original for i in removed):raise ValueError('unknown agent index')
        keep=tuple(i for i in range(original) if i not in removed)
        if not indices or len(keep)<self.limits.min_agents:raise ValueError('invalid pruning')
        return self._transform(len(keep),self.model.width,keep=keep,action=f'prune_{sorted(removed)}')

    def suggest(self,validation_gain:float,gradient_norm:float)->dict:
        """Simple, inspectable NAS heuristic. Never applies changes itself."""
        if validation_gain < 0.002 and gradient_norm > 0.01:
            if self.model.agents + 1 <= self.limits.max_agents:
                return {'action':'recruit','count':1,'reason':'plateau with gradient signal'}
        if validation_gain < 0.002 and gradient_norm <= 0.01:
            if self.model.width + 16 <= self.limits.max_width:
                return {'action':'widen','neurons':16,'reason':'plateau with small gradients'}
        return {'action':'none','reason':'no reliable plateau detected'}

    @staticmethod
    def cloud_capacity_request(agents:int,width:int)->dict:
        """Planning only. No cloud API calls, credentials or deployment."""
        if agents<=0 or width<=0:raise ValueError('positive dimensions required')
        return {'proposal_only':True,'requires_human_approval':True,
                'agents':agents,'width':width,'activation_bytes_fp32':agents*width*4,
                'status':'not_provisioned'}

    def report(self):
        return {'current':self.model.report(),'limits':asdict(self.limits),
                'history':[asdict(e) for e in self.history]}
