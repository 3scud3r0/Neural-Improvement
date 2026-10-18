"""Supervised search-distillation into NEXUS-M policies on verified examples.

Verified here means consistent with supplied examples, not formally verified
against every input or the unknown hidden rule. Updates are bounded and can
be kept episodic without altering a stored checkpoint.
"""
from __future__ import annotations
from dataclasses import dataclass,asdict
import copy
import torch
from torch.nn import functional as F
from .model import NeuralSociety
from .features import encode
from .tasks import Task
from .grammar import TOKEN_TO_ID
from .hierarchical_abstraction_search import evaluate_macro_program

_MACRO_TOKEN={'ADD_1':'ADD','ADD_2':'ADD','ADD_3':'ADD',
              'SUB_1':'SUB','SUB_2':'SUB','MUL_2':'MUL','MUL_3':'MUL',
              'NEG':'NEG','ABS':'ABS','SQUARE':'SQUARE','HALF_EVEN':'HALF'}

@dataclass
class DistillationResult:
    initial_loss:float
    final_loss:float
    steps:int
    persisted:bool
    sample_consistent:bool
    def as_dict(self):return asdict(self)


def distill_macro_program(model:NeuralSociety,examples:list[tuple[int,int]],
                          program:list[str],steps:int=20,lr:float=.0008,
                          persist:bool=True):
    if not 1<=steps<=100:raise ValueError('max 100 temporary update steps')
    if not 0<lr<=.01:raise ValueError('learning rate out of range')
    if not program or any(op not in _MACRO_TOKEN for op in program):
        raise ValueError('program uses unknown macro')
    if evaluate_macro_program(program,[x for x,_ in examples])!=[y for _,y in examples]:
        raise ValueError('cannot distill a program inconsistent with evidence')
    work=model if persist else copy.deepcopy(model)
    task=Task('unlabeled',tuple(examples),(),None)
    parameter=next(work.parameters())
    optimizer=torch.optim.AdamW(work.parameters(),lr=lr)
    targets=[_MACRO_TOKEN[name] for name in program]
    original_mode=work.training
    work.train(); losses=[]
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        losses_each=[]
        for i,token in enumerate(targets):
            state=encode(task.public(),targets[:i],device=parameter.device,dtype=parameter.dtype)
            logits=work(state).logits
            label=torch.tensor([TOKEN_TO_ID[token]],device=parameter.device)
            losses_each.append(F.cross_entropy(logits.unsqueeze(0),label))
        loss=torch.stack(losses_each).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(work.parameters(),1.)
        optimizer.step()
        losses.append(float(loss.detach().item()))
    work.train(original_mode)
    return DistillationResult(losses[0],losses[-1],steps,persist,True)
