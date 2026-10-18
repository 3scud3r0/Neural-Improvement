"""Optional learned predictive model of program-state transitions.

The *real* DSL simulator is exact; this auxiliary neural predictor learns how
state embeddings change when a token is appended. This is a limited world
model of a formal language, not a general model of the physical world.
"""
from __future__ import annotations
from pathlib import Path
import json,random,time
import torch
from torch import nn
from torch.nn import functional as F
from .features import encode,FEATURE_DIM
from .grammar import TOKENS,TOKEN_TO_ID
from .tasks import sample_task, Task

class HypothesisWorldModel(nn.Module):
    def __init__(self,hidden=96):
        super().__init__()
        self.hidden=hidden
        self.transition=nn.Sequential(nn.Linear(FEATURE_DIM+len(TOKENS),hidden),
                      nn.GELU(),nn.Linear(hidden,hidden),nn.GELU(),
                      nn.Linear(hidden,FEATURE_DIM))

    def forward(self,features,action_ids):
        # Batch-compatible, including scalar feature/action pairs.
        if features.ndim==1:features=features.unsqueeze(0)
        ids=torch.as_tensor(action_ids,device=features.device,dtype=torch.long).reshape(-1)
        action_hot=F.one_hot(ids,len(TOKENS)).to(features.dtype)
        if len(action_hot)==1 and features.shape[0]>1:
            action_hot=action_hot.expand(features.shape[0],-1)
        return self.transition(torch.cat((features,action_hot),dim=-1))

    @torch.no_grad()
    def surprise(self,task:Task,prefix,action):
        device=next(self.parameters()).device
        before=encode(task.public(),prefix,device=device)
        after=encode(task.public(),tuple(prefix)+(action,),device=device)
        prediction=self(before,TOKEN_TO_ID[action]).squeeze(0)
        return float(F.mse_loss(prediction,after).item())


def train_world_model(output_dir,steps=1000,seed=2026,hidden=96,lr=0.001):
    """Learn predictive dynamics from supervised training-family transitions."""
    directory=Path(output_dir);directory.mkdir(parents=True,exist_ok=True)
    rng=random.Random(seed)
    torch.manual_seed(seed)
    net=HypothesisWorldModel(hidden)
    optimizer=torch.optim.AdamW(net.parameters(),lr=lr)
    history=[];started=time.monotonic()
    for step in range(1,steps+1):
        task=sample_task(rng,'train')
        prefix=task.oracle[:rng.randrange(len(task.oracle))]
        action=task.oracle[len(prefix)]
        before=encode(task.public(),prefix)
        after=encode(task.public(),prefix+(action,))
        optimizer.zero_grad(set_to_none=True)
        prediction=net(before,TOKEN_TO_ID[action]).squeeze(0)
        loss=F.mse_loss(prediction,after)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),1.0)
        optimizer.step()
        if step==1 or step%max(1,steps//20)==0 or step==steps:
            history.append({'step':step,'transition_mse':float(loss.detach())})
    checkpoint=directory/'world_model.pt'
    torch.save({'model':net.state_dict(),'hidden':hidden,'steps':steps,'seed':seed},checkpoint)
    result={'steps':steps,'training_duration_sec':time.monotonic()-started,
            'last_transition_mse':history[-1]['transition_mse'],
            'checkpoint':str(checkpoint),'scope':'synthetic DSL feature transitions'}
    (directory/'world_training.json').write_text(json.dumps({'result':result,'history':history},indent=2))
    return result


def load_world_model(path):
    bundle=torch.load(path,map_location='cpu',weights_only=True)
    model=HypothesisWorldModel(bundle['hidden'])
    model.load_state_dict(bundle['model'])
    model.eval()
    return model
