"""JEPA-inspired predictive latent dynamics for macro-action transitions.

Self-supervised one-step prediction with an EMA target encoder. This is a
small, restricted model trained on synthetic integer transformations, not a
world model for physics, chemistry, psychology or the entire universe.
"""
from __future__ import annotations
from dataclasses import dataclass,asdict
import math,random
import torch
from torch import nn
from torch.nn import functional as F
from .hierarchical_abstraction_search import DEFAULT_MACROS


def state_features(state:tuple[int,...]|list[int],device='cpu'):
    if not state:raise ValueError('nonempty state')
    v=torch.tensor(state,dtype=torch.float32,device=device)
    scaled=torch.tanh(v/30)
    return torch.stack([scaled.mean(),scaled.std(unbiased=False),scaled.min(),
                        scaled.max(),(scaled**2).mean().sqrt(),
                        torch.tanh(v.mean()/100.)])

class LatentDynamics(nn.Module):
    def __init__(self,latent_dim=16,hidden=48,macros=DEFAULT_MACROS):
        super().__init__()
        self.names=[m.name for m in macros]
        self.macro_lookup={m.name:m for m in macros}
        self.encoder=nn.Sequential(nn.Linear(6,hidden),nn.Tanh(),
                                   nn.Linear(hidden,latent_dim))
        self.target_encoder=nn.Sequential(nn.Linear(6,hidden),nn.Tanh(),
                                          nn.Linear(hidden,latent_dim))
        self.target_encoder.load_state_dict(self.encoder.state_dict())
        for param in self.target_encoder.parameters():param.requires_grad_(False)
        self.action_embeddings=nn.Embedding(len(macros),latent_dim)
        self.predictor=nn.Sequential(nn.Linear(2*latent_dim,hidden),nn.ReLU(),
                                     nn.Linear(hidden,latent_dim))

    @torch.no_grad()
    def update_target(self,momentum=.98):
        for dest,src in zip(self.target_encoder.parameters(),self.encoder.parameters()):
            dest.lerp_(src,1-momentum)

    def predict(self,state:torch.Tensor,action_ids:torch.Tensor):
        z=self.encoder(state)
        return self.predictor(torch.cat([z,self.action_embeddings(action_ids)],dim=-1))

    def one_step_score(self,state:tuple[int,...],target:tuple[int,...],actions:list[str]):
        """Produce prior weights; never used as an exact verifier."""
        if not actions:return {}
        with torch.no_grad():
            current=state_features(state).unsqueeze(0).expand(len(actions),-1)
            target_z=F.normalize(self.target_encoder(state_features(target).unsqueeze(0)),dim=-1)
            ids=torch.tensor([self.names.index(a) for a in actions],dtype=torch.long)
            pred=F.normalize(self.predict(current,ids),dim=-1)
            similarities=(pred*target_z).sum(dim=-1)
            probs=torch.softmax(similarities*2.,dim=0).tolist()
        return dict(zip(actions,probs))

@dataclass
class LatentTrainResult:
    steps:int
    loss_initial:float
    loss_final:float
    heldout_prediction_cosine:float
    baseline_random_cosine:float
    def as_dict(self):return asdict(self)


def train_latent_model(model:LatentDynamics,steps=200,seed=2026,
                       batch_size=32,learning_rate=.002)->LatentTrainResult:
    if not 1<=steps<=5000:raise ValueError('steps must be in [1,5000]')
    rng=random.Random(seed);torch.manual_seed(seed)
    macros=list(model.macro_lookup.values())
    def draw_batch():
        xs=[];acts=[];ys=[]
        while len(xs)<batch_size:
            state=tuple(rng.randint(-20,20) for _ in range(5))
            macro=rng.choice(macros)
            next_state=tuple(macro.apply(v) for v in state)
            if None in next_state:continue
            xs.append(state_features(state));ys.append(state_features(next_state))
            acts.append(model.names.index(macro.name))
        return torch.stack(xs),torch.tensor(acts),torch.stack(ys)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=learning_rate)
    losses=[]
    model.train()
    for step in range(steps):
        x,a,y=draw_batch()
        optimizer.zero_grad()
        prediction=model.predict(x,a)
        with torch.no_grad():target=model.target_encoder(y)
        loss=(F.normalize(prediction,dim=-1)-F.normalize(target,dim=-1)).square().mean()
        # Variance regularization discourages collapsed latent predictions.
        if len(prediction)>2:
            stdev=torch.sqrt(prediction.var(dim=0,unbiased=False)+1e-4)
            loss+=0.03*F.relu(.20-stdev).mean()
        loss.backward();optimizer.step()
        model.update_target()
        losses.append(float(loss.item()))
    model.eval()
    x,a,y=draw_batch()
    with torch.no_grad():
        pred=F.normalize(model.predict(x,a),dim=-1)
        target=F.normalize(model.target_encoder(y),dim=-1)
        perm=torch.randperm(len(target))
        cosine=(pred*target).sum(dim=-1).mean().item()
        control=(pred*target[perm]).sum(dim=-1).mean().item()
    return LatentTrainResult(steps,losses[0],losses[-1],cosine,control)
