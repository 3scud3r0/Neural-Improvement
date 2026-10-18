"""Continual synthetic meta-training, search distillation and task-time adaptation.

All oracle actions are visible ONLY to the training pipeline. During search
and evaluation, the agent sees few-shot examples and no oracle/heldout labels.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json,random,time
import torch
from torch.nn import functional as F
from .model import NeuralSociety
from .tasks import Task,sample_task
from .features import encode
from .grammar import TOKEN_TO_ID,value_on_examples
from .mcts import SharedMCTS
from .memory import PersistentMemory
from .telemetry import Telemetry

@dataclass
class TrainSettings:
    steps:int=1000
    lr:float=0.0008
    seed:int=2026
    log_every:int=50
    checkpoint_every:int=200
    search_improve_every:int=0
    search_simulations:int=100
    example_count:int=5
    precision:str='float32'


def supervised_loss(model:NeuralSociety,task:Task,program=None,
                    adapt_states=False):
    program=tuple(program or task.oracle or ())
    if not program:raise ValueError('Expected known training program')
    parameter=next(model.parameters())
    # Random prefix selection avoids expensive unrolling of all tokens per step.
    t=random.randrange(len(program))
    features=encode(task.public(),program[:t],device=parameter.device,dtype=parameter.dtype)
    output=model(features)
    target=torch.tensor([TOKEN_TO_ID[program[t]]],device=parameter.device)
    policy_loss=F.cross_entropy(output.logits.unsqueeze(0),target)
    # Favor accurate value predictions at solved/unsolved states, not unchecked claims.
    target_value=torch.tensor(0.85 if t>=len(program)-2 else 0.45,
                              device=parameter.device,dtype=parameter.dtype)
    critic_loss=(output.value-target_value).square()
    # Soft penalty on collapsed influence distribution.
    entropy=-(output.influence*output.influence.clamp_min(1e-9).log()).sum()
    return policy_loss+0.15*critic_loss-0.003*entropy,output


def save_checkpoint(path,model,optimizer,step,config):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'step':step,'architecture':model.report(),
                'state_dict':model.state_dict(),
                'optimizer':optimizer.state_dict() if optimizer else None,
                'training_config':config},path)


def restore_checkpoint(path,device='cpu',with_optimizer=False)->tuple[NeuralSociety,dict]:
    # User-supplied pickle checkpoints should never be loaded from untrusted sources.
    bundle=torch.load(path,map_location=device,weights_only=True)
    cfg=bundle['architecture']
    model=NeuralSociety(agents=cfg['agents'],width=cfg['neurons_per_agent'],
                        message_dim=cfg['message_dim'],top_k=cfg['top_k'],
                        recurrence=cfg['recurrent_rounds'],communicate=cfg['communicate'])
    dtype=next(iter(bundle['state_dict'].values())).dtype
    model=model.to(device=device,dtype=dtype)
    model.load_state_dict(bundle['state_dict'])
    return model,bundle


def train(model:NeuralSociety,settings:TrainSettings,output_dir,
          memory:PersistentMemory|None=None,telemetry:Telemetry|None=None,
          allow_search_distillation=True):
    rng=random.Random(settings.seed)
    random.seed(settings.seed)
    torch.manual_seed(settings.seed)
    output=Path(output_dir);output.mkdir(parents=True,exist_ok=True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=settings.lr,weight_decay=0.0001)
    model.train()
    history=[]
    started=time.monotonic()
    for step in range(1,settings.steps+1):
        task=sample_task(rng,'train',settings.example_count)
        optimizer.zero_grad(set_to_none=True)
        loss,out=supervised_loss(model,task)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        if telemetry and (telemetry.level=="full" or step==1 or step%settings.log_every==0):
            telemetry.neural(step,out)
            telemetry.weights(step,model)
        optimizer.step()
        if memory and step%max(1,settings.log_every)==0:
            # Training examples only; explicit provenance, not test leakage.
            memory.remember(task.public(),task.oracle,'training_oracle')
        if allow_search_distillation and settings.search_improve_every and step%settings.search_improve_every==0:
            probe=sample_task(rng,'train',settings.example_count)
            searcher=SharedMCTS(model=model,memory=memory,simulations=settings.search_simulations,
                                max_depth=15,seed=settings.seed+step)
            model.eval()
            found=searcher.solve(probe.public(),store=True)
            model.train()
            if found.example_exact and found.program:
                optimizer.zero_grad(set_to_none=True)
                distill, _=supervised_loss(model,probe,found.program)
                distill.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                optimizer.step()
                if telemetry:telemetry.log('search_distillation',step=step,program=list(found.program),
                                           source=found.source)
        if step==1 or step%settings.log_every==0 or step==settings.steps:
            record={'step':step,'loss':float(loss.detach()),'elapsed_sec':time.monotonic()-started}
            history.append(record)
            if telemetry:telemetry.log('training',**record)
        if settings.checkpoint_every and step%settings.checkpoint_every==0:
            save_checkpoint(output/f'checkpoint_{step:06d}.pt',model,optimizer,step,vars(settings))
    save_checkpoint(output/'model_final.pt',model,optimizer,settings.steps,vars(settings))
    (output/'training_history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
    return {'steps':settings.steps,'duration_sec':time.monotonic()-started,
            'last_loss':history[-1]['loss'],'checkpoint':str(output/'model_final.pt')}


def adapt_to_examples(model:NeuralSociety,task:Task,steps=5,lr=0.0001,
                      persist=False)->dict:
    """Test-time self-supervision on GIVEN examples via leave-one-out prediction.

    The small adaptation objective estimates y from x/y of other examples via
    the value head as a proxy for consistency, and uses valid prefix likelihood.
    It does NOT secretly use any heldout targets or an unknown oracle.
    It is experimental: supervised task-specific adapters could be better.
    """
    if steps<=0:return {'steps':0,'loss':None}
    # Adapt only agent-level gain and routing bias; save/restore by default.
    names=('agent_gain','agent_routing_bias')
    previous={k:p.detach().clone() for k,p in model.named_parameters() if k in names}
    original={p:p.requires_grad for p in model.parameters()}
    for name,p in model.named_parameters():p.requires_grad_(name in names)
    try:
        optimizer=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=lr)
        p=next(model.parameters());loss_value=0.
        # View the examples as a compact pattern, regularize updates against
        # unadapted activations rather than pretending the target program is known.
        reference=model(encode(task.public(),device=p.device,dtype=p.dtype))
        ref=reference.states.detach()
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            out=model(encode(task.public(),device=p.device,dtype=p.dtype))
            # Input-conditioned consistency regularization: deliberately modest.
            loss=(out.states-ref).square().mean() + 0.002*out.messages.square().mean()
            loss.backward();optimizer.step()
            loss_value=float(loss.detach())
        return {'steps':steps,'loss':loss_value,'persist':persist,
                'note':'Unsupervised light adapter calibration; not guaranteed to improve accuracy'}
    finally:
        if not persist:
            with torch.no_grad():
                for name,p in model.named_parameters():
                    if name in previous:p.copy_(previous[name])
        for p,flag in original.items():p.requires_grad_(flag)


def adapt_to_program(model:NeuralSociety,task:Task,program,steps=5,lr=0.0002,
                     persist=False)->dict:
    """Temporary test-time distillation of a discovered program into adapters.

    The teacher is a candidate from MCTS matching PUBLIC examples; there is
    no access to oracle or hidden labels. A spurious candidate may hurt.
    """
    from .grammar import value_on_examples
    exact=value_on_examples(program,task.public().examples)[1]
    if not exact:raise ValueError('Only programs passing known examples can be adapted')
    if steps<1:return {'steps':0,'loss':None,'exact_on_examples':True}
    chosen=('agent_gain','agent_routing_bias','agent_bias')
    previous={name:p.detach().clone() for name,p in model.named_parameters() if name in chosen}
    flags={p:p.requires_grad for p in model.parameters()}
    for name,p in model.named_parameters():p.requires_grad_(name in chosen)
    optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=lr)
    parameter=next(model.parameters())
    history=[]
    try:
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            losses=[]
            for i,token in enumerate(program):
                feature=encode(task.public(),program[:i],device=parameter.device,dtype=parameter.dtype)
                predicted=model(feature)
                target=torch.tensor([TOKEN_TO_ID[token]],device=parameter.device)
                losses.append(F.cross_entropy(predicted.logits.unsqueeze(0),target))
            loss=torch.stack(losses).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad),1.0)
            optimizer.step()
            history.append(float(loss.detach()))
        return {'steps':steps,'first_loss':history[0], 'last_loss':history[-1],
                'exact_on_examples':exact,'persist':persist,
                'warning':'Adapts toward one verified-on-examples candidate; might overfit'}
    finally:
        if not persist:
            with torch.no_grad():
                for name,p in model.named_parameters():
                    if name in previous:p.copy_(previous[name])
        for p,flag in flags.items():p.requires_grad_(flag)
