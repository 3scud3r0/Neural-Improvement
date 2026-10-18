"""Streaming event log, parameter-delta statistics and optional full tensor dumps.

'full' is genuinely expensive: O(agents*width*recorded_steps) activation
storage, plus NxN routing matrices and optionally all parameter snapshots.
"""
from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
import json, time, os
import torch

class Telemetry:
    def __init__(self, directory, level='summary', flush_every=1,
                 checkpoint_every=0, record_weight_tensors=False):
        if level not in ('off','summary','full'): raise ValueError('level=off|summary|full')
        self.directory=Path(directory)
        self.directory.mkdir(parents=True,exist_ok=True)
        (self.directory/'tensor_snapshots').mkdir(exist_ok=True)
        self.level=level
        self.flush_every=max(1,flush_every)
        self.checkpoint_every=checkpoint_every
        self.record_weight_tensors=record_weight_tensors
        self.stream=None if level=='off' else (self.directory/'events.jsonl').open('a',encoding='utf-8',buffering=1)
        self.counter=0
        self.weights_previous={}

    def log(self,kind,**attributes):
        if self.stream is None:return
        self.counter+=1
        event={'utc':datetime.now(timezone.utc).isoformat(),'type':kind,'event_id':self.counter,**attributes}
        self.stream.write(json.dumps(event,ensure_ascii=False,default=str)+'\n')
        if self.counter%self.flush_every==0:self.stream.flush()

    @torch.no_grad()
    def neural(self,step,output,full_every=1):
        if self.level=='off' or step%max(1,full_every)!=0:return
        message=output.messages.detach().cpu()
        routing=output.routing.detach().cpu()
        self.log('communication',step=step,round_count=1,
                 messages=float((routing>0).sum().item()),
                 mean_message_norm=round(float(message.norm(dim=-1).mean()),6),
                 max_message_abs=round(float(message.abs().max()),6),
                 mean_active_routes=round(float((routing>0).sum(dim=-1).float().mean()),4),
                 influence_entropy=float(-(output.influence*output.influence.clamp_min(1e-12).log()).sum()),
                 state_rms=float(output.states.square().mean().sqrt()))
        if self.level=='full':
            path=self.directory/'tensor_snapshots'/f'neural_{step:09d}.pt'
            torch.save({'states':output.states.detach().cpu(),
                        'messages':message,'routing':routing,
                        'influence':output.influence.detach().cpu()},path)
            self.log('neural_dump',step=step,path=str(path.relative_to(self.directory)))

    @torch.no_grad()
    def weights(self,step,model):
        if self.level=='off':return
        changes={}; grad_norms={}; total=0
        for name,p in model.named_parameters():
            current=p.detach()
            total+=p.numel()
            current_norm=float(current.float().norm())
            if name in self.weights_previous:
                difference=(current.cpu().float()-self.weights_previous[name])
                delta=float(difference.norm())
            else: delta=None
            changes[name]={'norm':round(current_norm,6),'delta':None if delta is None else round(delta,6)}
            if p.grad is not None: grad_norms[name]=round(float(p.grad.detach().float().norm()),6)
            self.weights_previous[name]=current.cpu().float().clone()
        self.log('parameters',step=step,total_parameters=total,stats=changes,grad_norms=grad_norms)
        if self.record_weight_tensors or (self.checkpoint_every and step%self.checkpoint_every==0):
            output=self.directory/'tensor_snapshots'/f'weights_{step:09d}.pt'
            torch.save({n:p.detach().cpu() for n,p in model.named_parameters()},output)
            self.log('weights_dump',step=step,path=str(output.relative_to(self.directory)))

    def close(self):
        if self.stream:self.stream.flush();self.stream.close();self.stream=None
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
