"""Fixed-dimensional, normalized features; no hidden test information leaks in."""
from __future__ import annotations
import math
import torch
from .grammar import TOKENS, TOKEN_TO_ID
from .tasks import Task
FEATURE_DIM=48


def encode(task: Task, prefix=(), retrieved_count=0, device='cpu', dtype=torch.float32) -> torch.Tensor:
    xs=[x for x,_ in task.examples]
    ys=[y for _,y in task.examples]
    v=[0.0]*FEATURE_DIM
    if not xs: return torch.zeros(FEATURE_DIM,device=device,dtype=dtype)
    def bounded(z): return math.tanh(float(z)/20)
    for j,(x,y) in enumerate(task.examples[:5]):
        v[2*j]=bounded(x)
        v[2*j+1]=bounded(y)
    v[10]=bounded(sum(xs)/len(xs))
    v[11]=bounded(sum(ys)/len(ys))
    v[12]=bounded((max(xs)-min(xs)))
    v[13]=bounded((max(ys)-min(ys)))
    v[14]=len(task.examples)/10
    v[15]=min(len(prefix),32)/32
    v[16]=min(retrieved_count,20)/20
    # Histogram of instructions in partial program.
    for t in prefix:
        k=TOKEN_TO_ID.get(t)
        if k is not None and k < 19:
            v[17+k]+=1/16
    depth=0
    for t in prefix:
        if t=='X' or t.startswith('C'): depth+=1
        elif t in ('ADD','SUB','MUL','MIN','MAX'): depth-=1
    v[37]=depth/6
    v[38]=int('X' in prefix)
    v[39]=sum(int(y>=0) for y in ys)/len(ys)
    v[40]=bounded(sum(abs(y) for y in ys)/len(ys))
    v[41]=bounded(sum(x*y for x,y in task.examples)/len(xs))
    v[42]=bounded(sum(x*x for x in xs)/len(xs))
    v[43]=bounded(sum(y*y for y in ys)/len(xs))
    v[44]=bounded(len(set(ys)))
    v[45]=math.tanh(len(set(prefix))/10)
    v[46]=1.0
    v[47]=bounded(ys[0]-xs[0])
    return torch.tensor(v,dtype=dtype,device=device)
