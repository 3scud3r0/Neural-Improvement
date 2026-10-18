"""Shared PUCT-style Monte Carlo Tree Search for typed postfix programs.

Search values = float64 Python statistics. Verifier = exact bounded integers.
Policy can be uniform (rigorous random-network control) or a trained society.
Deep *maximum* depth does not imply searching all nodes at that depth.
"""
from __future__ import annotations
from dataclasses import dataclass,field
from typing import Callable,Sequence
import math,random,time
import torch
from .grammar import TOKENS,TOKEN_TO_ID,valid_actions,value_on_examples,HALT
from .features import encode
from .tasks import Task
from .model import NeuralSociety
from .memory import PersistentMemory
from .telemetry import Telemetry

@dataclass
class SearchNode:
    prefix:tuple[str,...]
    parent:'SearchNode|None'=None
    prior:float=1.0
    visits:int=0
    score_sum:float=0.
    children:dict[str,'SearchNode']=field(default_factory=dict)
    expanded:bool=False

    @property
    def q(self):return self.score_sum/self.visits if self.visits else 0.

@dataclass
class SearchResult:
    program:tuple[str,...]|None
    example_exact:bool
    score:float
    simulations:int
    expanded_nodes:int
    max_depth_reached:int
    source:str
    elapsed_sec:float

    def as_dict(self):
        return {'program':list(self.program) if self.program else None,
                'example_exact':self.example_exact,'score':self.score,
                'simulations':self.simulations,'expanded_nodes':self.expanded_nodes,
                'max_depth_reached':self.max_depth_reached,
                'source':self.source,'elapsed_sec':self.elapsed_sec}

class SharedMCTS:
    def __init__(self,model:NeuralSociety|None=None, memory:PersistentMemory|None=None,
                 telemetry:Telemetry|None=None, max_depth=15, simulations=300,
                 exploration=1.5, rollout_depth=15, seed=2026,
                 use_neural=True, stop_on_exact=True, record_every=0,
                 ablate_communication=False,world_model=None,world_weight=0.0):
        self.model=model
        self.memory=memory
        self.telemetry=telemetry
        self.max_depth=max_depth
        self.simulations=simulations
        self.exploration=exploration
        self.rollout_depth=rollout_depth
        self.rng=random.Random(seed)
        self.use_neural=use_neural and model is not None
        self.stop_on_exact=stop_on_exact
        self.record_every=record_every
        self.ablate_communication=ablate_communication
        self.world_model=world_model
        self.world_weight=world_weight
        self.eval_cache={}
        self.expanded_nodes=0
        self.max_depth_reached=0

    def _priors_and_value(self,task,prefix):
        key=tuple(prefix)
        if key in self.eval_cache:return self.eval_cache[key]
        actions=valid_actions(prefix,self.max_depth)
        if not actions:return {},0.
        if not self.use_neural:
            priors={a:1/len(actions) for a in actions}
            result=(priors,0.5)
        else:
            p=next(self.model.parameters())
            with torch.no_grad():
                out=self.model(encode(task.public(),prefix,device=p.device,dtype=p.dtype),
                               force_no_messages=self.ablate_communication)
                indices=torch.tensor([TOKEN_TO_ID[a] for a in actions],device=p.device)
                logits=out.logits[indices].float()
                probs=torch.softmax(logits,dim=0).cpu().tolist()
                result=({a:float(p) for a,p in zip(actions,probs)},float(out.value.item()))
                if self.telemetry and self.telemetry.level=='full':
                    self.telemetry.neural(self.expanded_nodes,out)
        # Optional learned model of symbolic transitions: favor states whose
        # effect the predictor understands. This is a heuristic, not a proof.
        if self.world_model is not None and self.world_weight>0:
            priors,critic=result
            import math
            penalties={a:math.exp(-min(50,self.world_weight*self.world_model.surprise(task,key,a)))
                       for a in priors}
            total=sum(priors[a]*penalties[a] for a in priors)
            if total>0:
                priors={a:priors[a]*penalties[a]/total for a in priors}
                result=(priors,critic)
        self.eval_cache[key]=result
        return result

    def _expand(self,node,task):
        if node.expanded:return
        priors,_=self._priors_and_value(task,node.prefix)
        node.children={a:SearchNode(prefix=node.prefix+(a,),parent=node,prior=float(prior))
                       for a,prior in priors.items()}
        node.expanded=True
        self.expanded_nodes+=1

    def _select(self,node):
        sqrt_parent=math.sqrt(max(1,node.visits))
        return max(node.children.values(),key=lambda child:
                   child.q+self.exploration*child.prior*sqrt_parent/(1+child.visits))

    def _rollout(self,task,prefix):
        p=list(prefix)
        for _ in range(max(0,min(self.rollout_depth,self.max_depth-len(p)))):
            if p and p[-1]==HALT: break
            available=valid_actions(p,self.max_depth)
            if not available:break
            # Rollouts use neural probabilities with 60% probability;
            # uniform sampling otherwise prevents complete policy lock-in.
            if self.use_neural and self.rng.random()<0.6:
                prior,_=self._priors_and_value(task,tuple(p))
                a=self.rng.choices(list(prior),weights=list(prior.values()),k=1)[0]
            else:
                # Prefer compiling complete programs to obtain feedback.
                if HALT in available and self.rng.random()<0.33:
                    a=HALT
                else:
                    a=self.rng.choice(available)
            p.append(a)
        return tuple(p) if p and p[-1]==HALT else None

    def solve(self,task:Task, store=True)->SearchResult:
        """Only task.examples are used for the search; heldout tests are ignored."""
        started=time.monotonic()
        public=task.public()
        self.eval_cache.clear();self.expanded_nodes=0;self.max_depth_reached=0
        best=None;best_score=-1.0;best_source='search'
        if self.memory is not None:
            for hit in self.memory.retrieve(public):
                if hit['score']>best_score:
                    best_score=hit['score'];best=tuple(hit['program']);best_source='memory'
                if hit['exact']:
                    result=SearchResult(best,True,best_score,0,0,len(best),'memory',time.monotonic()-started)
                    if self.telemetry:self.telemetry.log('memory_hit',family=task.family,score=best_score,program=list(best))
                    return result
        root=SearchNode(prefix=())
        self._expand(root,public)
        iterations=0
        for t in range(self.simulations):
            iterations=t+1
            node=root;trajectory=[root]
            while node.expanded and node.children and node.prefix[-1:]!=(HALT,):
                node=self._select(node)
                trajectory.append(node)
                if node.visits==0: break
            self.max_depth_reached=max(self.max_depth_reached,len(node.prefix))
            evaluated=None
            if node.prefix[-1:]==(HALT,):
                evaluated=node.prefix
            elif len(node.prefix)<self.max_depth:
                self._expand(node,public)
                evaluated=self._rollout(public,node.prefix)
            reward=0.0
            if evaluated is not None:
                score,exact=value_on_examples(evaluated,public.examples)
                # Mild description-length preference; does not affect exact verification.
                reward=score*(1.0-0.002*len(evaluated))
                if score>best_score or (score==best_score and (best is None or len(evaluated)<len(best))):
                    best_score=score;best=evaluated;best_source='search'
                if exact and self.stop_on_exact:
                    for n in trajectory:
                        n.visits+=1;n.score_sum+=reward
                    break
            if evaluated is None and self.use_neural:
                _,critic=self._priors_and_value(public,node.prefix)
                reward=critic*0.2
            for n in trajectory:
                n.visits+=1;n.score_sum+=float(reward)
            if self.telemetry and self.record_every and (t+1)%self.record_every==0:
                self.telemetry.log('mcts',simulations=t+1,nodes=self.expanded_nodes,
                                   best_score=best_score,best_program=list(best) if best else None,
                                   max_depth=self.max_depth_reached)
        exact=bool(best and value_on_examples(best,public.examples)[1])
        if exact and store and self.memory:self.memory.remember(public,best,'mcts')
        result=SearchResult(best,exact,max(best_score,0),iterations,self.expanded_nodes,
                            self.max_depth_reached,best_source,time.monotonic()-started)
        if self.telemetry:self.telemetry.log('search_complete',**result.as_dict(),family=task.family)
        return result
