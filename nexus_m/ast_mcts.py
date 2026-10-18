"""Monte Carlo Tree Search over compositional AST-construction decisions.

Search nodes represent *partial typed program sketches*, not string fragments.
Each complete sketch becomes genuine Python AST; correctness and speed are
measured by the child-process evaluator. No LLM is involved.
"""
from __future__ import annotations
from dataclasses import dataclass,field,asdict
from pathlib import Path
import json,math,random,time
from .ast_program_synthesis import FIELDS,RouterSpec,compile_source
from .ast_candidate_runtime import CandidateMeasurement,evaluate

@dataclass
class _Node:
    choices:tuple[str,...]=()
    visits:int=0
    total_reward:float=0.
    children:dict[str,'_Node']=field(default_factory=dict)
    @property
    def mean(self):return self.total_reward/self.visits if self.visits else 0.

@dataclass
class SynthesisReport:
    simulations:int
    unique_candidates:int
    accepted_candidates:int
    best_spec:dict
    best_train_gain:float
    elapsed_seconds:float
    trajectory:list[dict]
    candidates:list[dict]
    best_source:str
    def as_dict(self):return asdict(self)

class ASTSynthesisMCTS:
    def __init__(self,simulations=96,seed=2026,exploration=1.7,
                 checks=45,timing=22,timeout=8.0,baseline_spec:RouterSpec|None=None):
        if not 1<=simulations<=500:raise ValueError('synthesis budget 1..500')
        self.simulations=simulations;self.rng=random.Random(seed)
        self.seed=seed;self.exploration=exploration
        self.checks=checks;self.timing=timing;self.timeout=timeout
        self.baseline_spec=baseline_spec or RouterSpec()
        self.cache:dict[RouterSpec,CandidateMeasurement]={}
        self.root=_Node()

    def _select(self):
        node=self.root;path=[node]
        while len(node.choices)<len(FIELDS):
            name,options=FIELDS[len(node.choices)]
            unexpanded=[v for v in options if v not in node.children]
            if unexpanded:
                chosen=self.rng.choice(unexpanded)
                child=_Node(node.choices+(chosen,))
                node.children[chosen]=child
                node=child;path.append(node)
                break
            parent_visits=max(1,node.visits)
            node=max(node.children.values(),key=lambda c:
                (c.mean+self.exploration*math.sqrt(math.log(parent_visits+1)/(c.visits+1))
                 + self.rng.random()*1e-8))
            path.append(node)
        choices=list(node.choices)
        while len(choices)<len(FIELDS):
            choices.append(self.rng.choice(FIELDS[len(choices)][1]))
        return RouterSpec(*choices),path

    @staticmethod
    def _reward(m:CandidateMeasurement)->float:
        if not m.equivalent:return -1.
        return 0.12+max(-0.9,min(0.88,m.improvement))

    def search(self,*,max_unique:int=24)->SynthesisReport:
        started=time.perf_counter();trajectory=[]
        best=RouterSpec()
        best_gain=-1e20
        for t in range(self.simulations):
            spec,path=self._select()
            if spec not in self.cache and len(self.cache)<max_unique:
                self.cache[spec]=evaluate(spec,seed=self.seed+11,checks=self.checks,
                            timing=self.timing,timeout=self.timeout,baseline_spec=self.baseline_spec)
            # Once unique budget is exhausted, replay only cached results.
            m=self.cache.get(spec)
            reward=self._reward(m) if m else -1.
            for n in path:
                n.visits+=1;n.total_reward+=reward
            if m and m.equivalent and m.improvement>best_gain:
                best_gain=m.improvement;best=spec
            trajectory.append({'simulation':t+1,'candidate':spec.as_dict(),
                    'cached':m is not None,'reward':reward,
                    'best_gain':best_gain if best_gain>-1e10 else None,
                    'unique_evaluated':len(self.cache)})
        # In case no candidate passed, return canonical baseline.
        if best_gain==-1e20:
            best=RouterSpec();best_gain=0.
        return SynthesisReport(self.simulations,len(self.cache),
            sum(m.equivalent for m in self.cache.values()),best.as_dict(),
            best_gain,time.perf_counter()-started,trajectory,
            [m.as_dict() for m in self.cache.values()],compile_source(best))
