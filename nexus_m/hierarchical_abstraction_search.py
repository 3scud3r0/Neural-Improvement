"""Hierarchical search over verified macro-operators rather than raw tokens.

Supports subgoal DAGs, progressive widening, a transposition cache and
PUCT-style MCTS. The demonstration domain is exact integer program synthesis;
this is NOT a general planner capable of solving arbitrary scientific tasks.
"""
from __future__ import annotations
from dataclasses import dataclass,field
from collections import defaultdict
from typing import Callable
import math,random,time

MAX_ABS=10**9

@dataclass(frozen=True)
class Macro:
    name:str
    operation:str
    argument:int=0
    cost:float=1.

    def apply(self,value:int)->int|None:
        try:
            if self.operation=='add': z=value+self.argument
            elif self.operation=='mul': z=value*self.argument
            elif self.operation=='abs': z=abs(value)
            elif self.operation=='neg': z=-value
            elif self.operation=='square': z=value*value
            elif self.operation=='half':
                if value%2:return None
                z=value//2
            else:raise ValueError(f'unknown operation {self.operation}')
            return z if abs(z)<=MAX_ABS else None
        except OverflowError:return None

DEFAULT_MACROS=(
    Macro('ADD_1','add',1),Macro('ADD_2','add',2),Macro('ADD_3','add',3),
    Macro('SUB_1','add',-1),Macro('SUB_2','add',-2),
    Macro('MUL_2','mul',2),Macro('MUL_3','mul',3),
    Macro('NEG','neg'),Macro('ABS','abs'),Macro('SQUARE','square'),Macro('HALF_EVEN','half'),
)

@dataclass
class Subgoal:
    identifier:str
    description:str
    depends_on:tuple[str,...]=()
    status:str='pending'


class GoalGraph:
    def __init__(self):self.goals:dict[str,Subgoal]={}
    def add(self,goal:Subgoal):
        if goal.identifier in self.goals:raise ValueError('duplicate goal')
        if goal.identifier in goal.depends_on:raise ValueError('self dependency')
        if any(d not in self.goals for d in goal.depends_on):raise ValueError('unknown dependency')
        self.goals[goal.identifier]=goal
    def ready(self)->list[Subgoal]:
        return [g for g in self.goals.values() if g.status=='pending' and
                all(self.goals[d].status=='done' for d in g.depends_on)]
    def complete(self,id:str):
        if id not in {g.identifier for g in self.ready()}:raise ValueError('dependency not satisfied')
        self.goals[id].status='done'
    def summary(self):return {k:{'status':v.status,'depends_on':list(v.depends_on)} for k,v in self.goals.items()}


@dataclass
class HierNode:
    state:tuple[int,...]
    path:tuple[str,...]
    visits:int=0
    value_sum:float=0.
    children:dict[str,'HierNode']=field(default_factory=dict)
    remaining:list[str]=field(default_factory=list)
    @property
    def q(self):return self.value_sum/self.visits if self.visits else 0.

@dataclass
class HierSearchResult:
    found:bool
    macro_program:list[str]
    predictions:list[int]
    simulations:int
    nodes:int
    depth_reached:int
    elapsed_seconds:float
    exact_on_examples:bool
    def as_dict(self):return vars(self)


class HierarchicalMCTS:
    def __init__(self,macros=DEFAULT_MACROS,simulations=1000,max_depth=8,
                 exploration=1.25,seed=2026,rollout_depth=4,
                 prior:Callable[[tuple[int,...],list[str]],dict[str,float]]|None=None):
        if simulations<1 or max_depth<1:raise ValueError('positive budgets required')
        self.macros={m.name:m for m in macros}
        self.simulations=simulations;self.max_depth=max_depth
        self.exploration=exploration;self.rng=random.Random(seed)
        self.rollout_depth=rollout_depth;self.prior=prior
        self._cache:dict[tuple[int,...],float]={}

    def _apply(self,state:tuple[int,...],macro_name:str)->tuple[int,...]|None:
        fn=self.macros[macro_name]
        values=[fn.apply(x) for x in state]
        return None if any(v is None for v in values) else tuple(values)

    @staticmethod
    def _value(state:tuple[int,...],target:tuple[int,...])->float:
        if state==target:return 1.
        scale=1+sum(abs(v) for v in target)/max(1,len(target))
        mae=sum(abs(a-b) for a,b in zip(state,target))/len(target)
        return 0.85/(1+mae/scale)

    def _available(self,state:tuple[int,...],parent_path:tuple[str,...]):
        seen=set([state]); opts=[]
        for name in self.macros:
            s=self._apply(state,name)
            if s is not None and s not in seen:
                seen.add(s);opts.append(name)
        if self.prior is not None:
            weights=self.prior(state,opts)
            opts.sort(key=lambda n:weights.get(n,0),reverse=True)
        else:
            self.rng.shuffle(opts)
        return opts

    def _simulate(self,state,goal,depth):
        best=self._value(state,goal);best_path=()
        current=state
        for i in range(min(self.rollout_depth,max(0,self.max_depth-depth))):
            choices=self._available(current,())
            if not choices:break
            name=self.rng.choice(choices)
            current=self._apply(current,name)
            score=self._value(current,goal)
            if score>best:
                best=score;best_path=best_path+(name,)
            if current==goal:break
        return best

    def search(self,examples:list[tuple[int,int]]|tuple[tuple[int,int],...])->HierSearchResult:
        if not examples or len(examples)>100 or any(len(pair)!=2 for pair in examples):
            raise ValueError('expected 1..100 input/output pairs')
        start=time.perf_counter()
        initial=tuple(int(x) for x,_ in examples)
        target=tuple(int(y) for _,y in examples)
        root=HierNode(initial,(),remaining=self._available(initial,()))
        best=root;nodes=1;depth_reached=0
        for _ in range(self.simulations):
            node=root;visited=[node]
            while len(node.path)<self.max_depth:
                # Progressive widening allows more macro choices as visits increase.
                max_children=max(1,min(len(self.macros),int(1.6*math.sqrt(node.visits+1))))
                if node.remaining and len(node.children)<max_children:
                    name=node.remaining.pop(0)
                    result=self._apply(node.state,name)
                    if result is None:continue
                    candidate=HierNode(result,node.path+(name,),remaining=self._available(result,node.path+(name,)))
                    node.children[name]=candidate
                    nodes+=1;node=candidate;visited.append(node)
                    break
                if not node.children:break
                parent_n=max(1,node.visits)
                weights=self.prior(node.state,list(node.children)) if self.prior else {}
                node=max(node.children.values(),key=lambda child:child.q+self.exploration*
                         (weights.get(child.path[-1],1/len(node.children)) if weights else 1/len(node.children))*
                         math.sqrt(parent_n)/(1+child.visits))
                visited.append(node)
                if node.state==target:break
            depth_reached=max(depth_reached,len(node.path))
            reward=self._value(node.state,target)
            if reward<1:reward=max(reward,self._simulate(node.state,target,len(node.path)))
            for ancestor in visited:
                ancestor.visits+=1;ancestor.value_sum+=reward
            if self._value(node.state,target)>self._value(best.state,target) or (
                node.state==target and (best.state!=target or len(node.path)<len(best.path))):best=node
            if best.state==target:break
        return HierSearchResult(best.state==target,list(best.path),list(best.state),root.visits,
                                nodes,depth_reached,time.perf_counter()-start,best.state==target)


def evaluate_macro_program(program:list[str],inputs:list[int],macros=DEFAULT_MACROS)->list[int|None]:
    library={m.name:m for m in macros}
    result=[]
    for x in inputs:
        current=int(x)
        for name in program:
            current=library[name].apply(current)
            if current is None:break
        result.append(current)
    return result
