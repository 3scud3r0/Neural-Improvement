"""Finite structural causal models, do-interventions and graphical criteria.

Implements exact binary SCM enumeration, the backdoor/frontdoor adjustment
formulas for their supported graph patterns, and a small physics-regression
experiment. NOT a complete do-calculus engine or a universal physics simulator.
"""
from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations,product
from typing import Callable
import networkx as nx
import numpy as np
from networkx.algorithms.d_separation import is_d_separator


class CausalError(ValueError):pass

@dataclass
class CausalEstimate:
    value:float
    method:str
    identifiable:bool
    assumptions:list[str]
    def as_dict(self):return vars(self)


class CausalGraph:
    def __init__(self,edges:list[tuple[str,str]],nodes:list[str]|None=None,
                 latent:set[str]|None=None):
        self.graph=nx.DiGraph()
        self.graph.add_nodes_from(nodes or [])
        self.graph.add_edges_from(edges)
        if not nx.is_directed_acyclic_graph(self.graph):raise CausalError('causal graph must be a DAG')
        self.latent=set(latent or ())
        if not self.latent<=set(self.graph.nodes):raise CausalError('unknown latent node')

    def backdoor_valid(self,x:str,y:str,z:set[str]|None=None)->bool:
        z=set(z or ())
        if not {x,y}|z<=set(self.graph.nodes):raise CausalError('unknown node')
        if any(a in nx.descendants(self.graph,x) for a in z):return False
        if z & self.latent:return False
        backdoor=self.graph.copy()
        backdoor.remove_edges_from(list(backdoor.out_edges(x)))
        return bool(is_d_separator(backdoor,{x},{y},z))

    def backdoor_sets(self,x:str,y:str,max_set=4):
        options=sorted(set(self.graph.nodes)-self.latent-{x,y}-nx.descendants(self.graph,x))
        sets=[]
        for length in range(min(len(options),max_set)+1):
            for subset in combinations(options,length):
                if self.backdoor_valid(x,y,set(subset)):sets.append(subset)
        return sets

    def frontdoor_valid(self,x:str,y:str,m:str)->bool:
        if len({x,y,m})<3:return False
        if self.latent.intersection({x,y,m}):return False
        if m not in self.graph:return False
        if not nx.has_path(self.graph,x,m) or not nx.has_path(self.graph,m,y):return False
        # Every directed path from X to Y passes through mediator M.
        if any(m not in path for path in nx.all_simple_paths(self.graph,x,y)):return False
        return self.backdoor_valid(x,m,set()) and self.backdoor_valid(m,y,{x})


class BinarySCM:
    """Small SCM: each binary node is a function(parents, exogenous_bit).

    All exogenous bits are mutually independent with configurable Bernoulli
    probabilities. Probabilities are calculated by exact enumeration; maximum
    of 16 endogenous variables avoids exponential blow-up.
    """
    def __init__(self,graph:CausalGraph,functions:dict[str,Callable],
                 exogenous_probabilities:dict[str,float]|None=None):
        self.graph=graph
        self.order=list(nx.topological_sort(graph.graph))
        if len(self.order)>16:raise CausalError('exact enumeration limited to 16 nodes')
        if set(functions)!=set(self.order):raise CausalError('one structural function per node')
        self.functions=functions
        self.noise_p={v:float((exogenous_probabilities or {}).get(v,0.5)) for v in self.order}
        if any(not 0<=v<=1 for v in self.noise_p.values()):raise CausalError('invalid noise probability')

    def _enumerate(self,do:dict[str,int]|None=None):
        do=do or {}
        if not set(do)<=set(self.order) or any(v not in (0,1) for v in do.values()):
            raise CausalError('invalid do intervention')
        for noise in product((0,1),repeat=len(self.order)):
            probability=1.
            states={}
            for name,bit in zip(self.order,noise):
                p=self.noise_p[name]
                probability*=p if bit else 1-p
                parents={parent:states[parent] for parent in self.graph.graph.predecessors(name)}
                states[name]=do[name] if name in do else int(self.functions[name](parents,bit))
                if states[name] not in (0,1):raise CausalError('structural functions must return binary')
            if probability>0:yield states,probability

    def probability(self,query:dict[str,int],condition:dict[str,int]|None=None,
                    do:dict[str,int]|None=None)->float:
        condition=condition or {}
        variables=set(query)|set(condition)
        if not variables<=set(self.order):raise CausalError('unknown variable')
        total=0.;success=0.
        for row,weight in self._enumerate(do):
            if all(row[k]==v for k,v in condition.items()):
                total+=weight
                if all(row[k]==v for k,v in query.items()):success+=weight
        if total<1e-12:raise CausalError('zero-probability conditioning event')
        return success/total

    def do_effect(self,x:str,y:str,value:int)->CausalEstimate:
        return CausalEstimate(self.probability({y:1},do={x:value}),
                              'exact_SCM_do_enumeration',True,
                              ['fully specified binary structural functions',
                               'independent exogenous noises'])

    def identify_backdoor(self,x:str,y:str,value:int,adjust_for:tuple[str,...]|None=None)->CausalEstimate:
        choices=self.graph.backdoor_sets(x,y)
        if adjust_for is None:
            if not choices:return CausalEstimate(float('nan'),'backdoor',False,['no observed valid adjustment set'])
            adjust_for=choices[0]
        if not self.graph.backdoor_valid(x,y,set(adjust_for)):
            return CausalEstimate(float('nan'),'backdoor',False,['invalid backdoor adjustment'])
        answer=0.
        for vals in product((0,1),repeat=len(adjust_for)):
            z=dict(zip(adjust_for,vals))
            p_z=self.probability(z)
            if not p_z:continue
            answer+=self.probability({y:1},{x:value,**z})*p_z
        return CausalEstimate(answer,'backdoor_adjustment',True,
                              ['causal DAG correct','no unblocked confounders after adjustment','positivity'])

    def identify_frontdoor(self,x:str,y:str,m:str,value:int)->CausalEstimate:
        if not self.graph.frontdoor_valid(x,y,m):
            return CausalEstimate(float('nan'),'frontdoor',False,['frontdoor graphical criterion not satisfied'])
        answer=0.
        for mediator in (0,1):
            p_m=self.probability({m:mediator},{x:value})
            if not p_m:continue
            sum_x=0.
            for observed_x in (0,1):
                p_x=self.probability({x:observed_x})
                if p_x:sum_x+=self.probability({y:1},{m:mediator,x:observed_x})*p_x
            answer+=p_m*sum_x
        return CausalEstimate(answer,'frontdoor_adjustment',True,
                              ['causal DAG correct','all frontdoor criteria hold','positivity'])


@dataclass
class PhysicsFit:
    coefficients:dict[str,float]
    heldout_r2:float
    family:str


def infer_linear_damped_motion(positions,velocities,accelerations,seed=2026)->PhysicsFit:
    """Fit a = c_x*x + c_v*v + c_0; one restricted physics family."""
    x=np.asarray(positions,dtype=float)
    v=np.asarray(velocities,dtype=float)
    a=np.asarray(accelerations,dtype=float)
    if len(x)!=len(v) or len(x)!=len(a) or len(x)<12:
        raise ValueError('at least 12 equal-length measurements')
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(v)) or not np.all(np.isfinite(a)):
        raise ValueError('measurements must be finite')
    design=np.column_stack([x,v,np.ones(len(x))])
    rng=np.random.default_rng(seed)
    order=rng.permutation(len(x));split=max(8,int(.75*len(x)))
    train,test=order[:split],order[split:]
    coefs,_,_,_=np.linalg.lstsq(design[train],a[train],rcond=None)
    predicted=design[test]@coefs
    denominator=np.sum((a[test]-a[test].mean())**2)
    r2=1-float(np.sum((a[test]-predicted)**2))/denominator if denominator>1e-12 else float('nan')
    return PhysicsFit({'position_coefficient':float(coefs[0]),'velocity_coefficient':float(coefs[1]),
                       'intercept':float(coefs[2])},r2,'damped_linear_motion')
