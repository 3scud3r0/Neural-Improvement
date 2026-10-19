"""Equality saturation over pure mathematical integer polynomials.

No rewrites of floats, tensor products, calls or effectful Python. E-graphs
can grow exponentially too; node/iteration budgets are part of this engine.
"""
from __future__ import annotations
import ast
from copy import deepcopy
from dataclasses import dataclass
import math
from .typed_program import IntExpr, typed_function
from .function_ast import Proposal


@dataclass(frozen=True)
class ENode:
    op: str
    args: tuple[int,...] = ()
    value: int | str | None = None


class EGraph:
    RULES = {'commutative','associative','identities','constant_fold','distribute','factor'}
    def __init__(self, max_nodes=10_000):
        if max_nodes<1:raise ValueError('max_nodes must be positive')
        self.memo={};self.classes={};self.union_find={};self.next_id=0
        self.max_nodes=max_nodes

    def find(self, cid):
        if cid not in self.union_find:raise KeyError(cid)
        root=cid
        while self.union_find[root]!=root:root=self.union_find[root]
        while cid!=root:
            parent=self.union_find[cid];self.union_find[cid]=root;cid=parent
        return root

    def add(self, enode):
        if enode.op not in ('var','const','add','sub','mul'):raise ValueError('e-graph requires integer polynomials')
        arity=0 if enode.op in ('var','const') else 2
        if len(enode.args)!=arity:raise ValueError('bad e-node arity')
        if enode.op=='const' and type(enode.value) is not int:raise ValueError('integer constant required')
        if enode.op=='var' and not isinstance(enode.value,str):raise ValueError('variable name required')
        node=ENode(enode.op,tuple(self.find(a) for a in enode.args),enode.value)
        if node in self.memo:return self.find(self.memo[node])
        if sum(map(len,self.classes.values()))>=self.max_nodes:raise OverflowError('e-graph node budget')
        cid=self.next_id;self.next_id+=1
        self.union_find[cid]=cid;self.classes[cid]={node};self.memo[node]=cid
        return cid

    def union(self,a,b):
        a,b=self.find(a),self.find(b)
        if a==b:return a
        a,b=min(a,b),max(a,b)
        self.union_find[b]=a;self.classes[a].update(self.classes.pop(b))
        return a

    def rebuild(self):
        """Restore canonical children and congruence after unions."""
        while True:
            memo={};collision=False
            for cid,nodes in list(self.classes.items()):
                for node in tuple(nodes):
                    canonical=ENode(node.op,tuple(self.find(a) for a in node.args),node.value)
                    if canonical in memo and self.find(cid)!=self.find(memo[canonical]):
                        self.union(cid,memo[canonical]);collision=True
                    memo[canonical]=self.find(cid)
            if not collision:break
        classes={}
        for node,cid in memo.items():classes.setdefault(self.find(cid),set()).add(node)
        self.classes=classes;self.memo={n:cid for cid,nodes in classes.items() for n in nodes}

    def from_expr(self, expression):
        return self.add(ENode(expression.op,tuple(self.from_expr(a) for a in expression.args),expression.value))

    def from_ast(self,node,variables=('x',)):
        return self.from_expr(IntExpr.from_ast(node,variables))

    def saturate(self,rewrite_axioms=None,*,max_iterations=12):
        rules=set(rewrite_axioms) if rewrite_axioms is not None else self.RULES
        if not rules<=self.RULES:raise ValueError('unknown rewrite axiom')
        if max_iterations<1:raise ValueError('positive iteration budget required')
        stopped='iteration_budget';iterations=0
        def constant(cid,value):return ENode('const',(),value) in self.classes[self.find(cid)]
        try:
            for iterations in range(1,max_iterations+1):
                before=(len(self.memo),len(self.classes),self.next_id)
                for cid,nodes in list(self.classes.items()):
                    for node in list(nodes):
                        if len(node.args)!=2:continue
                        a,b=map(self.find,node.args)
                        if 'commutative' in rules and node.op in ('add','mul'):
                            self.union(cid,self.add(ENode(node.op,(b,a))))
                        if 'identities' in rules:
                            if (node.op=='add' and constant(b,0)) or (node.op=='sub' and constant(b,0)) or (node.op=='mul' and constant(b,1)):
                                self.union(cid,a)
                            if node.op=='mul' and (constant(a,0) or constant(b,0)):
                                self.union(cid,self.add(ENode('const',(),0)))
                            if node.op=='sub' and a==b:self.union(cid,self.add(ENode('const',(),0)))
                        left=list(self.classes[self.find(a)]);right=list(self.classes[self.find(b)])
                        if 'constant_fold' in rules:
                            for x in left:
                                for y in right:
                                    if x.op==y.op=='const':
                                        value={'add':lambda:x.value+y.value,'sub':lambda:x.value-y.value,
                                               'mul':lambda:x.value*y.value}[node.op]()
                                        if abs(value).bit_length()<=4096:self.union(cid,self.add(ENode('const',(),value)))
                        for child in right:
                            if 'associative' in rules and child.op==node.op and node.op in ('add','mul'):
                                x,y=child.args
                                inner=self.add(ENode(node.op,(a,x)))
                                self.union(cid,self.add(ENode(node.op,(inner,y))))
                            if 'distribute' in rules and node.op=='mul' and child.op=='add':
                                x,y=child.args
                                ax=self.add(ENode('mul',(a,x)));ay=self.add(ENode('mul',(a,y)))
                                self.union(cid,self.add(ENode('add',(ax,ay))))
                        if 'factor' in rules and node.op=='add':
                            for x in left:
                                for y in right:
                                    if x.op==y.op=='mul' and self.find(x.args[0])==self.find(y.args[0]):
                                        tail=self.add(ENode('add',(x.args[1],y.args[1])))
                                        self.union(cid,self.add(ENode('mul',(x.args[0],tail))))
                self.rebuild()
                if before==(len(self.memo),len(self.classes),self.next_id):stopped='fixed_point';break
        except OverflowError:
            stopped='node_budget';self.rebuild()
        return dict(iterations=iterations,stop_reason=stopped,nodes=len(self.memo),classes=len(self.classes))

    def extract_expr(self,cid,cost_weights=None):
        costs={'const':1.,'var':1.,'add':2.,'sub':2.,'mul':4.,**(cost_weights or {})}
        if any(not math.isfinite(v) or v<=0 for v in costs.values()):raise ValueError('costs must be positive and finite')
        self.rebuild();best={}
        # Positive-cost fixed point handles self-cycles such as x == x + 0.
        for _ in range(len(self.classes)+1):
            changed=False
            for owner,nodes in sorted(self.classes.items()):
                for node in sorted(nodes,key=repr):
                    children=[self.find(a) for a in node.args]
                    if not all(a in best for a in children):continue
                    cost=costs[node.op]+sum(best[a][0] for a in children)
                    if owner not in best or cost<best[owner][0]:
                        best[owner]=(cost,IntExpr(node.op,tuple(best[a][1] for a in children),node.value));changed=True
            if not changed:break
        root=self.find(cid)
        if root not in best:raise ValueError('e-class has no finite extractable expression')
        return best[root][1]

    def extract_optimal(self,class_id,cost_weights=None):
        return ast.fix_missing_locations(self.extract_expr(class_id,cost_weights).to_ast())


def egraph_proposals(function,rng):
    try:
        variables,original=typed_function(function)
        graph=EGraph(max_nodes=2000);root=graph.from_expr(original)
        graph.saturate(max_iterations=6)
        selected=graph.extract_expr(root)
        if selected==original:return
        if original.polynomial()!=selected.polynomial():raise AssertionError('e-graph rewrite violated integer polynomial identity')
    except (ValueError,OverflowError):return
    candidate=deepcopy(function)
    candidate.body=[ast.Return(value=selected.to_ast())]
    yield Proposal('egraph-integer-polynomial',ast.fix_missing_locations(candidate))
