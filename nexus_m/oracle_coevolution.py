"""Frozen-reference falsification and scoped proofs, not a source of intent.

Fuzzing finds counterexamples; passing it is not proof. Polynomial normal forms
prove equality over mathematical integers. Optional Z3 handles the supported
integer IR, including definedness, with a solver timeout and explicit unknown.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import ast
import math
import random
import time
from .typed_program import IntExpr


@dataclass
class FalsificationResult:
    status: str
    counterexample: dict[str,int] | None = None
    expected: tuple | None = None
    observed: tuple | None = None
    witness_iterations: int = 0
    proof_kind: str | None = None
    scope: str = 'mathematical integer expressions, not arbitrary Python'
    @property
    def falsified(self):return self.status=='falsified'
    @property
    def proved(self):return self.status=='proved'
    def as_dict(self):return asdict(self)


def outcome(expression,bindings,max_bits=4096):
    try:return ('value',expression.evaluate(bindings,max_bits=max_bits))
    except (ZeroDivisionError,OverflowError) as exc:return ('error',type(exc).__name__)


class AdversarialOracle:
    def __init__(self,seed=2026):self.seed=seed

    def cases(self,variables,*,count=256,seed=None):
        if not variables or count<1:raise ValueError('variables and positive case count required')
        rng=random.Random(self.seed if seed is None else seed)
        boundaries=(0,1,-1,2,-2,2**31-1,-2**31,2**63-1,-2**63,10**9,-10**9)
        for i in range(count):
            if i<len(boundaries):yield dict.fromkeys(variables,boundaries[i])
            elif i<len(boundaries)+len(variables)*len(boundaries):
                index=i-len(boundaries);case=dict.fromkeys(variables,0)
                case[variables[index%len(variables)]]=boundaries[index//len(variables)]
                yield case
            else:yield {v:rng.choice(boundaries) if rng.random()<.2 else rng.randint(-10**6,10**6) for v in variables}

    def compare(self,reference:IntExpr,candidate:IntExpr,variables=('x',),*,
                cases=256,solver=True,solver_timeout_ms=1000):
        if solver_timeout_ms<1:raise ValueError('solver timeout must be positive')
        # Sound algebraic equality is checked before finite testing.
        try:
            if reference.polynomial()==candidate.polynomial():
                return FalsificationResult('proved',proof_kind='integer_polynomial_normal_form')
        except ValueError:pass
        checked=0
        for binding in self.cases(variables,count=cases):
            checked+=1
            expected,observed=outcome(reference,binding),outcome(candidate,binding)
            if expected!=observed:return FalsificationResult('falsified',binding,expected,observed,checked)
        if solver:
            result=self._smt(reference,candidate,variables,solver_timeout_ms)
            if result is not None:
                result.witness_iterations=checked
                return result
        return FalsificationResult('tested_no_counterexample',witness_iterations=checked)

    @staticmethod
    def _smt(reference,candidate,variables,timeout):
        try:import z3
        except ImportError:return None
        symbols={name:z3.Int(name) for name in variables}
        def encode(expression):
            if expression.op=='var':return symbols[expression.value],z3.BoolVal(True)
            if expression.op=='const':return z3.IntVal(expression.value),z3.BoolVal(True)
            (a,da),(b,db)=(encode(child) for child in expression.args)
            defined=z3.And(da,db)
            if expression.op=='add':value=a+b
            elif expression.op=='sub':value=a-b
            elif expression.op=='mul':value=a*b
            elif expression.op=='min':value=z3.If(a<b,a,b)
            elif expression.op=='max':value=z3.If(a>b,a,b)
            else:
                defined=z3.And(defined,b!=0)
                # Z3 division with a positive denominator is Python floor div.
                value=z3.If(b>0,a/b,(-a)/(-b))
            return value,defined
        a,da=encode(reference);b,db=encode(candidate)
        solver=z3.Solver();solver.set(timeout=timeout)
        solver.add(z3.Or(da!=db,z3.And(da,db,a!=b)))
        status=solver.check()
        if status==z3.unsat:return FalsificationResult('proved',proof_kind='z3_integer_definedness_equivalence')
        if status==z3.sat:
            model=solver.model();case={name:model.eval(value,model_completion=True).as_long() for name,value in symbols.items()}
            return FalsificationResult('falsified',case,outcome(reference,case),outcome(candidate,case),proof_kind='z3_counterexample')
        return FalsificationResult('unknown',proof_kind='z3:'+solver.reason_unknown())

    @staticmethod
    def calculate_mdl_reward(original_code,candidate_code,speedup_ratio,verified):
        """A description-length heuristic, never Kolmogorov complexity or proof."""
        if not verified:return -100.
        if not math.isfinite(speedup_ratio):raise ValueError('speedup must be finite')
        old=len(list(ast.walk(ast.parse(original_code))));new=len(list(ast.walk(ast.parse(candidate_code))))
        compression=math.log2(max(1,old)/max(1,new))
        return .8*max(-1.,min(1.,speedup_ratio))+.2*max(-1.,min(1.,compression))
