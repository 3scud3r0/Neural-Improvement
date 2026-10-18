"""Evidence gate for synthesized router programs.

Provides a *construction/termination argument* and exhaustive equivalence on a
finite domain. This is NOT a machine-checked Lean/Coq proof of arbitrary Python
or a general guarantee of memory safety. Fail closed for unsupported templates.
"""
from __future__ import annotations
from dataclasses import dataclass,asdict
import heapq,itertools
from .ast_program_synthesis import RouterSpec,compile_source,validate_generated_source,reference_route

@dataclass
class VerificationEvidence:
    canonical_ast:bool
    bounded_loops:bool
    finite_domain_equivalent:bool
    checked_cases:int
    checker_name:str
    machine_checked_universal_proof:bool
    statement:str
    def as_dict(self):return asdict(self)


def verify(spec:RouterSpec,values=(-2.0,-1.0,0.0,1.0),max_n:int=4)->VerificationEvidence:
    if not 0<=max_n<=6: raise ValueError('max_n must be in 0..6')
    source=compile_source(spec)
    canonical=validate_generated_source(source)==spec
    bounded=spec.validation=='strict' and spec.strategy in ('sort','heap','scan')
    # This exec is exclusively over canonical program AST produced by our DSL.
    env={'__builtins__':{'range':range,'len':len,'sorted':sorted,'min':min,
                          'list':list,'ValueError':ValueError},'heapq':heapq}
    exec(compile(source,'<canonical_router_ast>','exec'),env)
    fn=env['route']
    count=0;passed=True
    for n in range(max_n+1):
        for scores in itertools.product(values,repeat=n):
            for k in range(-1,n+2):
                count+=1
                a=list(scores)
                try: expected=('ok',reference_route(a,k))
                except Exception as exc:expected=('error',type(exc).__name__)
                try: observed=('ok',fn(a,k))
                except Exception as exc:observed=('error',type(exc).__name__)
                if expected!=observed:
                    passed=False;break
            if not passed:break
        if not passed:break
    return VerificationEvidence(canonical,bounded,passed,count,
          'finite_exhaustive_and_structural_reasoning',False,
          'Exhaustive over finite inputs only. Structural termination follows from '
          'bounded list primitives or a finite range(k) under the strict k guard; '
          'no universal formal theorem proof has been generated.')


def promotion_eligible(proof:VerificationEvidence)->bool:
    return proof.canonical_ast and proof.bounded_loops and proof.finite_domain_equivalent
