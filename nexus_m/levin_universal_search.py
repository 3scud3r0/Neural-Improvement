"""Finite, prefix-code-weighted dovetailing for a typed integer language.

Uses an explicit computable description length, NOT the uncomputable K(p).
This bounded implementation is neither a universal machine nor a solution to
the halting problem. Each candidate interpreter is resumed under actual fuel.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import ast
import math
import time
from .typed_program import IntExpr, enumerate_typed
from .oracle_coevolution import outcome


def _gamma(value):
    bits=bin(value)[2:]
    return '0'*(len(bits)-1)+bits


def prefix_code(expression,variables=('x',)):
    codes={'add':'1011','mul':'1100','sub':'1101','floordiv':'11100','min':'11101','max':'11110'}
    if expression.op=='var':
        index=variables.index(expression.value)
        return '00' if index==0 else '111110'+_gamma(index)
    if expression.op=='const':
        code={0:'010',1:'011',2:'100',-1:'1010'}.get(expression.value)
        if code:return code
        zigzag=2*expression.value if expression.value>=0 else -2*expression.value-1
        return '111111'+_gamma(zigzag+1)
    return codes[expression.op]+''.join(prefix_code(a,variables) for a in expression.args)


@dataclass
class LevinSolution:
    expression: IntExpr
    fuel_spent: int
    programs_generated: int
    description_bits: int
    rounds: int
    backend: str = 'prefix_weighted_scalar'
    variables: tuple[str,...] = ('x',)
    @property
    def program_ast(self):return self.expression.to_ast()
    @property
    def solution_code(self):return 'def solution('+', '.join(self.variables)+'):\n    return '+self.expression.source()+'\n'


@dataclass
class SearchStatistics:
    fuel_spent:int=0
    programs_generated:int=0
    rounds:int=0
    stop_reason:str='not_started'
    allocations:list[dict]=field(default_factory=list)
    backend:str='prefix_weighted_scalar'
    vector_instruction_slots:int=0


class LevinUniversalSearch:
    def __init__(self,max_depth=7,*,variables=('x',),operations=('add','sub','mul'),
                 max_programs=4096,max_integer_bits=256):
        if min(max_depth,max_programs,max_integer_bits)<1:raise ValueError('positive budgets required')
        self.max_depth=max_depth;self.variables=tuple(variables);self.operations=operations
        self.max_programs=max_programs;self.max_integer_bits=max_integer_bits
        self.last_statistics=SearchStatistics()

    def _generate_candidate_asts(self,max_nodes):
        for expression in enumerate_typed(self.variables,operations=self.operations,
                max_nodes=max_nodes,max_programs=self.max_programs):yield expression.to_ast()

    def _verify_steps(self,expression,examples):
        for binding,expected in examples:
            try:observed=('value',(yield from expression.steps(binding,max_bits=self.max_integer_bits)))
            except (ZeroDivisionError,OverflowError) as exc:observed=('error',type(exc).__name__)
            yield None  # comparison itself is charged too
            if observed!=expected:return False
        return True

    def search(self,examples,*,max_fuel=50_000,wall_seconds=10.):
        examples=list(examples)
        if max_fuel<1 or not math.isfinite(wall_seconds) or wall_seconds<=0 or not examples:raise ValueError('examples and positive budgets required')
        cases=[]
        for binding,expected in examples:
            if set(binding)!=set(self.variables) or any(type(v) is not int for v in binding.values()):
                raise ValueError('examples must bind every integer variable')
            expected=expected if isinstance(expected,tuple) else ('value',expected)
            if len(expected)!=2 or expected[0] not in ('value','error') or (expected[0]=='value' and type(expected[1]) is not int):
                raise ValueError('expected result must be an integer or explicit error outcome')
            cases.append((binding,expected))
        stats=self.last_statistics=SearchStatistics(stop_reason='fuel_budget')
        generator=iter(enumerate_typed(self.variables,operations=self.operations,
            max_nodes=self.max_depth,max_programs=self.max_programs))
        tasks=[];exhausted=False;started=time.monotonic()
        for phase in range(1,1025):
            stats.rounds=phase
            if time.monotonic()-started>wall_seconds:stats.stop_reason='wall_budget';return None
            admit=min(self.max_programs,1 << min(phase//2,30))
            while len(tasks)<admit and not exhausted:
                try:expression=next(generator)
                except StopIteration:exhausted=True;break
                tasks.append(dict(expression=expression,bits=len(prefix_code(expression,self.variables)),
                                  iterator=self._verify_steps(expression,cases),finished=False,spent=0))
                stats.programs_generated+=1
            if len(tasks)>=self.max_programs:exhausted=True
            work=0
            for task in sorted(tasks,key=lambda item:item['bits']):
                if task['finished'] or task['bits']>phase:continue
                allocation=min(max_fuel-stats.fuel_spent,1 << min(phase-task['bits'],62))
                for _ in range(allocation):
                    if time.monotonic()-started>wall_seconds:stats.stop_reason='wall_budget';return None
                    stats.fuel_spent+=1;task['spent']+=1;work+=1
                    try:next(task['iterator'])
                    except StopIteration as completed:
                        task['finished']=True
                        if completed.value:
                            stats.stop_reason='solution'
                            return LevinSolution(task['expression'],stats.fuel_spent,
                                stats.programs_generated,task['bits'],phase,variables=self.variables)
                        break
                if stats.fuel_spent>=max_fuel:return None
            stats.allocations.append(dict(round=phase,verification_steps=work,admitted=len(tasks)))
            if exhausted and all(t['finished'] for t in tasks):stats.stop_reason='search_space_exhausted';return None
        stats.stop_reason='round_budget';return None

    def search_batched(self,examples,*,device='auto',batch_size=256,max_programs=None,max_instruction_slots=1_000_000):
        """Typed enumeration + vectorized finite prefilter, NOT weighted Lsearch.

        Only one int64 input is supported here. Survivors are checked again by
        the exact scalar interpreter; no equivalence proof is implied.
        """
        from .gpu_program_evaluator import GPUBatchEvaluator
        import torch
        examples=list(examples)
        if len(self.variables)!=1 or not examples:raise ValueError('vector prefilter needs one variable and examples')
        stats=self.last_statistics=SearchStatistics(stop_reason='search_space_exhausted',backend='vectorized_prefilter')
        if batch_size<1 or max_instruction_slots<1:raise ValueError('positive batch/work budgets required')
        limit=self.max_programs if max_programs is None else min(self.max_programs,max_programs)
        if limit<1:raise ValueError('positive program budget required')
        vm=GPUBatchEvaluator(device=device)
        values=[];expected=[]
        for binding,result in examples:
            value=binding[self.variables[0]]
            expected_value=result[1] if isinstance(result,tuple) and result[0]=='value' else result
            if type(value) is not int or type(expected_value) is not int or not -(2**63)<=value<2**63 or not -(2**63)<=expected_value<2**63:
                raise ValueError('vector examples must be defined int64 values')
            values.append(value);expected.append(expected_value)
        inputs=torch.tensor(values,dtype=torch.int64,device=vm.device)
        outputs=torch.tensor(expected,dtype=torch.int64,device=vm.device)
        pending=[];generated=0
        generator=enumerate_typed(self.variables,operations=self.operations,max_nodes=self.max_depth,max_programs=limit)
        def check(batch):
            matrix=vm.compile_bytecode_matrix([e.postfix(self.variables[0]) for e in batch])
            evaluated=vm.evaluate_batch(matrix,outputs,inputs)
            stats.vector_instruction_slots+=evaluated.instruction_slots
            for index in evaluated.surviving_program_indices.cpu().tolist():
                expression=batch[index]
                if all(outcome(expression,binding)==('value',target) for binding,target in zip(
                        [b for b,_ in examples],expected)):
                    stats.stop_reason='solution'
                    return LevinSolution(expression,0,generated,len(prefix_code(expression,self.variables)),0,'vectorized_prefilter',self.variables)
            return None
        for expression in generator:
            trial_slots=(len(pending)+1)*max([e.nodes for e in pending]+[expression.nodes])*len(examples)
            if stats.vector_instruction_slots+trial_slots>max_instruction_slots:
                result=check(pending) if pending else None
                if result is None:stats.stop_reason='vector_instruction_budget'
                return result
            pending.append(expression);generated+=1
            stats.programs_generated=generated
            if len(pending)>=batch_size:
                solution=check(pending)
                if solution:return solution
                pending=[]
        return check(pending) if pending else None
