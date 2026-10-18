"""Budgeted Python-subset interpreter, optional compiler diagnostics, proof adapters.

The subset (integer variables, if, while, arithmetic) is computationally
universal IN PRINCIPLE; the implementation intentionally enforces fuel, integer
size, loop and AST limits. It is not a security boundary for hostile native
compilers or a universal bug-freedom proof. No eval/exec of submitted code.
"""
from __future__ import annotations
import ast
from dataclasses import dataclass, asdict
from pathlib import Path
import operator
import shutil
import subprocess
import tempfile
import time
from typing import Any, Callable


class BudgetExceeded(RuntimeError):
    pass

class ProgramRejected(ValueError):
    pass

@dataclass(frozen=True)
class ExecutionBudget:
    fuel: int = 20_000
    max_bits: int = 4096
    max_ast_nodes: int = 800
    max_source_chars: int = 15_000
    deadline_seconds: float = 2.0

@dataclass
class ExecutionResult:
    result: int | bool | None
    steps: int
    elapsed_seconds: float
    completed: bool
    error: str | None = None

BIN_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
}
CMP_OPS = {ast.Eq:operator.eq,ast.NotEq:operator.ne,ast.Lt:operator.lt,
           ast.LtE:operator.le,ast.Gt:operator.gt,ast.GtE:operator.ge}


class _ReturnSignal(Exception):
    def __init__(self, value): self.value = value


class BudgetedPython:
    """Interpret a restricted Python function `def main(x): ...`.

    All integer computations are exact. Loops are legal, but fuel is bounded.
    No imports, files, attributes, subprocess, native calls, introspection,
    arbitrary Python objects, eval, exec or dynamic code compilation.
    """
    def __init__(self, budget: ExecutionBudget | None = None):
        self.budget = budget or ExecutionBudget()
        self.steps = 0
        self.start = 0.0

    def _tick(self):
        self.steps += 1
        if self.steps > self.budget.fuel:
            raise BudgetExceeded('instruction fuel exhausted')
        if self.steps % 64 == 0 and time.monotonic()-self.start > self.budget.deadline_seconds:
            raise BudgetExceeded('wall-clock budget exhausted')

    def _check(self, x: Any):
        if type(x) not in (int,bool): raise ProgramRejected('only integer/bool values')
        if type(x) is int and x.bit_length() > self.budget.max_bits:
            raise BudgetExceeded('integer bit budget exceeded')
        return x

    @staticmethod
    def _syntax(source: str, budget: ExecutionBudget) -> ast.FunctionDef:
        if len(source) > budget.max_source_chars: raise ProgramRejected('source too long')
        try: tree = ast.parse(source)
        except SyntaxError as ex: raise ProgramRejected(str(ex)) from ex
        if sum(1 for _ in ast.walk(tree)) > budget.max_ast_nodes:
            raise ProgramRejected('AST node limit')
        if len(tree.body)!=1 or not isinstance(tree.body[0],ast.FunctionDef):
            raise ProgramRejected('one function required: def main(x)')
        fn=tree.body[0]
        if fn.name!='main' or len(fn.args.args)!=1 or fn.args.args[0].arg!='x' or fn.decorator_list or fn.args.defaults or fn.args.kwonlyargs:
            raise ProgramRejected('signature must be def main(x)')
        allowed=(ast.Module,ast.FunctionDef,ast.arguments,ast.arg,ast.Assign,
                 ast.AugAssign,ast.If,ast.While,ast.Return,ast.Pass,ast.Name,ast.Load,ast.Store,ast.Constant,
                 ast.BinOp,ast.UnaryOp,ast.BoolOp,ast.Compare,ast.IfExp,
                 ast.Add,ast.Sub,ast.Mult,ast.FloorDiv,ast.Mod,ast.UAdd,
                 ast.USub,ast.Not,ast.And,ast.Or,ast.Eq,ast.NotEq,ast.Lt,
                 ast.LtE,ast.Gt,ast.GtE)
        for node in ast.walk(fn):
            if not isinstance(node,allowed):
                raise ProgramRejected(f'unsupported construct: {type(node).__name__}')
            if isinstance(node,ast.Constant) and type(node.value) not in (int,bool):
                raise ProgramRejected('only integer/bool constants')
        return fn

    def _expr(self,node,env):
        self._tick()
        if isinstance(node,ast.Constant): return self._check(node.value)
        if isinstance(node,ast.Name):
            if node.id not in env: raise ProgramRejected(f'undefined variable {node.id}')
            return env[node.id]
        if isinstance(node,ast.BinOp):
            op=BIN_OPS.get(type(node.op))
            if op is None: raise ProgramRejected('unsupported arithmetic')
            a,b=self._expr(node.left,env),self._expr(node.right,env)
            if isinstance(node.op,(ast.FloorDiv,ast.Mod)) and b==0: raise ProgramRejected('division by zero')
            return self._check(op(a,b))
        if isinstance(node,ast.UnaryOp):
            val=self._expr(node.operand,env)
            if isinstance(node.op,ast.USub): return self._check(-val)
            if isinstance(node.op,ast.UAdd): return self._check(+val)
            if isinstance(node.op,ast.Not): return not val
        if isinstance(node,ast.IfExp):
            return self._expr(node.body if self._expr(node.test,env) else node.orelse,env)
        if isinstance(node,ast.BoolOp):
            if isinstance(node.op,ast.And):
                return all(self._expr(v,env) for v in node.values)
            if isinstance(node.op,ast.Or):
                return any(self._expr(v,env) for v in node.values)
        if isinstance(node,ast.Compare):
            left=self._expr(node.left,env)
            for op,right_node in zip(node.ops,node.comparators):
                right=self._expr(right_node,env)
                fn=CMP_OPS.get(type(op))
                if fn is None: raise ProgramRejected('unsupported comparison')
                if not fn(left,right): return False
                left=right
            return True
        raise ProgramRejected(f'bad expression: {type(node).__name__}')

    def _block(self,statements,env):
        for stmt in statements:
            self._tick()
            if isinstance(stmt,(ast.Assign,ast.AugAssign)):
                if isinstance(stmt,ast.Assign):
                    if len(stmt.targets)!=1 or not isinstance(stmt.targets[0],ast.Name):
                        raise ProgramRejected('assignment must target a variable')
                    env[stmt.targets[0].id]=self._expr(stmt.value,env)
                else:
                    if not isinstance(stmt.target,ast.Name):raise ProgramRejected('bad augmented assignment')
                    name=stmt.target.id
                    op=BIN_OPS.get(type(stmt.op))
                    if op is None or name not in env:raise ProgramRejected('bad augmented assignment')
                    rhs=self._expr(stmt.value,env)
                    if isinstance(stmt.op,(ast.Mod,ast.FloorDiv)) and rhs==0: raise ProgramRejected('division by zero')
                    env[name]=self._check(op(env[name],rhs))
            elif isinstance(stmt,ast.Return):
                raise _ReturnSignal(self._expr(stmt.value,env) if stmt.value else None)
            elif isinstance(stmt,ast.If):
                self._block(stmt.body if self._expr(stmt.test,env) else stmt.orelse,env)
            elif isinstance(stmt,ast.While):
                while self._expr(stmt.test,env):
                    self._tick()
                    self._block(stmt.body,env)
                if stmt.orelse: self._block(stmt.orelse,env)
            elif isinstance(stmt,ast.Pass):pass
            else: raise ProgramRejected(f'unsupported statement {type(stmt).__name__}')

    def run(self,source:str,x:int) -> ExecutionResult:
        self.steps=0;self.start=time.monotonic()
        try:
            fn=self._syntax(source,self.budget)
            env={'x':self._check(x)}
            try:
                self._block(fn.body,env)
                return ExecutionResult(None,self.steps,time.monotonic()-self.start,False,'missing return')
            except _ReturnSignal as signal:
                return ExecutionResult(signal.value,self.steps,time.monotonic()-self.start,True)
        except (ProgramRejected,BudgetExceeded,ZeroDivisionError,OverflowError) as ex:
            return ExecutionResult(None,self.steps,time.monotonic()-self.start,False,str(ex))


@dataclass
class ProofResult:
    status: str # proved, counterexample, bounded_pass, unknown
    method: str
    details: str
    checked_cases: int = 0
    witness: int | None = None


def bounded_equivalence(source_a:str,source_b:str,lo=-100,hi=100,
                        budget:ExecutionBudget|None=None)->ProofResult:
    """Exhaustive *finite-domain* verification, not general equivalence proof."""
    if hi<lo or hi-lo>100_000: raise ValueError('invalid finite domain')
    interpreter=BudgetedPython(budget)
    for i,x in enumerate(range(lo,hi+1),1):
        a=interpreter.run(source_a,x)
        b=interpreter.run(source_b,x)
        if not a.completed or not b.completed:
            return ProofResult('unknown','bounded_exhaustive','program failed or exceeded fuel',i,x)
        if a.result != b.result:
            return ProofResult('counterexample','bounded_exhaustive',f'outputs {a.result!r} and {b.result!r}',i,x)
    return ProofResult('bounded_pass','bounded_exhaustive',f'equivalent only for integers in [{lo},{hi}]',hi-lo+1)


def _parse_polynomial_expression(expression: str):
    """Translate only a tiny trusted AST to SymPy, never sympify arbitrary text."""
    import sympy as sp
    if len(expression)>300:raise ValueError('symbolic expression too long')
    tree=ast.parse(expression,mode='eval')
    if sum(1 for _ in ast.walk(tree))>60:raise ValueError('symbolic AST too large')
    x=sp.Symbol('x',integer=True)
    def parse(node):
        if isinstance(node,ast.Constant) and type(node.value) is int and abs(node.value)<10**9:
            return sp.Integer(node.value)
        if isinstance(node,ast.Name) and node.id=='x':return x
        if isinstance(node,ast.UnaryOp):
            val=parse(node.operand)
            if isinstance(node.op,ast.USub):return -val
            if isinstance(node.op,ast.UAdd):return val
        if isinstance(node,ast.BinOp):
            a,b=parse(node.left),parse(node.right)
            if isinstance(node.op,ast.Add):return a+b
            if isinstance(node.op,ast.Sub):return a-b
            if isinstance(node.op,ast.Mult):return a*b
            if isinstance(node.op,ast.Pow) and isinstance(node.right,ast.Constant):
                exponent=node.right.value
                if type(exponent) is int and 0<=exponent<=12:return a**exponent
        raise ValueError(f'unsupported symbolic expression: {type(node).__name__}')
    return parse(tree.body),x


def prove_polynomial_identity(left:str,right:str)->ProofResult:
    """Symbolic polynomial equivalence with strictly parsed exact expressions."""
    import sympy as sp
    try:
        a,x=_parse_polynomial_expression(left)
        b,_=_parse_polynomial_expression(right)
        if not a.is_polynomial(x) or not b.is_polynomial(x):
            return ProofResult('unknown','sympy-polynomial','non-polynomial expression')
        diff=sp.Poly(sp.cancel(a-b),x)
        if diff.is_zero:
            return ProofResult('proved','symbolic_polynomial','polynomial identity over integers')
        return ProofResult('counterexample','symbolic_polynomial',f'nonzero difference: {diff.as_expr()}')
    except (ValueError,TypeError,SyntaxError,sp.SympifyError) as ex:
        return ProofResult('unknown','sympy-polynomial',str(ex))


def optional_proof_assistant_check(assistant:str,proof_source:str,
                                    allow_external:bool=False,timeout:float=5.0)->dict:
    """Lean/Coq kernel adapter. It reports proof only when tool REALLY succeeds.

    Runs external theorem prover only on user-trusted source with opt-in. The
    host operating system must isolate untrusted source/processes separately.
    """
    if assistant not in ('lean','coq'):raise ValueError('assistant must be lean or coq')
    if not allow_external:
        return {'proved':False,'available':False,'reason':'external prover requires opt-in'}
    if not 0<len(proof_source)<=30_000:raise ValueError('invalid source size')
    tool='lean' if assistant=='lean' else 'coqc'
    executable=shutil.which(tool)
    if not executable:return {'proved':False,'available':False,'reason':f'{tool} not installed'}
    with tempfile.TemporaryDirectory(prefix='nexus_proof_') as tmp:
        target=Path(tmp)/('Proof.lean' if assistant=='lean' else 'Proof.v')
        target.write_text(proof_source,encoding='utf-8')
        args=[executable,str(target)] if assistant=='lean' else [executable,'-q',str(target)]
        try:
            proc=subprocess.run(args,cwd=tmp,timeout=timeout,text=True,capture_output=True)
            return {'proved':proc.returncode==0,'available':True,'returncode':proc.returncode,
                    'diagnostics':(proc.stderr or proc.stdout)[-2000:]}
        except subprocess.TimeoutExpired:
            return {'proved':False,'available':True,'reason':'theorem prover timeout'}


COMPILER_COMMANDS={
    'python': None,
    'cpp': ('clang++','-std=c++17','-fsyntax-only','-x','c++'),
    'rust': ('rustc','--edition=2021','--emit=metadata'),
    'assembly': ('clang','-c','-x','assembler'),
}

def optional_compiler_check(language:str,source:str,allow_native_compiler:bool=False,
                            timeout:float=3.0)->dict:
    """Optional compilation diagnostics; compilation is NOT a formal proof.

    Native compilers process attacker-controlled text and are not a sandbox.
    This is OFF unless explicitly opted in on a trusted local machine.
    No compiled binary is executed.
    """
    if language not in COMPILER_COMMANDS:raise ValueError('language must be python/cpp/rust/assembly')
    if language=='python':
        try: ast.parse(source);return {'available':True,'syntax_ok':True,'proof':False}
        except SyntaxError as ex:return {'available':True,'syntax_ok':False,'error':str(ex),'proof':False}
    if not allow_native_compiler:
        return {'available':False,'reason':'native compiler opt-in required','proof':False}
    command=COMPILER_COMMANDS[language]
    executable=shutil.which(command[0])
    if not executable:return {'available':False,'reason':'compiler not installed','proof':False}
    if len(source)>100_000: raise ValueError('native source size exceeded')
    with tempfile.TemporaryDirectory(prefix='nexus_compile_') as temp:
        ext={'cpp':'.cpp','rust':'.rs','assembly':'.s'}[language]
        target=Path(temp)/('candidate'+ext)
        target.write_text(source,encoding='utf-8')
        args=[executable,*command[1:],str(target)]
        if language=='rust':args.extend(['-o',str(Path(temp)/'candidate.rmeta')])
        if language=='assembly':args.extend(['-o',str(Path(temp)/'candidate.o')])
        try:
            cp=subprocess.run(args,cwd=temp,capture_output=True,text=True,timeout=timeout,check=False)
            return {'available':True,'syntax_ok':cp.returncode==0,'returncode':cp.returncode,
                    'diagnostics':(cp.stderr or cp.stdout)[-1500:],'proof':False}
        except subprocess.TimeoutExpired:
            return {'available':True,'syntax_ok':False,'reason':'compiler timeout','proof':False}
