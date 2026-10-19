"""Small typed integer IR shared by synthesis, proofs and the vector VM.

This is deliberately an explicit mathematical domain, not arbitrary Python,
floating point, tensor algebra or a claim of Turing completeness.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from itertools import product
from typing import Iterator, Mapping

OPS = ('add', 'sub', 'mul', 'floordiv', 'min', 'max')
AST_OPS = {ast.Add:'add', ast.Sub:'sub', ast.Mult:'mul', ast.FloorDiv:'floordiv'}


@dataclass(frozen=True)
class IntExpr:
    op: str
    args: tuple['IntExpr', ...] = ()
    value: int | str | None = None

    def __post_init__(self):
        if self.op == 'const':
            valid = type(self.value) is int and not self.args
        elif self.op == 'var':
            valid = isinstance(self.value, str) and self.value.isidentifier() and not self.args
        else:
            valid = self.op in OPS and len(self.args) == 2 and self.value is None
        if not valid or not all(isinstance(a, IntExpr) for a in self.args):
            raise ValueError('invalid typed integer expression')

    @property
    def nodes(self): return 1 + sum(a.nodes for a in self.args)

    @classmethod
    def from_ast(cls, node: ast.AST, variables=('x',), *, builtin_calls=False):
        if isinstance(node, ast.Name) and node.id in variables:
            return cls('var', value=node.id)
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return cls('const', value=node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            operand = cls.from_ast(node.operand, variables, builtin_calls=builtin_calls)
            return cls('const', value=-operand.value) if operand.op == 'const' else cls(
                'sub', (cls('const', value=0), operand))
        if isinstance(node, ast.BinOp) and type(node.op) in AST_OPS:
            return cls(AST_OPS[type(node.op)], tuple(cls.from_ast(a, variables,
                builtin_calls=builtin_calls) for a in (node.left, node.right)))
        if (builtin_calls and isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in ('min', 'max') and len(node.args) == 2 and not node.keywords):
            return cls(node.func.id, tuple(cls.from_ast(a, variables, builtin_calls=True) for a in node.args))
        raise ValueError(f'unsupported integer IR node: {type(node).__name__}')

    def to_ast(self):
        if self.op == 'var': return ast.Name(id=self.value, ctx=ast.Load())
        if self.op == 'const': return ast.Constant(value=self.value)
        children = [a.to_ast() for a in self.args]
        if self.op in ('min', 'max'):
            return ast.Call(func=ast.Name(id=self.op, ctx=ast.Load()), args=children, keywords=[])
        operator = {'add':ast.Add, 'sub':ast.Sub, 'mul':ast.Mult, 'floordiv':ast.FloorDiv}[self.op]
        return ast.BinOp(left=children[0], op=operator(), right=children[1])

    def source(self): return ast.unparse(ast.fix_missing_locations(self.to_ast()))

    def steps(self, bindings: Mapping[str, int], *, max_bits=4096) -> Iterator[int]:
        """Resumable interpreter: each yielded value consumes one node of fuel."""
        if self.op == 'const': result = self.value
        elif self.op == 'var':
            result = bindings[self.value]
            if type(result) is not int: raise TypeError('IR inputs must be Python ints')
        else:
            values = []
            for child in self.args:
                values.append((yield from child.steps(bindings, max_bits=max_bits)))
            a, b = values
            if self.op == 'add': result = a+b
            elif self.op == 'sub': result = a-b
            elif self.op == 'mul': result = a*b
            elif self.op == 'floordiv': result = a//b
            elif self.op == 'min': result = min(a,b)
            else: result = max(a,b)
        if abs(result).bit_length() > max_bits: raise OverflowError('IR integer bit budget exceeded')
        yield result
        return result

    def evaluate(self, bindings, *, max_bits=4096, fuel=100_000):
        if fuel<self.nodes:raise RuntimeError('IR fuel exhausted')
        iterator = self.steps(bindings, max_bits=max_bits)
        while True:
            try:next(iterator)
            except StopIteration as result:return result.value

    def polynomial(self, *, max_terms=4096, max_degree=32):
        """Exact normal form for Z[variables], independent of testing samples."""
        if self.op == 'var': return {(self.value,):1}
        if self.op == 'const': return {():self.value} if self.value else {}
        if self.op not in ('add','sub','mul'): raise ValueError('not an integer polynomial')
        a, b = (child.polynomial(max_terms=max_terms, max_degree=max_degree) for child in self.args)
        out = dict(a) if self.op != 'mul' else {}
        if self.op == 'mul':
            if len(a)*len(b) > max_terms*4: raise ValueError('polynomial expansion budget exceeded')
            items = ((tuple(sorted(x+y)), u*v) for x,u in a.items() for y,v in b.items())
        else:
            items = ((x, v if self.op == 'add' else -v) for x,v in b.items())
        for monomial, coefficient in items:
            if len(monomial) > max_degree: raise ValueError('polynomial degree budget exceeded')
            out[monomial] = out.get(monomial,0)+coefficient
            if not out[monomial]: del out[monomial]
            if len(out) > max_terms: raise ValueError('polynomial term budget exceeded')
        return out

    def postfix(self, variable='x'):
        if self.op == 'var':
            if self.value != variable: raise ValueError('tensor VM supports one integer input variable')
            return ['X']
        if self.op == 'const':
            token = {0:'C0',1:'C1',2:'C2',-1:'C-1'}.get(self.value)
            if token is None: raise ValueError('constant not in tensor VM instruction set')
            return [token]
        return self.args[0].postfix(variable)+self.args[1].postfix(variable)+[{
            'add':'ADD','sub':'SUB','mul':'MUL','floordiv':'DIV','min':'MIN','max':'MAX'}[self.op]]


def typed_function(function: ast.FunctionDef):
    if not isinstance(function, ast.FunctionDef) or function.decorator_list:
        raise ValueError('automatic integer synthesis requires an undecorated synchronous function')
    args = function.args
    if args.vararg or args.kwarg or args.kwonlyargs or args.defaults or args.posonlyargs:
        raise ValueError('automatic integer synthesis requires simple positional int parameters')
    if not args.args or any(not isinstance(a.annotation, ast.Name) or a.annotation.id != 'int' for a in args.args):
        raise ValueError('explicit int parameter annotations are required for algebraic rewrites')
    body = function.body
    if body and isinstance(body[0],ast.Expr) and isinstance(body[0].value,ast.Constant) and isinstance(body[0].value.value,str):
        body = body[1:]
    if len(body)!=1 or not isinstance(body[0],ast.Return):
        raise ValueError('automatic IR currently supports one pure integer return expression')
    variables = tuple(a.arg for a in args.args)
    return variables, IntExpr.from_ast(body[0].value, variables)


def enumerate_typed(variables=('x',), constants=(0,1,2,-1), operations=('add','sub','mul'),
                    *, max_nodes=7, max_programs=10_000):
    """Generate well-typed terms by tree size, with explicit finite budgets."""
    if max_nodes<1 or max_programs<1 or not variables: raise ValueError('positive synthesis budgets required')
    if not set(operations)<=set(OPS): raise ValueError('unknown primitive')
    levels = {1:[IntExpr('var',value=v) for v in variables]+[IntExpr('const',value=c) for c in constants]}
    count=0
    for size in range(1,max_nodes+1):
        if size>1:
            levels[size]=[]
            for left in range(1,size-1):
                right=size-1-left
                for op,a,b in product(operations,levels.get(left,()),levels.get(right,())):
                    term=IntExpr(op,(a,b))
                    levels[size].append(term)
                    count+=1
                    yield term
                    if count>=max_programs:return
        else:
            for term in levels[1]:
                count+=1
                yield term
                if count>=max_programs:return
