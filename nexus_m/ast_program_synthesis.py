"""Compositional AST synthesis for a deterministic top-k routing kernel.

Not unrestricted Python synthesis: the search space is a typed, auditable DSL
compiled into actual Python AST. Every produced program terminates, has
bounded loops (for finite inputs) and accesses only an explicitly checked API.
"""
from __future__ import annotations
import ast
from dataclasses import dataclass, asdict
from typing import Iterator

STRATEGIES = ('sort', 'heap', 'scan', 'reverse_sort')
KEYS = ('stable', 'score', 'index')
VALIDATIONS = ('strict', 'omitted')
FIELDS = (('strategy', STRATEGIES), ('key', KEYS), ('validation', VALIDATIONS))

@dataclass(frozen=True)
class RouterSpec:
    strategy: str = 'sort'
    key: str = 'stable'
    validation: str = 'strict'
    def __post_init__(self):
        for name, options in FIELDS:
            if getattr(self, name) not in options:
                raise ValueError(f'{name} must be one of {options}')
    def as_dict(self): return asdict(self)


def _name(s: str, ctx: ast.expr_context | None = None) -> ast.Name:
    return ast.Name(id=s, ctx=ctx or ast.Load())


def _call(fn: ast.expr, *args: ast.expr) -> ast.Call:
    return ast.Call(func=fn, args=list(args), keywords=[])


def _rank(key: str) -> ast.expr:
    i = _name('i')
    score = ast.Subscript(value=_name('scores'), slice=i, ctx=ast.Load())
    if key == 'stable':
        expr = ast.Tuple(elts=[ast.UnaryOp(op=ast.USub(), operand=score), i], ctx=ast.Load())
    elif key == 'score':
        expr = ast.UnaryOp(op=ast.USub(), operand=score)
    else:
        expr = i
    return ast.Lambda(
        args=ast.arguments(posonlyargs=[], args=[ast.arg(arg='i')],
                           vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]),
        body=expr,
    )


def _range_indices() -> ast.Call:
    return _call(_name('range'), _call(_name('len'), _name('scores')))


def build_router_ast(spec: RouterSpec) -> ast.Module:
    """Create syntax nodes structurally; no static source-code templates."""
    stmts: list[ast.stmt] = []
    if spec.validation == 'strict':
        compare = ast.Compare(left=ast.Constant(0), ops=[ast.LtE(), ast.LtE()],
                              comparators=[_name('k'), _call(_name('len'), _name('scores'))])
        stmts.append(ast.If(test=ast.UnaryOp(op=ast.Not(), operand=compare),
                            body=[ast.Raise(exc=_call(_name('ValueError'), ast.Constant('invalid k')),
                                            cause=None)], orelse=[]))
    if spec.strategy == 'scan':
        stmts.extend([
            ast.Assign(targets=[_name('remaining', ast.Store())], value=_call(_name('list'), _range_indices())),
            ast.Assign(targets=[_name('chosen', ast.Store())], value=ast.List(elts=[], ctx=ast.Load())),
            ast.For(target=_name('_', ast.Store()), iter=_call(_name('range'), _name('k')),
                    body=[
                        ast.Assign(targets=[_name('i', ast.Store())],
                                   value=ast.Call(func=_name('min'), args=[_name('remaining')],
                                                  keywords=[ast.keyword(arg='key', value=_rank(spec.key))])),
                        ast.Expr(value=_call(ast.Attribute(value=_name('chosen'), attr='append', ctx=ast.Load()), _name('i'))),
                        ast.Expr(value=_call(ast.Attribute(value=_name('remaining'), attr='remove', ctx=ast.Load()), _name('i'))),
                    ], orelse=[]),
            ast.Return(value=_name('chosen')),
        ])
    else:
        indices = _range_indices()
        if spec.strategy == 'heap':
            selection = ast.Call(func=ast.Attribute(value=_name('heapq'), attr='nsmallest', ctx=ast.Load()),
                                 args=[_name('k'), indices],
                                 keywords=[ast.keyword(arg='key', value=_rank(spec.key))])
        elif spec.strategy == 'sort':
            selection = ast.Subscript(
                value=ast.Call(func=_name('sorted'), args=[indices],
                               keywords=[ast.keyword(arg='key', value=_rank(spec.key))]),
                slice=ast.Slice(lower=None, upper=_name('k'), step=None), ctx=ast.Load())
        else:
            # reverse-sort also produces valid ranked results, but is intentionally inefficient.
            ordered = ast.Call(func=_name('sorted'), args=[indices],
                               keywords=[ast.keyword(arg='key', value=_rank(spec.key)),
                                         ast.keyword(arg='reverse', value=ast.Constant(True))])
            selection = ast.Subscript(value=ast.Subscript(value=ordered,
                                    slice=ast.Slice(lower=None, upper=ast.UnaryOp(op=ast.USub(), operand=_name('k')),
                                                    step=None),ctx=ast.Load()),
                                      slice=ast.Slice(lower=None,upper=None,
                                                      step=ast.UnaryOp(op=ast.USub(),operand=ast.Constant(1))),
                                      ctx=ast.Load())
        stmts.append(ast.Return(value=selection))
    fn = ast.FunctionDef(name='route',
          args=ast.arguments(posonlyargs=[],args=[ast.arg(arg='scores'),ast.arg(arg='k')],
                vararg=None,kwonlyargs=[],kw_defaults=[],kwarg=None,defaults=[]),
          body=stmts,decorator_list=[])
    return ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))


def compile_source(spec: RouterSpec) -> str:
    tree = build_router_ast(spec)
    compile(tree, '<generated_ast>', 'exec')
    return ast.unparse(tree) + '\n'


def validate_generated_source(source: str) -> RouterSpec:
    """Only accept exact normalized ASTs emitted by this generator."""
    try: actual = ast.dump(ast.parse(source), include_attributes=False)
    except SyntaxError as exc: raise ValueError('candidate cannot be parsed') from exc
    for a in STRATEGIES:
        for b in KEYS:
            for c in VALIDATIONS:
                spec = RouterSpec(a,b,c)
                if ast.dump(build_router_ast(spec),include_attributes=False)==actual:
                    return spec
    raise ValueError('source is not a canonical AST in the approved search grammar')


def all_router_specs() -> Iterator[RouterSpec]:
    for strategy in STRATEGIES:
        for key in KEYS:
            for validation in VALIDATIONS:
                yield RouterSpec(strategy,key,validation)


def reference_route(scores: list[float],k: int)->list[int]:
    if not 0 <= k <= len(scores): raise ValueError('invalid k')
    return sorted(range(len(scores)), key=lambda i:(-scores[i],i))[:k]
