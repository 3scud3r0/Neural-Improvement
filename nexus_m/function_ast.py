"""Repository function selection and extensible proposals, without a router DSL.

Only parse/compile candidate code here. Execution belongs in the copied-project
evaluator. Providers are trusted search plugins; candidate bodies are not loaded
into the search process.
"""
from __future__ import annotations

import ast
from copy import deepcopy
from dataclasses import dataclass
import importlib
from pathlib import Path
import random
import tokenize
from typing import Iterable, Protocol

Function = ast.FunctionDef | ast.AsyncFunctionDef


def digest(source: bytes) -> str:
    import hashlib
    return hashlib.sha256(source).hexdigest()


def scope_definitions(body):
    for node in body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node
        else:
            yield from scope_definitions(list(ast.iter_child_nodes(node)))


def function_names(tree: ast.Module, prefix=''):
    for node in scope_definitions(tree.body):
        name = prefix + node.name
        if not isinstance(node, ast.ClassDef):
            yield name
        yield from function_names(ast.Module(body=node.body, type_ignores=[]), name+'.')


def find_function(tree: ast.Module, qualified_name: str) -> Function:
    """Select module functions, methods or nested functions, rejecting ambiguity."""
    body = tree.body
    parts = qualified_name.split('.')
    for index, name in enumerate(parts):
        matches = [node for node in scope_definitions(body) if node.name == name]
        if len(matches) != 1:
            raise ValueError(f'function scope is missing or ambiguous: {qualified_name}')
        node = matches[0]
        if index == len(parts) - 1:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                raise ValueError('target must be a Python function or method')
            return node
        body = node.body
    raise ValueError('empty function name')


@dataclass
class FunctionTarget:
    workspace: Path
    relative_path: str
    qualified_name: str
    original: bytes
    encoding: str
    text: str
    function: Function

    @classmethod
    def read(cls, workspace: str | Path, relative_path: str, qualified_name: str):
        workspace = Path(workspace).resolve()
        path = (workspace / relative_path).resolve()
        relative = path.relative_to(workspace)
        if path.suffix != '.py' or not path.is_file():
            raise ValueError('target must be an existing Python source file in the workspace')
        # Remember the user's lexical path, so a later symlink change is detected.
        lexical = Path(relative_path)
        if lexical.is_absolute():
            lexical = relative
        raw = path.read_bytes()
        from io import BytesIO
        encoding, _ = tokenize.detect_encoding(BytesIO(raw).readline)
        text = raw.decode(encoding)
        tree = ast.parse(text, filename=str(path), type_comments=True)
        return cls(workspace, str(lexical), qualified_name, raw, encoding,
                   text, find_function(tree, qualified_name))

    @property
    def path(self):
        return self.workspace / self.relative_path

    def render(self, candidate: Function) -> bytes:
        """Replace just the selected function; retain all bytes outside its span."""
        if not isinstance(candidate, (ast.FunctionDef, ast.AsyncFunctionDef)):
            raise TypeError('proposal must be FunctionDef or AsyncFunctionDef')
        if candidate.name != self.function.name:
            raise ValueError('candidate must use the target function name')
        node = self.function
        lines = self.text.splitlines(keepends=True)
        start_line = min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1
        offsets = [0]
        for line in lines:
            offsets.append(offsets[-1] + len(line))

        def column(line, byte_column):
            # AST column positions count UTF-8 bytes, including in non-UTF8 files.
            return len(line.encode('utf-8')[:byte_column].decode('utf-8'))

        start_col = column(lines[start_line], node.col_offset)
        end_col = column(lines[node.end_lineno - 1], node.end_col_offset)
        start = offsets[start_line] + start_col
        end = offsets[node.end_lineno - 1] + end_col
        indent = lines[start_line][:start_col]
        newline = '\r\n' if '\r\n' in self.text else '\n'
        replacement = ast.unparse(ast.fix_missing_locations(deepcopy(candidate)))
        replacement = (newline + indent).join(replacement.splitlines())
        text = self.text[:start] + replacement + self.text[end:]
        compile(text, str(self.path), 'exec')
        return text.encode(self.encoding)


@dataclass
class Proposal:
    label: str
    function: Function


class ProposalProvider(Protocol):
    def __call__(self, function: Function, rng: random.Random) -> Iterable[Proposal]: ...


def parse_proposal(source: str) -> Function:
    tree = ast.parse(source, type_comments=True)
    nodes = list(tree.body)
    if nodes and isinstance(nodes[0], ast.Expr) and isinstance(nodes[0].value, ast.Constant) \
            and isinstance(nodes[0].value.value, str):
        nodes.pop(0)
    if len(nodes) != 1 or not isinstance(nodes[0], (ast.FunctionDef, ast.AsyncFunctionDef)):
        raise ValueError('a proposal file must contain exactly one complete function')
    return nodes[0]


class SourceProposals:
    """Full alternative algorithms supplied as Python files; no AST allowlist."""
    def __init__(self, paths: Iterable[Path]):
        self.items = []
        for path in paths:
            with tokenize.open(path) as stream:
                self.items.append(Proposal(str(path), parse_proposal(stream.read())))

    def __call__(self, function, rng):
        for item in self.items:
            yield Proposal(item.label, deepcopy(item.function))


def load_provider(reference: str) -> ProposalProvider:
    """A trusted plugin exports callable(function_ast, random) -> proposals."""
    module, separator, attribute = reference.partition(':')
    if not separator:
        raise ValueError('provider must be module:callable')
    provider = getattr(importlib.import_module(module), attribute)
    if not callable(provider):
        raise TypeError('provider must be callable')
    return provider


def _replace(root: Function, index: int, replacement: ast.AST) -> Function:
    clone = deepcopy(root)
    old = list(ast.walk(clone))[index]

    class Replace(ast.NodeTransformer):
        def visit(self, node):
            if node is old:
                return deepcopy(replacement)
            return super().visit(node)

    return ast.fix_missing_locations(Replace().visit(clone))


def mutations(function: Function, rng: random.Random) -> Iterable[Proposal]:
    """Domain-independent edits. Semantics are decided by the contract, not assumed.

    These operators are a starting search neighborhood, not a complete grammar
    for algorithm invention. Plugins can replace any part of a function AST.
    """
    # Recognize a common imperative reduction and propose a different algorithm.
    body = function.body
    prefix = []
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
            and isinstance(body[0].value.value, str):
        prefix, body = body[:1], body[1:]
    if len(body) == 3:
        initial, loop, result = body
        if (isinstance(initial, ast.Assign) and len(initial.targets) == 1
                and isinstance(initial.targets[0], ast.Name)
                and isinstance(initial.value, ast.Constant) and initial.value.value == 0
                and isinstance(loop, ast.For) and not loop.orelse and len(loop.body) == 1
                and isinstance(loop.body[0], ast.AugAssign) and isinstance(loop.body[0].op, ast.Add)
                and isinstance(loop.body[0].target, ast.Name) and isinstance(result, ast.Return)
                and isinstance(result.value, ast.Name)
                and initial.targets[0].id == loop.body[0].target.id == result.value.id):
            node = deepcopy(function)
            expression = ast.GeneratorExp(elt=deepcopy(loop.body[0].value), generators=[
                ast.comprehension(target=deepcopy(loop.target), iter=deepcopy(loop.iter), ifs=[], is_async=0)])
            if (isinstance(loop.target, ast.Name) and isinstance(loop.body[0].value, ast.Name)
                    and loop.target.id == loop.body[0].value.id):
                expression = deepcopy(loop.iter)
            node.body = deepcopy(prefix) + [ast.Return(value=ast.Call(
                func=ast.Name(id='sum', ctx=ast.Load()), args=[expression], keywords=[]))]
            yield Proposal('loop-reduction-to-sum', ast.fix_missing_locations(node))

    local = []
    for index, node in enumerate(ast.walk(function)):
        replacements = []
        if isinstance(node, ast.ListComp):
            replacements.append(('comprehension-to-generator', ast.GeneratorExp(
                elt=deepcopy(node.elt), generators=deepcopy(node.generators))))
        if isinstance(node, ast.BinOp):
            for operator in (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Pow):
                if not isinstance(node.op, operator):
                    replacements.append((f'binary-{operator.__name__}', ast.BinOp(
                        left=deepcopy(node.left), op=operator(), right=deepcopy(node.right))))
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            for value in (-1, 0, 1, 2):
                value = type(node.value)(value)
                if value != node.value:
                    replacements.append((f'constant-{value}', ast.Constant(value=value)))
        if isinstance(node, ast.If) and len(node.body) == len(node.orelse) == 1:
            a, b = node.body[0], node.orelse[0]
            if isinstance(a, ast.Return) and isinstance(b, ast.Return) and a.value and b.value:
                replacements.append(('conditional-return', ast.Return(value=ast.IfExp(
                    test=deepcopy(node.test), body=deepcopy(a.value), orelse=deepcopy(b.value)))))
        for label, replacement in replacements:
            local.append((index, label, replacement))
    rng.shuffle(local)
    for index, label, replacement in local:
        yield Proposal(f'{label}@{index}', _replace(function, index, replacement))
