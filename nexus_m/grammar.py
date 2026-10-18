"""Small safe, deterministic postfix language for verifiable program synthesis.

No eval/exec or external code execution. An individual program consumes one
integer x and returns one bounded integer; all operators are total on bounds.
"""
from __future__ import annotations
from typing import Sequence

CONSTANTS = (-3, -2, -1, 0, 1, 2, 3)
PUSH = ["X"] + [f"C{v}" for v in CONSTANTS]
UNARY = ("NEG", "ABS", "SQUARE", "MOD2", "HALF")
BINARY = ("ADD", "SUB", "MUL", "MIN", "MAX")
HALT = "HALT"
TOKENS = tuple(PUSH + list(UNARY) + list(BINARY) + [HALT])
TOKEN_TO_ID = {token: i for i, token in enumerate(TOKENS)}
LIMIT = 10**10


def valid_actions(program: Sequence[str], max_tokens: int = 15) -> list[str]:
    """Grammar pruning ensures all prefixes have a well-defined stack."""
    if not program or program[-1] != HALT:
        pass
    else:
        return []
    depth = 0
    uses_input = False
    for tok in program:
        if tok in PUSH:
            depth += 1
            uses_input |= tok == "X"
        elif tok in UNARY:
            if depth < 1:
                return []
        elif tok in BINARY:
            if depth < 2:
                return []
            depth -= 1
        else:
            return []
        if depth > 6:
            return []
    if len(program) >= max_tokens:
        return [HALT] if depth == 1 and uses_input else []
    available = []
    # Reserve at least one token for HALT, and enough operations to close stack.
    remaining = max_tokens - len(program)
    if remaining > depth + 1:
        if depth < 5:
            available.extend(PUSH)
        if depth >= 1:
            available.extend(UNARY)
        if depth >= 2:
            available.extend(BINARY)
    elif depth >= 2 and remaining >= depth:
        available.extend(BINARY)
    if depth == 1 and uses_input:
        available.append(HALT)
    return available


def execute(program: Sequence[str], x: int) -> int:
    """Pure bounded execution; ValueError for malformed or explosive programs."""
    stack: list[int] = []
    if not isinstance(x, int) or abs(x) > LIMIT:
        raise ValueError("Integer input out of bounds")
    for i, tok in enumerate(program):
        if tok == HALT:
            if i != len(program) - 1 or len(stack) != 1 or "X" not in program:
                raise ValueError("Invalid HALT")
            return stack.pop()
        if tok == "X":
            stack.append(x)
        elif tok.startswith("C") and tok in PUSH:
            stack.append(int(tok[1:]))
        elif tok in UNARY:
            if not stack:
                raise ValueError("Stack underflow")
            a = stack.pop()
            out = {"NEG": lambda: -a, "ABS": lambda: abs(a),
                   "SQUARE": lambda: a*a, "MOD2": lambda: a % 2,
                   "HALF": lambda: a // 2}[tok]()
            stack.append(out)
        elif tok in BINARY:
            if len(stack) < 2:
                raise ValueError("Stack underflow")
            b, a = stack.pop(), stack.pop()
            out = {"ADD": lambda: a+b, "SUB": lambda: a-b,
                   "MUL": lambda: a*b, "MIN": lambda: min(a,b),
                   "MAX": lambda: max(a,b)}[tok]()
            stack.append(out)
        else:
            raise ValueError(f"Unknown instruction {tok}")
        if len(stack) > 6 or (stack and abs(stack[-1]) > LIMIT):
            raise ValueError("Program overflow")
    raise ValueError("Missing HALT")


def safe_execute(program: Sequence[str], x: int) -> int | None:
    try:
        return execute(program, x)
    except (ValueError, OverflowError, TypeError):
        return None


def run_many(program: Sequence[str], inputs: Sequence[int]) -> list[int | None]:
    return [safe_execute(program, int(x)) for x in inputs]


def value_on_examples(program: Sequence[str], examples: Sequence[tuple[int,int]]) -> tuple[float,bool]:
    if not program or program[-1] != HALT:
        return 0.0, False
    actual = run_many(program, [int(x) for x, _ in examples])
    if any(v is None for v in actual):
        return 0.0, False
    errors = [abs(int(pred)-int(y)) for pred, (_, y) in zip(actual, examples)]
    exact = all(e == 0 for e in errors)
    scale = 1 + sum(abs(int(y)) for _,y in examples) / max(len(examples),1)
    mae = sum(errors) / max(1,len(errors))
    return float(1.0 / (1.0 + mae / scale)), exact


def pretty(program: Sequence[str]) -> str:
    stack: list[str] = []
    try:
        for t in program:
            if t == HALT: return stack[0] if len(stack)==1 else "<invalid>"
            if t == 'X': stack.append('x')
            elif t in PUSH: stack.append(t[1:])
            elif t in UNARY:
                a=stack.pop()
                stack.append({"NEG":f"(-{a})","ABS":f"abs({a})",
                              "SQUARE":f"({a}²)","MOD2":f"({a}%2)",
                              "HALF":f"floor({a}/2)"}[t])
            else:
                b,a=stack.pop(),stack.pop()
                op={"ADD":"+","SUB":"-","MUL":"*","MIN":"min","MAX":"max"}[t]
                stack.append(f"{op}({a},{b})" if t in ('MIN','MAX') else f"({a} {op} {b})")
    except (IndexError,KeyError):
        return "<invalid>"
    return '<unfinished>'
