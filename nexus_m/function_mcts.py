"""UCB tree search over complete Python function ASTs and composable edits."""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
from copy import deepcopy
import math
import random
import time
from typing import Iterator

from .function_ast import Function, FunctionTarget, Proposal, ProposalProvider, digest
from .function_evaluator import FunctionMeasurement, ProjectEvaluator


def ast_key(function):
    return ast.dump(function, include_attributes=False)


@dataclass
class FunctionNode:
    function: Function
    label: str
    depth: int = 0
    visits: int = 0
    reward_sum: float = 0.
    children: list['FunctionNode'] = field(default_factory=list)
    proposals: Iterator[Proposal] | None = None
    exhausted: bool = False
    attempts: int = 0
    @property
    def mean(self):
        return self.reward_sum / self.visits if self.visits else 0.


@dataclass
class FunctionCandidate:
    label: str
    function: Function
    source: bytes
    measurement: FunctionMeasurement
    gain: float


@dataclass
class FunctionSearch:
    baseline: FunctionMeasurement
    candidates: list[FunctionCandidate]
    trace: list[dict]
    elapsed_seconds: float


class FunctionMCTS:
    def __init__(self, target: FunctionTarget, evaluator: ProjectEvaluator,
                 providers: list[ProposalProvider], *, simulations=32, max_depth=3,
                 max_children=16, exploration=1.4, seed=2026,reward_fn=None):
        if min(simulations, max_depth, max_children) < 1:
            raise ValueError('simulations, max_depth and max_children must be positive')
        if not math.isfinite(exploration) or exploration < 0:
            raise ValueError('exploration must be finite and nonnegative')
        self.target, self.evaluator, self.providers = target, evaluator, providers
        self.simulations, self.max_depth, self.max_children = simulations, max_depth, max_children
        self.exploration, self.rng = exploration, random.Random(seed)
        self.root = FunctionNode(deepcopy(target.function), 'original')
        self.reward_fn=reward_fn

    def _proposals(self, function):
        for provider in self.providers:
            yield from provider(deepcopy(function), self.rng)

    def _expand(self, node, ancestor_keys):
        if node.exhausted or node.depth >= self.max_depth:
            return None
        if node.proposals is None:
            node.proposals = iter(self._proposals(node.function))
        # Trusted providers can be arbitrarily powerful. Still avoid spending an
        # unbounded number of iterations replaying duplicate proposals.
        while node.attempts < self.max_children * 16 and len(node.children) < self.max_children:
            node.attempts += 1
            try:
                proposal = next(node.proposals)
            except StopIteration:
                break
            if not isinstance(proposal, Proposal):
                raise TypeError('provider must yield Proposal instances')
            key = ast_key(proposal.function)
            if key in ancestor_keys or any(ast_key(child.function) == key for child in node.children):
                continue
            child = FunctionNode(deepcopy(proposal.function), proposal.label, node.depth + 1)
            node.children.append(child)
            return child
        node.exhausted = True
        return None

    def _select(self):
        node = self.root
        path = [node]
        keys = {ast_key(node.function)}
        while node.depth < self.max_depth:
            child = self._expand(node, keys)
            if child is not None:
                return child, path + [child]
            if not node.children:
                break
            node = max(node.children, key=lambda child: child.mean + self.exploration * math.sqrt(
                math.log(max(2, path[-1].visits + 1)) / max(1, child.visits)))
            path.append(node)
            keys.add(ast_key(node.function))
        return node, path

    def search(self) -> FunctionSearch:
        started = time.monotonic()
        baseline = self.evaluator.measure(self.target.original)
        if not baseline.passed:
            raise RuntimeError(f'baseline contract failed: {baseline.error}; {baseline.commands}')
        cache: dict[str, FunctionCandidate] = {}
        trace = []
        for index in range(self.simulations):
            node, path = self._select()
            error, key, gain = None, None, 0.
            cached, passed = False, False
            try:
                source = self.target.render(node.function)
                key = digest(source)
                if ast_key(node.function) == ast_key(self.target.function):
                    gain, passed = 0., True
                else:
                    cached = key in cache
                    if not cached:
                        measurement = self.evaluator.measure(source)
                        gain = ((baseline.seconds - measurement.seconds) / baseline.seconds
                                if measurement.passed else 0.)
                        cache[key] = FunctionCandidate(node.label, deepcopy(node.function),
                                                       source, measurement, gain)
                    item = cache[key]
                    gain, passed, error = item.gain, item.measurement.passed, item.measurement.error
            except (ValueError, TypeError, SyntaxError, UnicodeError) as exc:
                error = str(exc)
            reward = .1 + max(-.95, min(.95, gain)) if passed else -1.
            if self.reward_fn is not None:
                reward=self.reward_fn(self.target.function,node.function,gain,passed)
                if not math.isfinite(reward):raise ValueError('reward hook returned nonfinite score')
            for visited in path:
                visited.visits += 1
                visited.reward_sum += reward
            trace.append(dict(simulation=index + 1, depth=node.depth, proposal=node.label,
                              source_sha256=key, passed=passed, gain=gain,
                              reward=reward, cached=cached, error=error,
                              unique_candidates=len(cache)))
        return FunctionSearch(baseline, list(cache.values()), trace, time.monotonic()-started)
