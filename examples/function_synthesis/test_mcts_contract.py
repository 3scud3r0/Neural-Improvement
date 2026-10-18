"""Contract for the real SharedMCTS selector, including ties and zero visits."""
import math
import random
import pytest
from nexus_m.mcts import SearchNode, SharedMCTS


def check(seed, count):
    rng = random.Random(seed)
    search = SharedMCTS(model=None, exploration=1.5)
    for _ in range(count):
        parent = SearchNode((), visits=rng.randrange(0, 1000))
        for i in range(rng.randrange(1, 100)):
            parent.children[str(i)] = SearchNode((str(i),), prior=rng.random(),
                visits=rng.randrange(0, 200), score_sum=rng.uniform(-10, 100))
        expected = max(parent.children.values(), key=lambda c:
            c.q + search.exploration*c.prior*math.sqrt(max(1, parent.visits))/(1+c.visits))
        assert search._select(parent) is expected


def test_training_selector():
    check(2026, 60)


def test_heldout_selector():
    check(91991, 160)
    search = SharedMCTS(model=None)
    parent = SearchNode((), visits=0)
    parent.children = {'a': SearchNode(('a',), prior=0), 'b': SearchNode(('b',), prior=0)}
    assert search._select(parent) is parent.children['a']
    parent.children.clear()
    with pytest.raises(ValueError):
        search._select(parent)
