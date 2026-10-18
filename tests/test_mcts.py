import random
from nexus_m.tasks import generate_task
from nexus_m.mcts import SharedMCTS
from nexus_m.memory import PersistentMemory

def test_random_mcts_finds_simple_program():
    task=generate_task(random.Random(20),'shift',n_examples=5)
    result=SharedMCTS(model=None,simulations=1000,seed=1,stop_on_exact=True).solve(task.public())
    assert result.program is not None
    assert result.example_exact
    assert result.simulations<=1000

def test_memory_mcts_reuse(tmp_path):
    task=generate_task(random.Random(4),'square')
    with PersistentMemory(tmp_path/'m.sqlite') as m:
        m.remember(task.public(),task.oracle)
        res=SharedMCTS(model=None,memory=m,simulations=20).solve(task.public())
        assert res.source=='memory' and res.simulations==0
        assert res.example_exact
