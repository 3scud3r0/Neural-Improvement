import random
from nexus_m.tasks import generate_task
from nexus_m.memory import PersistentMemory

def test_persist_consolidate_retrieve(tmp_path):
    task=generate_task(random.Random(1),'shift')
    db=tmp_path/'memory.sqlite'
    with PersistentMemory(db) as m:
        saved=m.remember(task.public(),task.oracle,origin='test')
        assert saved['exact_on_examples']
        assert m.stats()['concepts']==1
        match=m.retrieve(task.public())[0]
        assert match['exact']
        m.snapshot(tmp_path/'backup.sqlite')
    with PersistentMemory(db) as m:
        assert m.retrieve(task.public())[0]['program']==list(task.oracle)
    assert (tmp_path/'backup.sqlite').exists()
