import random
import pytest
from nexus_m.grammar import execute,safe_execute,valid_actions,value_on_examples,pretty
from nexus_m.tasks import generate_task,TRAIN_FAMILIES,OOD_FAMILIES

def test_exact_integer_language():
    p=['X','C2','MUL','C1','ADD','HALT']
    assert [execute(p,x) for x in [-4,-1,0,11]]==[-7,-1,1,23]
    assert value_on_examples(p,[(0,1),(11,23)])[1]
    assert not value_on_examples(['X','HALT'],[(0,1)])[1]

def test_program_safety():
    assert safe_execute(['X','MUL','HALT'],5) is None
    assert safe_execute(['X','C2','MUL','C1','ADD','HALT'],10**11) is None
    assert 'MUL' not in valid_actions((),15)
    assert 'HALT' in valid_actions(['X'],15)
    with pytest.raises(ValueError):execute(['X','NO_SUCH_OPCODE','HALT'],5)

def test_family_separation_and_no_oracle_leak():
    assert not (set(TRAIN_FAMILIES)&set(OOD_FAMILIES))
    for f in (*TRAIN_FAMILIES,*OOD_FAMILIES):
        task=generate_task(random.Random(123),f)
        assert all(execute(task.oracle,x)==y for x,y in task.heldout)
        public=task.public()
        assert public.oracle is None
        assert public.heldout==()
        assert public.family=='unlabeled'
