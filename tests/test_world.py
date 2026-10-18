import random
from nexus_m.world_model import HypothesisWorldModel,train_world_model,load_world_model
from nexus_m.tasks import sample_task
from nexus_m.mcts import SharedMCTS

def test_learned_dsl_transition_and_mcts(tmp_path):
    outcome=train_world_model(tmp_path,steps=4,hidden=16)
    assert outcome['steps']==4
    predictor=load_world_model(tmp_path/'world_model.pt')
    task=sample_task(random.Random(77),'train')
    surprise=predictor.surprise(task.public(),('X',),'ABS')
    assert surprise>=0
    mcts=SharedMCTS(simulations=8,world_model=predictor,world_weight=.2,
                    use_neural=False)
    out=mcts.solve(task.public())
    assert out.simulations<=8