import random,torch
from nexus_m.learning import adapt_to_program
from nexus_m.model import NeuralSociety
from nexus_m.tasks import sample_task

def test_program_distillation_uses_observed_only_and_restores():
    torch.manual_seed(42)
    model=NeuralSociety(agents=4,width=10,message_dim=4,top_k=2,recurrence=1)
    task=sample_task(random.Random(7),'train')
    before={k:v.detach().clone() for k,v in model.state_dict().items()}
    out=adapt_to_program(model,task.public(),task.oracle,steps=2,persist=False)
    assert out['exact_on_examples']
    assert out['steps']==2
    assert all(torch.equal(model.state_dict()[k],v) for k,v in before.items())
