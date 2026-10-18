import random
import torch
from nexus_m.tasks import sample_task
from nexus_m.features import encode
from nexus_m.model import NeuralSociety
from nexus_m.learning import supervised_loss,save_checkpoint,restore_checkpoint
from nexus_m.diagnostics import communication_intervention,agent_lesion

def test_model_shapes_learning_and_routing(tmp_path):
    torch.manual_seed(1)
    model=NeuralSociety(agents=8,width=16,message_dim=4,top_k=2,recurrence=2)
    task=sample_task(random.Random(10))
    output=model(encode(task.public()))
    assert output.states.shape==(8,16)
    assert output.messages.shape==(8,4)
    assert output.routing.shape==(8,8)
    assert int((output.routing>0).sum())==8*2
    assert torch.allclose(output.routing.sum(-1),torch.ones(8),atol=1e-6)
    assert torch.allclose(output.routing.diag(),torch.zeros(8))
    assert output.logits.ndim==1
    loss,_=supervised_loss(model,task)
    loss.backward()
    assert model.agent_bias.grad is not None
    assert model.agent_bias.grad.abs().sum()>0
    opt=torch.optim.Adam(model.parameters(),lr=0.001)
    opt.step()
    save_checkpoint(tmp_path/'checkpoint.pt',model,opt,1,{'seed':1})
    clone,meta=restore_checkpoint(tmp_path/'checkpoint.pt')
    assert meta['step']==1
    assert torch.allclose(clone(encode(task.public())).logits,model(encode(task.public())).logits)

def test_float64_and_ablations():
    model=NeuralSociety(agents=4,width=10,top_k=2,message_dim=4).double()
    task=sample_task(random.Random(14))
    out=model(encode(task.public(),dtype=torch.float64))
    assert out.states.dtype==torch.float64
    assert (out.routing>0).sum().item()==8
    assert communication_intervention(model,task)['communication_links']==8
    assert agent_lesion(model,task,which=2)['agent']==2
