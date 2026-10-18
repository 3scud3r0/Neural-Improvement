import random
from decimal import Decimal
import torch
from nexus_m.decimal_network import DecimalSociety
from nexus_m.precision import decimal_matmul_2x2,network_dtype
from nexus_m.model import NeuralSociety
from nexus_m.tasks import sample_task
from nexus_m.features import encode
from nexus_m.telemetry import Telemetry

def test_decimal_arbitrary_weights_and_messages():
    society=DecimalSociety(agents=3,width=3,digits=75,seed=4)
    old=society.forward('0.3',advance=False)
    assert old.precision_digits==75
    trained=society.step('0.3','0.2')
    assert trained['prediction']
    assert society.w_in[0][0] != Decimal(old.weight_sample[0])
    result=decimal_matmul_2x2([['0.1','0.2'],['0.3','0.4']],
          [['0.5','0.6'],['0.7','0.8']],digits=75)
    assert result[0][0]=='0.19'
    assert network_dtype('float64')==torch.float64

def test_full_telemetry_files(tmp_path):
    model=NeuralSociety(agents=3,width=8,message_dim=3,top_k=1)
    task=sample_task(random.Random(3))
    with Telemetry(tmp_path,level='full',record_weight_tensors=True) as logs:
        out=model(encode(task.public()))
        logs.neural(1,out)
        logs.weights(1,model)
    files=list((tmp_path/'tensor_snapshots').glob('*.pt'))
    assert len(files)==2
    log=(tmp_path/'events.jsonl').read_text()
    assert 'communication' in log and 'parameters' in log
