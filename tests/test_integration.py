import csv,random
from nexus_m.model import NeuralSociety
from nexus_m.benchmark import run_benchmark
from nexus_m.learning import TrainSettings,train,restore_checkpoint,adapt_to_examples
from nexus_m.tasks import sample_task
from nexus_m.memory import PersistentMemory
from nexus_m.telemetry import Telemetry

def test_continual_pipeline(tmp_path):
    model=NeuralSociety(agents=4,width=8,message_dim=4,top_k=2)
    with PersistentMemory(tmp_path/'memory.sqlite') as memory,Telemetry(tmp_path/'telemetry') as logs:
        result=train(model,TrainSettings(steps=3,log_every=1,checkpoint_every=2),
              tmp_path,memory,logs)
        assert memory.stats()['episodes']==3
        assert result['steps']==3
    clone,_=restore_checkpoint(tmp_path/'model_final.pt')
    adap=adapt_to_examples(clone,sample_task(random.Random(6),'ood').public(),steps=2)
    assert adap['steps']==2

def test_benchmark_smoke(tmp_path):
    out=run_benchmark(tmp_path/'benchmark',widths=(8,),seeds=(11,),agents=4,
                  steps=2,simulations=4,episodes=2,message_dim=4,
                  top_k=2,recurrence=1,telemetry_level='off')
    assert len(out)==4
    assert (tmp_path/'benchmark'/'summary.csv').exists()
    assert {x['scenario'] for x in out}=={
         'neural_collective_memory','neural_collective_no_memory',
         'neural_isolated','uniform_mcts'}
