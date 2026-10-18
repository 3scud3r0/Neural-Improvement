"""Evaluate three ALREADY trained widths without rerunning 1000 training steps.

All sizes see the same public few-shot tasks. Hidden examples are accessed
only after the search completes. Avoid touching test examples in memory.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse,csv,json,random,statistics,time
from nexus_m.learning import restore_checkpoint
from nexus_m.memory import PersistentMemory
from nexus_m.benchmark import evaluate_configuration,aggregate,write_csv,plot_summary
from nexus_m.tasks import sample_task

p=argparse.ArgumentParser()
p.add_argument('--root',default='outputs')
p.add_argument('--output',default='outputs/comparacao_treinados')
p.add_argument('--widths',nargs='+',type=int,default=[100,1000,10000])
p.add_argument('--seeds',nargs='+',type=int,default=[777,778,779])
p.add_argument('--episodes',type=int,default=12)
p.add_argument('--simulations',type=int,default=300)
p.add_argument('--depth',type=int,default=32)
a=p.parse_args();root=Path(a.root);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
all_rows=[];summary=[]
for seed in a.seeds:
    rng=random.Random(seed)
    task_list=[sample_task(rng,'ood',5,20) for _ in range(a.episodes)]
    for width in a.widths:
        folder=root/f'real_100x{width}_1000'
        model,bundle=restore_checkpoint(folder/'model_final.pt')
        model.eval()
        with PersistentMemory(folder/'memory.sqlite') as memory:
            conditions=[('neural_collective_memory',True,True,True),
                        ('neural_collective_no_memory',True,True,False),
                        ('neural_isolated',False,True,False),
                        ('uniform_mcts',False,False,False)]
            for name,comm,neural,mem in conditions:
                rows=evaluate_configuration(model,memory,task_list,simulations=a.simulations,
                     max_depth=a.depth,seed=seed,communication=comm,neural=neural,
                     use_memory=mem,telemetry=None)
                for i,row in enumerate(rows):
                    all_rows.append({'seed':seed,'width':width,'agents':100,'scenario':name,
                                     'task_index':i,**row})
                ag=aggregate(rows)
                summary.append({'seed':seed,'width':width,'agents':100,
                                'neurons':model.neuron_count,
                                'parameters':model.report()['trainable_parameters'],
                                'scenario':name,'train_sec':0.,**ag})
                print('width=',width,'seed=',seed,'scenario=',name,
                      'ood_exact=',round(ag['ood_exact'],3),'ood_accuracy=',round(ag['ood_accuracy'],3),flush=True)
            write_csv(out/'results_by_task.csv',all_rows)
            write_csv(out/'summary.csv',summary)
(out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
plot_summary(out/'summary.csv',out/'comparison.png')
print('Saved',out)
