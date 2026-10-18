"""Reproducible, leakage-controlled benchmarks and architectural ablations."""
from __future__ import annotations
from pathlib import Path
import csv,json,random,time,statistics
import torch
from .tasks import sample_task
from .grammar import run_many,value_on_examples
from .mcts import SharedMCTS
from .model import NeuralSociety
from .learning import TrainSettings,train
from .memory import PersistentMemory
from .telemetry import Telemetry
from .precision import network_dtype


def eval_one(task,result)->dict:
    if result.program:
        predicted=run_many(result.program,[x for x,_ in task.heldout])
        correct=sum(int(p==y) for p,(_,y) in zip(predicted,task.heldout))
        accuracy=correct/max(1,len(task.heldout))
    else:accuracy=0.0
    return {'family':task.family,'example_exact':int(result.example_exact),
            'ood_accuracy':float(accuracy),'ood_exact':int(accuracy==1.),
            'simulations':result.simulations,'expanded_nodes':result.expanded_nodes,
            'max_depth_reached':result.max_depth_reached,'source':result.source,
            'elapsed_sec':result.elapsed_sec,
            'program':' '.join(result.program or ())}


def evaluate_configuration(model,memory,task_list,simulations=100,max_depth=15,
                           seed=2026,communication=True,neural=True,
                           use_memory=True,telemetry=None):
    rows=[]
    for i,task in enumerate(task_list):
        searcher=SharedMCTS(model=model if neural else None,
             memory=memory if use_memory else None,
             telemetry=telemetry,simulations=simulations,max_depth=max_depth,
             seed=seed+i*8191, use_neural=neural,
             ablate_communication=not communication)
        # Never persist OOD evaluation task solutions: no test-time memory leakage.
        result=searcher.solve(task.public(),store=False)
        rows.append(eval_one(task,result))
    return rows


def aggregate(rows):
    if not rows:return {}
    keys=('example_exact','ood_accuracy','ood_exact','simulations','expanded_nodes','elapsed_sec')
    return {k:float(statistics.mean(r[k] for r in rows)) for k in keys}


def run_benchmark(output_dir,widths=(100,1000,10000),seeds=(2026,),agents=100,
                  steps=1000,simulations=100,episodes=12,precision='adaptive',
                  message_dim=16,top_k=4,recurrence=2,
                  max_depth=15,run_ablation=True,telemetry_level='summary'):
    """Train each width and evaluate OOD tasks + controls.

    Large widths and 1000 steps can be CPU-expensive. Reporting actual timings
    rather than asserting runtimes or accuracy ahead of measurement.
    """
    output=Path(output_dir);output.mkdir(parents=True,exist_ok=True)
    all_rows=[];summary=[]
    for seed in seeds:
        rng=random.Random(seed+999)
        # OOD tasks are generated once per seed and reused at every width.
        tasks=[sample_task(rng,'ood',5,20) for _ in range(episodes)]
        for width in widths:
            run_dir=output/f'seed_{seed}'/f'width_{width}'
            run_dir.mkdir(parents=True,exist_ok=True)
            torch.manual_seed(seed+width)
            model=NeuralSociety(agents=agents,width=width,message_dim=message_dim,
                               top_k=min(top_k,agents-1),recurrence=recurrence)
            model=model.to(dtype=network_dtype(precision))
            with PersistentMemory(run_dir/'memory.sqlite') as memory, Telemetry(
                     run_dir/'telemetry',level=telemetry_level,
                     checkpoint_every=0) as logs:
                log_every=max(1,steps//20)
                settings=TrainSettings(steps=steps,seed=seed,log_every=log_every,
                                       checkpoint_every=steps if steps else 0)
                training=train(model,settings,run_dir,memory,logs)
                model.eval()
                scenarios=[('neural_collective_memory',True,True,True)]
                if run_ablation:
                    scenarios.extend([
                       ('neural_collective_no_memory',True,True,False),
                       ('neural_isolated',False,True,False),
                       ('uniform_mcts',False,False,False)])
                for name,comms,neural,remember in scenarios:
                    rows=evaluate_configuration(model,memory,tasks,simulations,max_depth,
                         seed,comms,neural,remember,telemetry=None)
                    for i,row in enumerate(rows):
                        all_rows.append({'seed':seed,'width':width,'agents':agents,
                                         'scenario':name,'task_index':i,**row})
                    agg=aggregate(rows)
                    summary.append({'seed':seed,'width':width,'agents':agents,'scenario':name,
                                    'neurons':width*agents,'parameters':model.report()['trainable_parameters'],
                                    'train_sec':training['duration_sec'],**agg})
                (run_dir/'architecture.json').write_text(json.dumps(model.report(),indent=2))
            write_csv(output/'results_by_task.csv',all_rows)
            write_csv(output/'summary.csv',summary)
            (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    return summary


def write_csv(path,rows):
    if not rows:return
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def plot_summary(csv_file,output_file):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with open(csv_file,newline='',encoding='utf-8') as f: rows=list(csv.DictReader(f))
    values={}
    for row in rows:
        key=(int(row['width']),row['scenario'])
        values.setdefault(key,[]).append(float(row['ood_exact']))
    widths=sorted(set(k[0] for k in values))
    names=sorted(set(k[1] for k in values))
    fig,ax=plt.subplots(figsize=(10,5))
    for n,name in enumerate(names):
        x=[i+(n-(len(names)-1)/2)*0.17 for i in range(len(widths))]
        y=[statistics.mean(values.get((w,name),[0])) for w in widths]
        ax.bar(x,y,width=.16,label=name)
    ax.set_xticks(list(range(len(widths))),[str(w) for w in widths])
    ax.set_xlabel('Neurons per agent')
    ax.set_ylabel('Exact OOD generalization rate')
    ax.set_ylim(0,1)
    ax.set_title('NEXUS-M: out-of-family verified program synthesis')
    ax.legend(fontsize=8)
    fig.tight_layout()
    Path(output_file).parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output_file,dpi=160)
    plt.close(fig)
