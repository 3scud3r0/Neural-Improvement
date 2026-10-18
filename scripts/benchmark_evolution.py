"""Small reproducible OOD-style macro-synthesis benchmark with explicit controls.

No training data leakage: search sees only five I/O demonstrations; test inputs
and oracle programs are accessed only AFTER prediction. The tasks share the
macro grammar, so they are not evidence of open-world AGI generalization.
"""
from __future__ import annotations
import argparse,csv,json,random,statistics,time
from collections import defaultdict
from pathlib import Path
from nexus_m.evolution_system import EvolutionSystem
from nexus_m.hierarchical_abstraction_search import evaluate_macro_program

FAMILIES={
 'affine_A':['ADD_2','MUL_2','SUB_1'],
 'affine_B':['MUL_3','SUB_2'],
 'absolute':['SUB_2','ABS','ADD_3'],
 'quadratic':['SQUARE','ADD_2'],
 'negative':['NEG','ADD_3'],
 'shift_square':['ADD_1','SQUARE'],
 'folded':['ADD_2','ABS','ADD_1'],
 'double_shift':['MUL_2','ADD_2'],
}
CONDITIONS=('uniform','neural_untrained','latent_only','collective_online')


def run(seeds=(2026,2027),simulations=300,output='outputs/evolution_benchmark'):
    root=Path(output);root.mkdir(parents=True,exist_ok=True)
    rows=[];started=time.perf_counter()
    for seed in seeds:
        rng=random.Random(seed)
        pairs=list(FAMILIES.items());rng.shuffle(pairs)
        public_x=[-4,-1,0,2,5]
        private_x=[-30,-15,-7,8,11,27,50]
        for condition in CONDITIONS:
            with EvolutionSystem(root/f'{seed}_{condition}',agents=8,width=32,seed=seed) as system:
                if condition in ('latent_only','collective_online'):
                    system.train_latent(steps=80)
                for name,truth in pairs:
                    public_y=evaluate_macro_program(truth,public_x)
                    private_y=evaluate_macro_program(truth,private_x)
                    examples=list(zip(public_x,public_y))
                    start=time.perf_counter()
                    result=system.solve(examples,simulations=simulations,max_depth=6,
                       use_neural=condition in ('neural_untrained','collective_online'),
                       use_memory=condition=='collective_online',
                       use_latent=condition in ('latent_only','collective_online'),
                       heldout_inputs=private_x)
                    duration=time.perf_counter()-start
                    accurate=bool(result['program']) and result['heldout_predictions_unscored']==private_y
                    simulations_spent=result['searched']['simulations'] if result['searched'] else 0
                    if condition=='collective_online' and result['program']:
                        system.distill(examples,result['program'],steps=3)
                    rows.append({'seed':seed,'condition':condition,'task':name,
                         'examples_exact':int(result['found_on_examples']),
                         'holdout_exact':int(accurate),
                         'mcts_simulations':simulations_spent,
                         'source':result['origin'],'elapsed_search_sec':duration,
                         'program':json.dumps(result['program']),
                         'oracle_hidden_during_search':json.dumps(truth),
                         'heldout_samples':len(private_x)})
    with (root/'results.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    aggregated=[]
    for condition in CONDITIONS:
        subset=[row for row in rows if row['condition']==condition]
        aggregated.append({'condition':condition,'tasks':len(subset),
          'example_solved_rate':sum(x['examples_exact'] for x in subset)/len(subset),
          'heldout_exact_rate':sum(x['holdout_exact'] for x in subset)/len(subset),
          'average_mcts_simulations':statistics.mean(x['mcts_simulations'] for x in subset),
          'total_search_seconds':sum(x['elapsed_search_sec'] for x in subset),
          'retrieved_from_memory':sum(x['source']=='knowledge_graph_retrieval' for x in subset)})
    summary={'seeds':list(seeds),'task_families':list(FAMILIES),
       'simulations_budget_per_task':simulations,'conditions':aggregated,
       'elapsed_total_sec':time.perf_counter()-started,
       'caveats':['All tasks share the same handcrafted macro grammar',
           'Latent/collective conditions have additional training compute',
           'Neural prior in neural_untrained condition has not been trained for macros',
           'A solution verified on samples is not a formal universal proof',
           'Small number of seeds; no claims of general performance']}
    (root/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    return summary

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--seeds',nargs='+',type=int,default=[2026,2027,2028])
    parser.add_argument('--simulations',type=int,default=350)
    parser.add_argument('--output',default='outputs/evolution_benchmark')
    args=parser.parse_args()
    print(json.dumps(run(args.seeds,args.simulations,args.output),indent=2,ensure_ascii=False))
