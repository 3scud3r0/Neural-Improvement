"""CLI: python -m nexus_m COMMAND --help / nexus-m COMMAND --help."""
from __future__ import annotations
import argparse,json,random,time
from pathlib import Path
import torch
from .model import NeuralSociety
from .learning import TrainSettings,train,restore_checkpoint,adapt_to_examples,adapt_to_program
from .memory import PersistentMemory
from .telemetry import Telemetry
from .mcts import SharedMCTS
from .tasks import Task,sample_task,generate_task,OOD_FAMILIES
from .grammar import pretty,run_many
from .precision import network_dtype,decimal_matmul_2x2,describe_precision
from .decimal_network import DecimalSociety
from .world_model import train_world_model,load_world_model
from .benchmark import run_benchmark,plot_summary,eval_one


def root_parser():
    p=argparse.ArgumentParser(prog='nexus-m',description='NEXUS-M modular neural research laboratory')
    sub=p.add_subparsers(dest='command',required=True)
    t=sub.add_parser('train',help='Train a society; save checkpoints + telemetry + SQLite memory')
    t.add_argument('--agents',type=int,default=100)
    t.add_argument('--width',type=int,choices=(100,1000,10000),default=100)
    t.add_argument('--steps',type=int,default=1000)
    t.add_argument('--message-dim',type=int,default=16)
    t.add_argument('--top-k',type=int,default=4)
    t.add_argument('--rounds',type=int,default=2)
    t.add_argument('--seed',type=int,default=2026)
    t.add_argument('--lr',type=float,default=0.0008)
    t.add_argument('--precision',choices=('float32','float64','adaptive'),default='adaptive')
    t.add_argument('--telemetry',choices=('off','summary','full'),default='summary')
    t.add_argument('--log-every',type=int,default=50)
    t.add_argument('--checkpoint-every',type=int,default=200)
    t.add_argument('--search-improve-every',type=int,default=0)
    t.add_argument('--output',default='runs/training')
    s=sub.add_parser('solve',help='Use MCTS to synthesize a verified program from few-shot samples')
    s.add_argument('--model',default='')
    s.add_argument('--family',choices=OOD_FAMILIES,default='shift_absolute')
    s.add_argument('--examples',type=int,default=5)
    s.add_argument('--examples-file',default='',help='JSON with public examples and optional heldout pairs')
    s.add_argument('--simulations',type=int,default=1000)
    s.add_argument('--depth',type=int,default=15)
    s.add_argument('--seed',type=int,default=2027)
    s.add_argument('--memory',default='')
    s.add_argument('--no-communication',action='store_true')
    s.add_argument('--no-neural',action='store_true')
    s.add_argument('--world-model',default='',help='Optional trained DSL transition predictor')
    s.add_argument('--world-weight',type=float,default=0.0)
    s.add_argument('--adapt-steps',type=int,default=0,help='Temporary distillation of a found sample-consistent program')
    s.add_argument('--output',default='runs/solve')
    b=sub.add_parser('benchmark',help='Train and compare widths, memory and communication controls')
    b.add_argument('--widths',type=int,nargs='+',choices=(100,1000,10000),default=[100,1000,10000])
    b.add_argument('--agents',type=int,default=100)
    b.add_argument('--steps',type=int,default=1000)
    b.add_argument('--simulations',type=int,default=100)
    b.add_argument('--episodes',type=int,default=12)
    b.add_argument('--seeds',nargs='+',type=int,default=[2026])
    b.add_argument('--precision',choices=('float32','float64','adaptive'),default='adaptive')
    b.add_argument('--message-dim',type=int,default=16)
    b.add_argument('--top-k',type=int,default=4)
    b.add_argument('--rounds',type=int,default=2)
    b.add_argument('--depth',type=int,default=15)
    b.add_argument('--telemetry',choices=('off','summary','full'),default='summary')
    b.add_argument('--output',default='runs/benchmark')
    d=sub.add_parser('decimal-demo',help='Actual arbitrary-precision Decimal weights AND messages')
    d.add_argument('--digits',type=int,default=80)
    d.add_argument('--agents',type=int,default=4)
    d.add_argument('--width',type=int,default=4)
    d.add_argument('--steps',type=int,default=2)
    d.add_argument('--output',default='runs/decimal_demo.json')
    w=sub.add_parser('world-train',help='Train optional predictor of DSL program-state transitions')
    w.add_argument('--steps',type=int,default=1000)
    w.add_argument('--seed',type=int,default=2026)
    w.add_argument('--hidden',type=int,default=96)
    w.add_argument('--output',default='runs/world_model')
    z=sub.add_parser('demo',help='Fast integrated end-to-end run, not a full training benchmark')
    z.add_argument('--output',default='runs/demo')
    z.add_argument('--agents',type=int,default=8)
    z.add_argument('--width',type=int,default=32)
    z.add_argument('--steps',type=int,default=12)
    z.add_argument('--simulations',type=int,default=80)
    v=sub.add_parser('inspect',help='Print model architecture or memory content')
    v.add_argument('--checkpoint',default='')
    v.add_argument('--memory',default='')
    a=sub.add_parser('adapt',help='Constrained unsupervised test-time adapter calibration')
    a.add_argument('--checkpoint',required=True)
    a.add_argument('--steps',type=int,default=5)
    a.add_argument('--seed',type=int,default=2027)
    return p


def main(argv=None):
    args=root_parser().parse_args(argv)
    torch.set_num_threads(min(4,torch.get_num_threads()))
    if args.command=='train':
        torch.manual_seed(args.seed)
        model=NeuralSociety(args.agents,args.width,args.message_dim,args.top_k,args.rounds)
        model=model.to(dtype=network_dtype(args.precision))
        path=Path(args.output)
        with PersistentMemory(path/'memory.sqlite') as memory, Telemetry(path/'telemetry',level=args.telemetry,
                           checkpoint_every=0,record_weight_tensors=(args.telemetry=='full')) as logs:
            out=train(model,TrainSettings(args.steps,args.lr,args.seed,args.log_every,
                        args.checkpoint_every,args.search_improve_every,
                        precision=args.precision),path,memory,logs)
            print(json.dumps({'architecture':model.report(),'result':out,'memory':memory.stats()},indent=2))
    elif args.command=='solve':
        rng=random.Random(args.seed)
        if args.examples_file:
            payload=json.loads(Path(args.examples_file).read_text(encoding='utf-8'))
            examples=tuple(tuple(int(z) for z in pair) for pair in payload['examples'])
            heldout=tuple(tuple(int(z) for z in pair) for pair in payload.get('heldout',[]))
            if not 1<=len(examples)<=100 or any(len(p)!=2 for p in examples+heldout):
                raise ValueError('Expected 1..100 public examples, each [integer_x, integer_y]')
            task=Task('custom',examples,heldout,None)
        else:
            task=generate_task(rng,args.family,args.examples)
        model=None
        if args.model and not args.no_neural:
            model,_=restore_checkpoint(args.model)
            model.eval()
        output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
        with PersistentMemory(args.memory or output/'memory.sqlite') as mem, Telemetry(output/'telemetry') as logs:
            solver=SharedMCTS(model,mem,logs,args.depth,args.simulations,seed=args.seed,
                              ablate_communication=args.no_communication,
                              world_model=load_world_model(args.world_model) if args.world_model else None,
                              world_weight=args.world_weight)
            result=solver.solve(task.public(),store=True)
            metrics=eval_one(task,result) if task.heldout else None
            adaptation=None
            if args.adapt_steps and model is not None and result.example_exact and result.program:
                import copy
                checkpoint=copy.deepcopy(model.state_dict())
                adaptation=adapt_to_program(model,task.public(),result.program,steps=args.adapt_steps,persist=True)
                after=SharedMCTS(model,mem,logs,args.depth,args.simulations,
                                 seed=args.seed+1,ablate_communication=args.no_communication).solve(
                                 task.public(),store=False)
                adaptation['after_search']=after.as_dict()
                adaptation['after_hidden_evaluation']=eval_one(task,after) if task.heldout else None
                model.load_state_dict(checkpoint)
            data={'adaptation':adaptation,'task_family' :task.family,'public_examples':task.examples,
                  'expression':pretty(result.program or []),'search':result.as_dict(),
                  'hidden_evaluation':metrics}
            (output/'result.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
            print(json.dumps(data,indent=2))
    elif args.command=='benchmark':
        results=run_benchmark(args.output,args.widths,args.seeds,args.agents,args.steps,
                     args.simulations,args.episodes,args.precision,args.message_dim,
                     args.top_k,args.rounds,args.depth,telemetry_level=args.telemetry)
        if results:plot_summary(Path(args.output)/'summary.csv',Path(args.output)/'comparison.png')
        print(json.dumps(results,indent=2))
    elif args.command=='world-train':
        print(json.dumps(train_world_model(args.output,args.steps,args.seed,args.hidden),indent=2))
    elif args.command=='decimal-demo':
        society=DecimalSociety(args.agents,args.width,args.digits)
        before=society.forward(1,advance=False)
        train_steps=[society.step(1,1) for _ in range(args.steps)]
        after=society.forward(1,advance=False)
        data={'description':'True Decimal weights/messages and manual SGD; slow reference backend',
              'before':vars(before),'steps':train_steps,'after':vars(after),
              '2x2_decimal_product':decimal_matmul_2x2([['0.1','0.2'],['0.3','0.4']],
                                [['0.5','0.6'],['0.7','0.8']],args.digits)}
        path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(data,indent=2),encoding='utf-8')
        print(json.dumps(data,indent=2))
    elif args.command=='demo':
        path=Path(args.output);path.mkdir(parents=True,exist_ok=True)
        torch.manual_seed(2026)
        model=NeuralSociety(args.agents,args.width,8,min(3,args.agents-1),2)
        with PersistentMemory(path/'memory.sqlite') as mem,Telemetry(path/'telemetry') as logs:
            history=train(model,TrainSettings(steps=args.steps,log_every=max(1,args.steps//3),
                            checkpoint_every=0),path,mem,logs)
            rng=random.Random(2027)
            task=sample_task(rng,'ood')
            solver=SharedMCTS(model,mem,logs,max_depth=15,simulations=args.simulations,seed=2027)
            result=solver.solve(task.public(),store=False)
            report={'architecture':model.report(),'training':history,
                    'memory':mem.stats(),'test':eval_one(task,result),'search':result.as_dict()}
            (path/'demo_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
            print(json.dumps(report,indent=2))
    elif args.command=='inspect':
        output={}
        if args.checkpoint:
            model,bundle=restore_checkpoint(args.checkpoint)
            output['architecture']=model.report();output['step']=bundle['step']
        if args.memory:
            with PersistentMemory(args.memory) as mem:
                output['memory']=mem.stats();output['concepts']=mem.concepts()
        print(json.dumps(output,indent=2))
    elif args.command=='adapt':
        model,_=restore_checkpoint(args.checkpoint)
        task=sample_task(random.Random(args.seed),'ood')
        print(json.dumps(adapt_to_examples(model,task.public(),steps=args.steps),indent=2))

if __name__=='__main__':main()
