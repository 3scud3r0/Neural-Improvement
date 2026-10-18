"""NEXUS-Evolution experimental command line."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from .evolution_system import EvolutionSystem
from .universal_compute import BudgetedPython,ExecutionBudget


def main(argv=None):
    p=argparse.ArgumentParser(description='NEXUS Evolution Lab: seven bounded research modules')
    sub=p.add_subparsers(dest='command',required=True)
    demo=sub.add_parser('demo',help='Run integrated controlled demonstration')
    demo.add_argument('--output',default='runs/evolution_demo')
    demo.add_argument('--agents',type=int,default=8)
    demo.add_argument('--width',type=int,default=32)
    demo.add_argument('--simulations',type=int,default=1500)
    demo.add_argument('--approve-code-change',action='store_true')
    solve=sub.add_parser('search',help='Hierarchical program synthesis on public examples')
    solve.add_argument('--examples',required=True,help='JSON file of input/output pairs')
    solve.add_argument('--output',default='runs/evolution_search')
    solve.add_argument('--simulations',type=int,default=1500)
    solve.add_argument('--depth',type=int,default=6)
    solve.add_argument('--uniform',action='store_true')
    optimize=sub.add_parser('improve',help='Propose and benchmark a reviewed source change')
    optimize.add_argument('--output',default='runs/evolution_improvement')
    optimize.add_argument('--approve',action='store_true')
    opt_run=sub.add_parser('run-subset',help='Execute Python integer subset in strict budget')
    opt_run.add_argument('--source',required=True,help='Path to Python-subset source file')
    opt_run.add_argument('--input',required=True,type=int)
    opt_run.add_argument('--fuel',default=20000,type=int)
    args=p.parse_args(argv)
    if args.command=='run-subset':
        result=BudgetedPython(ExecutionBudget(fuel=args.fuel)).run(Path(args.source).read_text(),args.input)
    else:
        with EvolutionSystem(args.output,getattr(args,'agents',8),getattr(args,'width',32)) as system:
            if args.command=='demo':result=system.demo(args.simulations,args.approve_code_change)
            elif args.command=='search':
                payload=json.loads(Path(args.examples).read_text(encoding='utf-8'))
                result=system.solve([tuple(pair) for pair in payload['examples']],
                       args.simulations,args.depth,not args.uniform,
                       heldout_inputs=payload.get('unlabeled_inputs'))
            elif args.command=='improve':result=system.improver.step(approve=args.approve).as_dict()
    print(json.dumps(result,indent=2,ensure_ascii=False,default=str))
    return 0

if __name__=='__main__':raise SystemExit(main())
