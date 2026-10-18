"""CLI for bounded, LLM-free source synthesis and optimization experiments."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from .alignment_and_value_core import AlignmentGuard,ResourcePolicy
from .search_driven_improver import SearchDrivenImprover
from .ast_program_synthesis import RouterSpec,compile_source
from .ast_formal_gate import verify
from .ast_performance_profiler import profile
from .capacity_planner import plan


def main(argv=None):
    p=argparse.ArgumentParser(description='NEXUS AutoSynth: MCTS AST synthesis without an LLM')
    sub=p.add_subparsers(dest='command',required=True)
    s=sub.add_parser('search',help='discover improved managed router with MCTS')
    s.add_argument('--output',default='runs/autosynth')
    s.add_argument('--simulations',type=int,default=96)
    s.add_argument('--seed',type=int,default=2026)
    s.add_argument('--min-gain',type=float,default=.05)
    s.add_argument('--approve',action=argparse.BooleanOptionalAction,default=None,help='automatic by default; --no-approve stages only')
    s.add_argument('--target',default='managed/ast_router.py')
    prof=sub.add_parser('profile',help='profile bottlenecks in the managed router')
    prof.add_argument('--output',default='runs/autosynth')
    prof.add_argument('--iterations',type=int,default=100)
    cap=sub.add_parser('capacity',help='plan compute and sharding WITHOUT provisioning cloud')
    cap.add_argument('--agents',type=int,default=100)
    cap.add_argument('--width',type=int,default=10000)
    cap.add_argument('--nodes',type=int,default=100000)
    cap.add_argument('--simulations',type=int,default=100000)
    cap.add_argument('--workers',type=int,default=4)
    r=sub.add_parser('rollback',help='restore previous managed router')
    r.add_argument('--output',default='runs/autosynth')
    r.add_argument('--approve',action=argparse.BooleanOptionalAction,default=None)
    r.add_argument('--target',default='managed/ast_router.py')
    v=sub.add_parser('verify',help='finite exhaustive verification + structural termination check')
    v.add_argument('--strategy',choices=('sort','heap','scan','reverse_sort'),default='heap')
    v.add_argument('--key',choices=('stable','score','index'),default='stable')
    v.add_argument('--validation',choices=('strict','omitted'),default='strict')
    args=p.parse_args(argv)
    if args.command=='capacity':
        print(json.dumps(plan(args.agents,args.width,args.nodes,
                              args.simulations,args.workers).as_dict(),indent=2));return 0
    if args.command=='verify':
        spec=RouterSpec(args.strategy,args.key,args.validation)
        print(json.dumps({'source':compile_source(spec),'evidence':verify(spec).as_dict()},
                         indent=2,ensure_ascii=False));return 0
    guard=AlignmentGuard(Path(args.output),ResourcePolicy())
    improver=SearchDrivenImprover(guard,min_gain=getattr(args,'min_gain',.05),
                                 target_path=getattr(args,'target','managed/ast_router.py'))
    if args.command=='profile':
        print(json.dumps(profile(improver,iterations=args.iterations).as_dict(),
                         ensure_ascii=False,indent=2));return 0
    if args.command=='search':
        report=improver.run(approve=args.approve,simulations=args.simulations,
                            seed=args.seed)
        print(json.dumps(report.as_dict(),ensure_ascii=False,indent=2));return 0
    ok=improver.rollback(approve=args.approve)
    print(json.dumps({'rolled_back':ok,'active':str(improver.active)}));return 0

if __name__=='__main__':raise SystemExit(main())
