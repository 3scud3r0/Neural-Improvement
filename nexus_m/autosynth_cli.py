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
    generic=sub.add_parser('function-search',help='MCTS over complete repository function ASTs')
    generic.add_argument('--workspace',type=Path,default=Path('.'))
    generic.add_argument('--target',required=True,help='Python file relative to workspace')
    generic.add_argument('--function',required=True,help='qualified name, e.g. SharedMCTS._select')
    generic.add_argument('--contract',required=True,type=Path,help='JSON test/benchmark commands')
    generic.add_argument('--output',type=Path,default=None)
    generic.add_argument('--simulations',type=int,default=32)
    generic.add_argument('--depth',type=int,default=3)
    generic.add_argument('--max-children',type=int,default=16)
    generic.add_argument('--seed',type=int,default=2026)
    generic.add_argument('--min-gain',type=float,default=.05)
    generic.add_argument('--proposals',type=Path,action='append',default=[],help='complete function file or directory')
    generic.add_argument('--provider',action='append',default=[],help='trusted module:callable AST proposer')
    generic.add_argument('--no-mutations',action='store_true',help='use only supplied proposal providers')
    generic.add_argument('--approve',action=argparse.BooleanOptionalAction,default=None)
    undo=sub.add_parser('function-rollback',help='restore a promoted repository function source file')
    undo.add_argument('--workspace',type=Path,default=Path('.'))
    undo.add_argument('--output',type=Path,required=True)
    undo.add_argument('--approve',action=argparse.BooleanOptionalAction,default=None)
    listing=sub.add_parser('functions',help='list functions and methods in a Python source file')
    listing.add_argument('--workspace',type=Path,default=Path('.'))
    listing.add_argument('--target',required=True)
    args=p.parse_args(argv)
    if args.command=='functions':
        import ast,tokenize
        from .function_ast import function_names
        workspace=args.workspace.resolve()
        path=(workspace/args.target).resolve()
        path.relative_to(workspace)
        with tokenize.open(path) as stream:tree=ast.parse(stream.read())
        print(json.dumps(list(function_names(tree)),indent=2));return 0
    if args.command=='function-rollback':
        from .repository_improver import rollback_function
        print(json.dumps(rollback_function(args.workspace,args.output,approve=args.approve),indent=2))
        return 0
    if args.command=='function-search':
        from .function_ast import SourceProposals,load_provider,mutations
        from .function_evaluator import FunctionContract
        from .repository_improver import RepositoryFunctionImprover
        providers=[]
        for path in args.proposals:
            paths=sorted(path.glob('*.py')) if path.is_dir() else [path]
            if not paths:p.error(f'no proposal files found: {path}')
            providers.append(SourceProposals(paths))
        providers.extend(load_provider(ref) for ref in args.provider)
        if not args.no_mutations:providers.append(mutations)
        if not providers:p.error('supply proposals/providers or enable mutations')
        output=args.output or args.workspace/'runs/function_synthesis'
        engine=RepositoryFunctionImprover(args.workspace,args.target,args.function,
                                         FunctionContract.read(args.contract),output)
        result=engine.run(simulations=args.simulations,max_depth=args.depth,
                          max_children=args.max_children,min_gain=args.min_gain,
                          providers=providers,seed=args.seed,approve=args.approve)
        print(json.dumps(result,indent=2,ensure_ascii=False))
        return 0
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
