"""Executable automatic integer synthesis and validated repository evolution.

The name is historical, not a claim of sovereignty, AGI or a Gödel machine.
Existing pure int code supplies the frozen specification. New intent is NOT
inferred. E-graphs, typed enumeration, scoped proofs, differential cases and
actual benchmarks are connected here; no stub returns a fictional promotion.
"""
from __future__ import annotations
import argparse
import ast
from copy import deepcopy
import json
from pathlib import Path
import random
from .function_ast import FunctionTarget,Proposal,digest
from .function_evaluator import FunctionContract
from .egraph_synthesis import egraph_proposals
from .levin_universal_search import LevinUniversalSearch
from .oracle_coevolution import AdversarialOracle,outcome
from .repository_improver import RepositoryFunctionImprover
from .repository_transaction import RepositoryTransaction
from .typed_program import typed_function
from .whole_program_pdg import ProgramDependencyGraph


DRIVER = '''import hashlib,importlib,json,os,statistics,sys,time
from pathlib import Path
data=Path(__file__).with_name('reference.json').read_bytes()
if hashlib.sha256(data).hexdigest()!=REFERENCE_HASH: raise RuntimeError('frozen reference hash mismatch')
config=json.loads(data)
functions=[]
for target in config['targets']:
    function=importlib.import_module(target['module'])
    for part in target['qualified_name'].split('.'): function=getattr(function,part)
    if Path(function.__code__.co_filename).resolve()!=Path(target['relative_path']).resolve():
        raise RuntimeError('import resolved outside the copied target module')
    if function.__code__.co_name!=target['qualified_name'].split('.')[-1]:
        raise RuntimeError('runtime binding differs from the selected function')
    functions.append((target,function))
mode=sys.argv[1]
if mode!='benchmark':
    for target,function in functions:
        for case in target[mode]:
            try:
                value=function(*[case['bindings'][name] for name in target['variables']])
                if type(value) is not int: raise AssertionError('non-int return')
                observed=['value',value]
            except (ZeroDivisionError,OverflowError) as exc: observed=['error',type(exc).__name__]
            if observed!=case['expected']: raise AssertionError((case['bindings'],case['expected'],observed))
else:
    calls=[]
    for target,function in functions:
        for case in target['search'][:32]:
            if case['expected'][0]=='value': calls.append((function,tuple(case['bindings'][v] for v in target['variables'])))
    if not calls: raise RuntimeError('no defined cases to benchmark')
    for function,args in calls: function(*args)
    samples=[]
    for _ in range(5):
        start=time.process_time()
        for index in range(config['benchmark_iterations']):
            function,args=calls[index%len(calls)];function(*args)
        samples.append(time.process_time()-start)
    Path(os.environ['NEXUS_BENCHMARK_FILE']).write_text(json.dumps({'seconds':statistics.median(samples)}))
'''


def automatic_contract(workspace,targets,output,*,seed=2026,cases=96,benchmark_iterations=5000):
    """Generate a repeatable reference oracle, not a specification from nothing."""
    if cases<1 or benchmark_iterations<1:raise ValueError('positive automatic evaluation budgets required')
    oracle=AdversarialOracle(seed);specs=[]
    for target in targets:
        variables,reference=typed_function(target.function)
        if reference.nodes>512:raise ValueError('automatic reference AST budget exceeded')
        relative=Path(target.relative_path).with_suffix('')
        parts=list(relative.parts)
        if parts[-1]=='__init__':parts.pop()
        module='.'.join(parts) or '__init__'
        spec=dict(module=module,qualified_name=target.qualified_name,relative_path=target.relative_path,
                  variables=variables,original_sha256=digest(target.original))
        for mode,offset in (('search',0),('heldout',1)):
            spec[mode]=[dict(bindings=b,expected=outcome(reference,b,max_bits=65536))
                        for b in oracle.cases(variables,count=cases,seed=seed+offset)]
        specs.append(spec)
    output=Path(output).resolve()/'oracle';output.mkdir(parents=True,exist_ok=True)
    data=json.dumps(dict(targets=specs,benchmark_iterations=benchmark_iterations),sort_keys=True).encode()
    reference_path=output/'reference.json';reference_path.write_bytes(data)
    driver=output/'driver.py'
    driver.write_text('REFERENCE_HASH='+repr(digest(data))+'\n'+DRIVER,encoding='utf-8')
    contract=FunctionContract(tests=[['{python}',str(driver),'search']],
        heldout=[['{python}',str(driver),'heldout']],benchmark=['{python}',str(driver),'benchmark'])
    return contract,dict(reference_sha256=digest(data),reference_is_specification=True,
                         intent_inferred=False,path=str(reference_path))


class SovereignImprover:
    def __init__(self,workspace,*,seed=2026):
        self.workspace=Path(workspace).resolve();self.seed=seed
        self.pdg=ProgramDependencyGraph(self.workspace);self.oracle=AdversarialOracle(seed)

    def _target(self,symbol):
        name=self.pdg.resolve_symbol(symbol);info=self.pdg.symbols[name]
        return FunctionTarget.read(self.workspace,str(info.path.relative_to(self.workspace)),info.qualified_name)

    def _proposals(self,target,*,max_fuel,backend,device):
        variables,reference=typed_function(target.function)
        proposals=list(egraph_proposals(target.function,random.Random(self.seed)))
        search=LevinUniversalSearch(max_depth=7,variables=variables,max_programs=2048)
        examples=[(b,outcome(reference,b,max_bits=256)) for b in self.oracle.cases(variables,count=32)]
        if backend=='vectorized':
            defined=[(b,value) for b,value in examples if value[0]=='value' and -(2**63)<=value[1]<2**63]
            solution=search.search_batched(defined,device=device,max_programs=2048,max_instruction_slots=max_fuel)
        else:solution=search.search(examples,max_fuel=max_fuel,wall_seconds=5.)
        if solution:
            node=deepcopy(target.function);node.body=[ast.Return(value=solution.expression.to_ast())]
            proposals.append(Proposal('typed-'+solution.backend,ast.fix_missing_locations(node)))
        valid=[];proofs=[];seen=set()
        for proposed in proposals:
            _,candidate=typed_function(proposed.function)
            if candidate==reference or candidate in seen:continue
            seen.add(candidate)
            proof=self.oracle.compare(reference,candidate,variables)
            proofs.append(dict(proposal=proposed.label,expression=candidate.source(),evidence=proof.as_dict()))
            if proof.proved:valid.append(proposed)
        return valid,proofs,vars(search.last_statistics)

    def self_improve_cycle(self,target_function_name,*,output=None,contract=None,
                          max_fuel=20_000,simulations=4,min_gain=.05,approve=None,
                          backend='scalar',device='auto'):
        self.pdg.build_graph();target=self._target(target_function_name)
        output=Path(output or self.workspace/'runs/sovereign').resolve()
        try:
            proposals,proofs,search=self._proposals(target,max_fuel=max_fuel,backend=backend,device=device)
        except ValueError as exc:return dict(status='unsupported_auto_domain',promoted=False,reason=str(exc))
        output.mkdir(parents=True,exist_ok=True)
        reference_info=None
        if contract is None:contract,reference_info=automatic_contract(self.workspace,[target],output,seed=self.seed)
        report=dict(status='no_improvement',promoted=False,proofs=proofs,typed_search=search,
                    affected=self.pdg.impact_report(target_function_name),reference=reference_info)
        if proposals:
            def provider(function,rng):
                for proposal in proposals:yield deepcopy(proposal)
            engine=RepositoryFunctionImprover(self.workspace,target.relative_path,target.qualified_name,contract,output)
            def reward(original,candidate,gain,passed):
                return self.oracle.calculate_mdl_reward(ast.unparse(original),ast.unparse(candidate),gain,passed)
            result=engine.run(simulations=simulations,max_depth=2,min_gain=min_gain,approve=approve,
                              providers=[provider],seed=self.seed,reward_fn=reward)
            report.update(status='promoted' if result['promoted'] else 'not_promoted',promoted=result['promoted'],evaluation=result)
        (output/'sovereign_report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
        return report

    def self_improve_bundle(self,symbols,*,output,contract=None,max_fuel=20_000,min_gain=.05,
                            approve=None,backend='scalar',device='auto'):
        self.pdg.build_graph();patches={};targets=[];proofs=[]
        for symbol in symbols:
            target=self._target(symbol)
            proposals,evidence,_=self._proposals(target,max_fuel=max_fuel,backend=backend,device=device)
            targets.append(target);proofs.extend(evidence)
            if proposals:
                best=min(proposals,key=lambda proposal:len(list(ast.walk(proposal.function))))
                patches[self.pdg.resolve_symbol(symbol)]=ast.unparse(best.function)
        if not patches:return dict(status='no_improvement',promoted=False,proofs=proofs)
        files=self.pdg.create_multi_file_transaction(patches)
        reference_info=None
        if contract is None:contract,reference_info=automatic_contract(self.workspace,targets,output,seed=self.seed)
        transaction=RepositoryTransaction(self.workspace,output)
        result=transaction.validate_and_apply(files,contract,approve=approve,min_gain=min_gain)
        report=dict(status=result['status'],promoted=result['promoted'],proofs=proofs,
                    evaluation=result,reference=reference_info,
                    affected=[self.pdg.impact_report(name) for name in symbols])
        (Path(output)/'sovereign_report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
        return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    improve=sub.add_parser('improve')
    improve.add_argument('--workspace',type=Path,default=Path('.'))
    improve.add_argument('--symbol',action='append',required=True)
    improve.add_argument('--output',type=Path,default=Path('runs/sovereign'))
    improve.add_argument('--contract',type=Path)
    improve.add_argument('--max-fuel',type=int,default=20_000)
    improve.add_argument('--simulations',type=int,default=4)
    improve.add_argument('--min-gain',type=float,default=.05)
    improve.add_argument('--backend',choices=('scalar','vectorized'),default='scalar')
    improve.add_argument('--device',default='auto')
    improve.add_argument('--approve',action=argparse.BooleanOptionalAction,default=None)
    impact=sub.add_parser('impact');impact.add_argument('--workspace',type=Path,default=Path('.'));impact.add_argument('--symbol',required=True)
    restore=sub.add_parser('recover');restore.add_argument('--workspace',type=Path,default=Path('.'));restore.add_argument('--output',required=True,type=Path)
    args=parser.parse_args(argv)
    if args.command=='recover':result=RepositoryTransaction(args.workspace,args.output).recover()
    elif args.command=='impact':result=ProgramDependencyGraph(args.workspace).impact_report(args.symbol)
    else:
        engine=SovereignImprover(args.workspace)
        options=dict(output=args.output,contract=FunctionContract.read(args.contract) if args.contract else None,
                     max_fuel=args.max_fuel,min_gain=args.min_gain,approve=args.approve,backend=args.backend,device=args.device)
        result=(engine.self_improve_cycle(args.symbol[0],simulations=args.simulations,**options) if len(args.symbol)==1
                else engine.self_improve_bundle(args.symbol,**options))
    print(json.dumps(result,indent=2));return 0


if __name__=='__main__':raise SystemExit(main())
