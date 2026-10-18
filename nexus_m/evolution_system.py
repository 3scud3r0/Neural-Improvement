"""Orchestration layer connecting the seven research modules to NEXUS-M.

Designed for testable algorithmic research, not autonomous real-world action.
All demo experiments use synthetic/in-memory observations. No network calls.
"""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import json,random,time
import numpy as np
import torch
from .model import NeuralSociety
from .features import encode
from .tasks import Task
from .grammar import TOKEN_TO_ID
from .universal_compute import (BudgetedPython,ExecutionBudget,
    bounded_equivalence,prove_polynomial_identity)
from .hierarchical_abstraction_search import HierarchicalMCTS,evaluate_macro_program
from .causal_inference_engine import (CausalGraph,BinarySCM,infer_linear_damped_motion)
from .conceptual_knowledge_graph import ConceptualKnowledgeGraph,seed_cross_domain_example
from .dynamic_topology import DynamicTopology,TopologyLimits
from .recursive_self_improvement import ManagedSelfImprover
from .latent_world_model import LatentDynamics,train_latent_model
from .meta_learning import distill_macro_program
from .alignment_and_value_core import AlignmentGuard,ResourcePolicy

_MACRO_TOKEN={'ADD_1':'ADD','ADD_2':'ADD','ADD_3':'ADD',
              'SUB_1':'SUB','SUB_2':'SUB','MUL_2':'MUL','MUL_3':'MUL',
              'NEG':'NEG','ABS':'ABS','SQUARE':'SQUARE','HALF_EVEN':'HALF'}


class EvolutionSystem:
    def __init__(self,workspace:str|Path,agents:int=8,width:int=32,
                 policy:ResourcePolicy|None=None,seed:int=2026):
        self.workspace=Path(workspace)
        self.workspace.mkdir(parents=True,exist_ok=True)
        self.guard=AlignmentGuard(self.workspace,policy)
        torch.manual_seed(seed)
        self.topology=DynamicTopology(NeuralSociety(agents=agents,width=width,
                                  top_k=min(4,agents-1)),
                TopologyLimits(max_agents=self.guard.policy.max_agents,
                               max_width=self.guard.policy.max_neurons_per_agent,
                               max_parameters=self.guard.policy.max_parameters))
        stats=self.topology.model.report()
        if not self.guard.authorize('topology',{'agents':agents,'width':width,
                                 'parameters':stats['trainable_parameters']}).allowed:
            raise ValueError('initial neural architecture rejected')
        self.graph=ConceptualKnowledgeGraph(self.workspace/'knowledge.sqlite')
        self.improver=ManagedSelfImprover(self.guard)
        self.seed=seed
        self.latent_model=None
        self.events=self.workspace/'events.jsonl'

    def generate_text(self,prompt:str,model_path:str|Path,**generation)->str:
        """Run the optional NeuroOS pipeline; symbolic solve() remains independent."""
        from .neuroos_bridge import NeuroOSBridge
        if not hasattr(self,'_neuroos'):
            self._neuroos=NeuroOSBridge(model_path,self.workspace/'neuroos')
        elif self._neuroos.model_path!=Path(model_path).resolve():
            raise ValueError('close this system before switching GGUF models')
        response=self._neuroos.generate(prompt,**generation)
        self.record('neuroos_generation',{'prompt_length':len(prompt),'response_length':len(response)})
        return response

    def close(self):
        try:
            if hasattr(self,'_neuroos'):self._neuroos.close()
        finally:self.graph.close()
    def __enter__(self):return self
    def __exit__(self,*args):self.close()

    def record(self,kind:str,payload:dict):
        with self.events.open('a',encoding='utf-8') as f:
            f.write(json.dumps({'timestamp':time.time(),'kind':kind,'payload':payload},
                               ensure_ascii=False,default=str)+'\n')

    def _neural_prior(self,examples:list[tuple[int,int]]):
        """Untrained neural proposals: explicitly compared against uniform search."""
        task=Task('unlabeled',tuple(examples),(),None)
        model=self.topology.model
        with torch.no_grad():
            tensor=encode(task.public(),device=next(model.parameters()).device,
                          dtype=next(model.parameters()).dtype)
            output=model(tensor)
            logits=output.logits.detach().float().cpu().numpy()
        def prior(state:tuple[int,...],available:list[str]):
            if not available:return {}
            raw=np.array([float(logits[TOKEN_TO_ID[_MACRO_TOKEN[name]]])
                          for name in available],dtype=float)
            raw-=raw.max();p=np.exp(raw);p/=p.sum()
            return dict(zip(available,(0.2/len(available)+0.8*p).tolist()))
        return prior

    def _remember_solution(self,program:list[str],examples:list[tuple[int,int]],origin:str):
        import hashlib
        canonical=json.dumps(program)
        ident='program:'+hashlib.sha256(canonical.encode()).hexdigest()[:20]
        self.graph.add_concept(ident,'macro_program',canonical,
             attributes={'program':program,'examples':examples,'verified_scope':'training_examples_only'},
             provenance=origin,verified=False)
        return ident

    def _retrieve_valid_program(self,examples):
        rows=self.graph.connection.execute('''SELECT id,attributes FROM concepts
                                                  WHERE type='macro_program' LIMIT 300''').fetchall()
        for id,attrs in rows:
            program=json.loads(attrs)['program']
            predictions=evaluate_macro_program(program,[x for x,_ in examples])
            if predictions==[y for _,y in examples]:return id,program
        return None,None

    def solve(self,examples:list[tuple[int,int]],simulations:int=500,
              max_depth:int=6,use_neural:bool=True,use_memory:bool=True,
              heldout_inputs:list[int]|None=None,use_latent:bool=False)->dict:
        approval=self.guard.authorize('mcts',{'simulations':simulations,'depth':max_depth})
        if not approval.allowed:raise PermissionError(approval.reason)
        if not 1<=len(examples)<=100:raise ValueError('1..100 examples required')
        examples=[(int(x),int(y)) for x,y in examples]
        source='hierarchical_search'
        memory_id=None
        program=None
        if use_memory:
            memory_id,program=self._retrieve_valid_program(examples)
            if program is not None:source='knowledge_graph_retrieval'
        search=None
        if program is None:
            neural_prior=self._neural_prior(examples) if use_neural else None
            target=tuple(y for _,y in examples)
            def combined_prior(state,actions):
                latent=self.latent_model.one_step_score(state,target,actions) if (
                    use_latent and self.latent_model is not None) else None
                if latent is None:
                    return neural_prior(state,actions) if neural_prior is not None else {}
                if neural_prior is None:return latent
                neural=neural_prior(state,actions)
                return {a:0.5*latent[a]+0.5*neural[a] for a in actions}
            searcher=HierarchicalMCTS(simulations=simulations,max_depth=max_depth,
                         prior=combined_prior if (use_neural or use_latent) else None,
                         seed=self.seed)
            search=searcher.search(examples)
            if search.found:
                program=search.macro_program
                if use_memory:memory_id=self._remember_solution(program,examples,'search_exact_on_examples')
        predictions=(evaluate_macro_program(program,heldout_inputs)
                      if program is not None and heldout_inputs else None)
        result={'found_on_examples':program is not None,'program':program,'origin':source,
                'retrieved_id':memory_id,'searched':asdict(search) if search else None,
                'heldout_predictions_unscored':predictions,
                'generalization_proven':False,
                'neural_policy':'neural_weights' if use_neural else 'uniform',
                'latent_model_used':bool(use_latent and self.latent_model is not None)}
        self.record('solve',result)
        return result

    def distill(self,examples:list[tuple[int,int]],program:list[str],steps=20,persist=True)->dict:
        result=distill_macro_program(self.topology.model,examples,program,steps,persist=persist)
        report=result.as_dict()
        self.record('distillation',report)
        if persist:
            torch.save({'architecture':self.topology.model.report(),
                        'state_dict':self.topology.model.state_dict()},
                        self.workspace/'neural_distilled_weights.pt')
        return report

    def train_latent(self,steps=100)->dict:
        self.latent_model=LatentDynamics()
        report=train_latent_model(self.latent_model,steps=steps,seed=self.seed).as_dict()
        torch.save({'state_dict':self.latent_model.state_dict(),
                    'macros':self.latent_model.names},self.workspace/'latent_weights.pt')
        self.record('latent_training',report)
        return report

    def demonstrate_causality(self):
        graph=CausalGraph([('U','X'),('U','Y'),('X','M'),('M','Y')],latent={'U'})
        scm=BinarySCM(graph,{
          'U':lambda p,n:n,
          'X':lambda p,n:p['U']^n,
          'M':lambda p,n:p['X']^n,
          'Y':lambda p,n:(p['M'] or p['U'])^n,
        },{'U':.5,'X':.15,'M':.1,'Y':.07})
        exact=scm.do_effect('X','Y',1)
        front=scm.identify_frontdoor('X','Y','M',1)
        naïve=scm.probability({'Y':1},{'X':1})
        result={'intervention':exact.as_dict(),'frontdoor':front.as_dict(),
                'observational_conditional':naïve,
                'absolute_error_frontdoor':abs(exact.value-front.value),
                'backdoor_sets':graph.backdoor_sets('X','Y')}
        self.record('causality',result)
        return result

    def demonstrate_physics(self):
        rng=np.random.default_rng(self.seed)
        x=rng.uniform(-3,3,120);v=rng.uniform(-2,2,120)
        acceleration=-2.4*x-0.3*v+rng.normal(0,0.025,120)
        report=asdict(infer_linear_damped_motion(x,v,acceleration,self.seed))
        self.record('physics',report)
        return report

    def demonstrate_analogy(self):
        analogies=seed_cross_domain_example(self.graph)
        for a in analogies:self.graph.preserve_analogy(a,'synthetic matching structure; unverified transfer')
        report={'count':len(analogies),'candidates':[a.as_dict() for a in analogies[:3]],
                'graph':self.graph.stats()}
        self.record('analogies',report)
        return report

    def demonstrate_universal_compute(self):
        factorial='''def main(x):\n    n = x\n    result = 1\n    while n > 1:\n        result *= n\n        n -= 1\n    return result\n'''
        engine=BudgetedPython(ExecutionBudget(fuel=2000,max_bits=4096))
        report={'factorial_6':asdict(engine.run(factorial,6)),
                'symbolic_proof':asdict(prove_polynomial_identity('(x+1)**2','x**2+2*x+1')),
                'bounded_equivalence':asdict(bounded_equivalence('def main(x):\n    return x + x',
                  'def main(x):\n    return 2 * x',-25,25))}
        self.record('compute',report)
        return report

    def demonstrate_topology(self):
        before=self.topology.model.report()
        self.topology.recruit(2)
        self.topology.widen(8)
        self.topology.prune([0])
        report={'before':before,'after':self.topology.model.report(),
                'events':[asdict(e) for e in self.topology.history],
                'cloud_proposal':self.topology.cloud_capacity_request(100,10000)}
        self.record('topology',report)
        return report

    def demo(self,simulations=1200,approve_code_changes=None):
        result={
          'compute':self.demonstrate_universal_compute(),
          'causal':self.demonstrate_causality(),
          'physics':self.demonstrate_physics(),
          'analogies':self.demonstrate_analogy(),
        }
        # First solving episode populates memory; second checks verified reuse.
        samples=[(-4,-5),(-1,1),(0,3),(2,7),(5,13)] # 2*x+3
        result['first_search']=self.solve(samples,simulations=simulations,max_depth=6,
                                           heldout_inputs=[-30,11,100])
        result['distillation']=self.distill(samples,result['first_search']['program'],steps=8) if result['first_search']['program'] else None
        result['latent_training']=self.train_latent(steps=80)
        result['latent_search']=self.solve(samples,simulations=simulations,max_depth=6,
                                            use_neural=False,use_memory=False,use_latent=True)
        result['memory_replay']=self.solve(samples,simulations=25,max_depth=6,
                                            heldout_inputs=[-30,11,100])
        # Same budget, same data, but no neural priors and no memory.
        result['uniform_control']=self.solve(samples,simulations=simulations,max_depth=6,
                                            use_neural=False,use_memory=False)
        result['topology']=self.demonstrate_topology()
        result['improvement']=[r.as_dict() for r in self.improver.optimize(1,approve=approve_code_changes)]
        result['guard']=self.guard.invariant_report()
        result['knowledge_graph']=self.graph.stats()
        report=self.workspace/'demo_result.json'
        report.write_text(json.dumps(result,indent=2,ensure_ascii=False,default=str),encoding='utf-8')
        return result
