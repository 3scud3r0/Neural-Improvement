"""Functional, negative and integration tests for seven new modules."""
import json,math,hashlib
from pathlib import Path
import pytest
import torch
import numpy as np
from nexus_m.universal_compute import (BudgetedPython,ExecutionBudget,
    prove_polynomial_identity,bounded_equivalence,optional_compiler_check,
    optional_proof_assistant_check)
from nexus_m.hierarchical_abstraction_search import (HierarchicalMCTS,
    evaluate_macro_program,GoalGraph,Subgoal)
from nexus_m.causal_inference_engine import (CausalGraph,BinarySCM,infer_linear_damped_motion)
from nexus_m.conceptual_knowledge_graph import ConceptualKnowledgeGraph,seed_cross_domain_example
from nexus_m.dynamic_topology import DynamicTopology,TopologyLimits
from nexus_m.alignment_and_value_core import AlignmentGuard,ResourcePolicy
from nexus_m.recursive_self_improvement import ManagedSelfImprover,HEAP_TEMPLATE
from nexus_m.evolution_system import EvolutionSystem
from nexus_m.model import NeuralSociety
from nexus_m.features import FEATURE_DIM


def test_budgeted_factorial():
    src='def main(x):\n    out = 1\n    while x > 1:\n        out *= x\n        x -= 1\n    return out\n'
    interp=BudgetedPython()
    assert interp.run(src,6).result==720
    assert interp.run(src,1).result==1


def test_budget_rejects_infinite_loop():
    assert not BudgetedPython(ExecutionBudget(fuel=100)).run(
        'def main(x):\n    while x >= 0:\n        x += 1\n    return x',1).completed

@pytest.mark.parametrize('payload',["import os\ndef main(x): return x",
          "def main(x):\n    return __import__('os').system('echo no')",
          'def main(x):\n    return x.__class__',
          'def main(x):\n    return [x]'])
def test_budget_rejects_unsafe_constructs(payload):
    assert not BudgetedPython().run(payload,4).completed


def test_polynomial_proof_and_bounded_counterexample():
    assert prove_polynomial_identity('(x+2)*(x+2)','x**2+4*x+4').status=='proved'
    assert prove_polynomial_identity('x*x','x+1').status=='counterexample'
    assert prove_polynomial_identity("__import__('os').system('no')",'x').status=='unknown'
    a='def main(x):\n    return x*x'
    b='def main(x):\n    return x+x'
    assert bounded_equivalence(a,b,-4,4).status=='counterexample'


def test_compiler_and_prover_are_opt_in():
    assert not optional_compiler_check('cpp','int main(){}')['available']
    assert not optional_proof_assistant_check('lean','theorem yes : True := by trivial')['available']
    assert optional_compiler_check('python','x = 3')['syntax_ok']


def test_macro_mcts_finds_program_and_holds_out():
    examples=[(x,2*x+3) for x in (-4,-1,0,2,5)]
    result=HierarchicalMCTS(simulations=700,max_depth=6,seed=2026).search(examples)
    assert result.found,result
    assert evaluate_macro_program(result.macro_program,[-30,50])==[-57,103]
    assert result.nodes>1


def test_goal_dependency_graph():
    g=GoalGraph();g.add(Subgoal('hypothesis','Propose a rule'))
    g.add(Subgoal('test','Verify it',('hypothesis',)))
    assert [a.identifier for a in g.ready()]==['hypothesis']
    with pytest.raises(ValueError):g.complete('test')
    g.complete('hypothesis'); assert [a.identifier for a in g.ready()]==['test']


def test_frontdoor_identification():
    g=CausalGraph([('U','X'),('U','Y'),('X','M'),('M','Y')],latent={'U'})
    scm=BinarySCM(g,{'U':lambda p,n:n,'X':lambda p,n:p['U']^n,
                     'M':lambda p,n:p['X']^n,'Y':lambda p,n:(p['M'] or p['U'])^n},
                  {'U':.5,'X':.15,'M':.1,'Y':.07})
    assert g.frontdoor_valid('X','Y','M')
    assert not g.backdoor_sets('X','Y')
    for value in (0,1):
        assert scm.identify_frontdoor('X','Y','M',value).value == pytest.approx(scm.do_effect('X','Y',value).value)


def test_backdoor_adjustment():
    g=CausalGraph([('Z','X'),('Z','Y'),('X','Y')])
    assert g.backdoor_valid('X','Y',{'Z'})
    assert not g.backdoor_valid('X','Y',set())
    scm=BinarySCM(g,{'Z':lambda p,n:n,'X':lambda p,n:p['Z']^n,
                     'Y':lambda p,n:(p['X'] or p['Z'])^n},
                  {'Z':.5,'X':.2,'Y':.1})
    assert scm.identify_backdoor('X','Y',1).value==pytest.approx(scm.do_effect('X','Y',1).value)


def test_physics_generalization():
    rng=np.random.default_rng(5)
    x=rng.normal(size=200);v=rng.normal(size=200)
    acceleration=-4*x-.7*v+rng.normal(scale=.03,size=len(x))
    fit=infer_linear_damped_motion(x,v,acceleration)
    assert fit.heldout_r2>.999
    assert fit.coefficients['position_coefficient']==pytest.approx(-4,abs=.03)


def test_knowledge_graph_analogies_are_not_verified(tmp_path):
    with ConceptualKnowledgeGraph(tmp_path/'graph.sqlite') as graph:
        matches=seed_cross_domain_example(graph)
        assert matches
        assert all(m.status=='unverified_hypothesis' for m in matches)
        graph.preserve_analogy(matches[0])
        assert graph.stats()['concepts']==6
        assert graph.stats()['analogies']==1
        assert graph.lexical_search('electric')
        graph.register_artifact('img1','dataset://image1','image/png','f'*64,
                                'electric:voltage','synthetic')
        assert graph.stats()['artifacts']==1
    with ConceptualKnowledgeGraph(tmp_path/'graph.sqlite') as graph:
        assert graph.stats()['concepts']==6


def test_topology_growth_preserves_existing_weights_and_pruning():
    torch.manual_seed(3)
    model=NeuralSociety(agents=6,width=16,top_k=2)
    old=model.agent_bias.detach().clone()
    topo=DynamicTopology(model)
    topo.recruit(2)
    assert topo.model.agents==8
    assert torch.equal(topo.model.agent_bias[:6],old)
    topo.widen(8)
    assert topo.model.width==24
    assert torch.equal(topo.model.agent_bias[:6,:16],old)
    topo.prune([1])
    assert topo.model.agents==7
    assert torch.equal(topo.model.agent_bias[0,:16],old[0])
    assert torch.equal(topo.model.agent_bias[1,:16],old[2])
    assert topo.model(torch.zeros(FEATURE_DIM)).states.shape==(7,24)


def test_topology_budget_denial():
    topo=DynamicTopology(NeuralSociety(agents=4,width=16,top_k=2),
                         TopologyLimits(max_agents=5))
    with pytest.raises(ValueError):topo.recruit(3)
    assert topo.model.agents==4
    proposal=topo.cloud_capacity_request(100,10000)
    assert proposal['proposal_only'] and proposal['status']=='not_provisioned'


def test_restricted_policy_denies_sensitive_actions_and_prevents_escape(tmp_path):
    g=AlignmentGuard(tmp_path,ResourcePolicy.restricted())
    assert not g.authorize('cloud_provision',{},approved=True).allowed
    assert not g.authorize('network',{},approved=True).allowed
    assert not g.authorize('source_edit',{'relative_path':'../outside.py'},approved=True).allowed
    assert not g.authorize('source_edit',{'relative_path':'nexus_m/model.py'},approved=True).allowed
    assert not g.authorize('source_edit',{'relative_path':'managed/router.py'}).allowed
    assert g.authorize('source_edit',{'relative_path':'managed/router.py'},approved=True).allowed
    assert g.verify_ledger()
    assert 'human value alignment' in g.invariant_report()['not_guaranteed']


def test_audit_tampering_detected(tmp_path):
    guard=AlignmentGuard(tmp_path)
    guard.authorize('mcts',{'simulations':10,'depth':6})
    with guard.audit.open('a') as f:f.write('{"tampered":true}\n')
    with pytest.raises(Exception):AlignmentGuard(tmp_path)


def test_managed_self_improvement_needs_approval(tmp_path):
    g=AlignmentGuard(tmp_path)
    engine=ManagedSelfImprover(g,min_speedup=-1)
    initial=engine.path.read_text()
    proposal=engine.step(approve=False,batches=50)
    assert proposal.equivalent and not proposal.promoted
    assert engine.path.read_text()==initial
    applied=engine.step(approve=True,batches=50)
    assert applied.promoted and engine.path.read_text()==HEAP_TEMPLATE
    assert engine.route([1.,3.,2.],2)==[1,2]
    assert g.verify_ledger()


def test_system_end_to_end_and_reusable_memory(tmp_path):
    with EvolutionSystem(tmp_path,agents=5,width=16) as system:
        samples=[(x,2*x+3) for x in (-4,-1,0,2,5)]
        first=system.solve(samples,simulations=500,max_depth=6)
        assert first['found_on_examples']
        again=system.solve(samples,simulations=1,max_depth=6)
        assert again['origin']=='knowledge_graph_retrieval'
        assert system.demonstrate_causality()['absolute_error_frontdoor']<1e-9
        assert system.demonstrate_universal_compute()['factorial_6']['result']==720
        assert system.guard.verify_ledger()
    with EvolutionSystem(tmp_path,agents=5,width=16) as system:
        assert system.solve(samples,simulations=1,max_depth=6)['origin']=='knowledge_graph_retrieval'


def test_latent_predictions_and_training():
    from nexus_m.latent_world_model import LatentDynamics,train_latent_model
    model=LatentDynamics()
    report=train_latent_model(model,steps=35,batch_size=16,seed=10)
    assert math.isfinite(report.loss_final)
    assert report.steps==35
    result=model.one_step_score((1,2,3),(3,4,5),['ADD_2','MUL_2'])
    assert set(result)=={'ADD_2','MUL_2'}
    assert sum(result.values())==pytest.approx(1)


def test_macro_distillation_is_bounded_and_effective():
    from nexus_m.meta_learning import distill_macro_program
    model=NeuralSociety(agents=4,width=12,top_k=2)
    examples=[(x,2*x+3) for x in (-4,-1,0,2,5)]
    result=distill_macro_program(model,examples,['MUL_2','ADD_3'],steps=8)
    assert result.final_loss<result.initial_loss
    with pytest.raises(ValueError):
        distill_macro_program(model,examples,['ADD_1'],steps=4)


def test_distributed_search_local_workers():
    from nexus_m.distributed_coordinator import DistributedSearchCoordinator
    examples=[(x,2*x+3) for x in (-4,-1,0,2,5)]
    coordinator=DistributedSearchCoordinator(workers=2,timeout_seconds=20)
    result=coordinator.search(examples,simulations_per_worker=350,depth=6)
    assert len(result['worker_reports'])==2
    assert result['executed_on']=='local_multiprocessing_only'
    assert result['total_simulations']<=700
    assert result['found']
