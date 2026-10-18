"""Contract, finiteness, diff testing, constrained synthesis and promotion tests."""
from __future__ import annotations
import ast,json
from pathlib import Path
import pytest
from nexus_m.ast_program_synthesis import RouterSpec,all_router_specs,compile_source,validate_generated_source
from nexus_m.ast_candidate_runtime import evaluate
from nexus_m.ast_formal_gate import verify,promotion_eligible
from nexus_m.ast_mcts import ASTSynthesisMCTS
from nexus_m.alignment_and_value_core import AlignmentGuard,ResourcePolicy
from nexus_m.search_driven_improver import SearchDrivenImprover


def test_every_ast_is_valid_python_and_a_canonical_source():
    assert len(list(all_router_specs()))==24
    for spec in all_router_specs():
        src=compile_source(spec)
        compile(src,'<candidate>','exec')
        assert validate_generated_source(src)==spec
        assert len(list(ast.walk(ast.parse(src))))<120


def test_unreviewed_source_rejected():
    with pytest.raises(ValueError):
        validate_generated_source('def route(scores,k):\n import os\n os.system("echo bad")\n')
    with pytest.raises(ValueError):
        validate_generated_source('def route(scores,k):\n return __import__("os")\n')


def test_finite_equivalence_and_bad_candidate():
    for algo in ['sort','heap','scan']:
        proof=verify(RouterSpec(algo,'stable','strict'),max_n=4)
        assert promotion_eligible(proof)
        assert proof.checked_cases>1000
        assert not proof.machine_checked_universal_proof
    bad=verify(RouterSpec('heap','index','strict'),max_n=3)
    assert not bad.finite_domain_equivalent
    assert not promotion_eligible(bad)


def test_candidate_in_bounded_child_process():
    m=evaluate(RouterSpec('heap','stable','strict'),seed=87,checks=9,timing=8)
    assert m.equivalent and m.error is None
    assert m.baseline_ms is not None and m.candidate_ms is not None
    bad=evaluate(RouterSpec('sort','index','strict'),seed=87,checks=9,timing=8)
    assert not bad.equivalent


def test_ast_mcts_discovers_correct_candidates():
    report=ASTSynthesisMCTS(simulations=30,seed=2026,checks=8,timing=6).search()
    assert report.unique_candidates>1
    assert report.accepted_candidates>=1
    assert len(report.trajectory)==30
    assert report.best_source.startswith('def route(')


def test_explicit_preview_no_automatic_promotion(tmp_path,monkeypatch):
    guard=AlignmentGuard(tmp_path)
    improver=SearchDrivenImprover(guard)
    before=improver.active.read_text(encoding='utf-8')
    # Deterministic successful candidate; search itself is separately tested.
    import nexus_m.search_driven_improver as impmod
    class FakeSearch:
        def __init__(self,**kw):pass
        def search(self):
            return type('R',(),{'best_spec':RouterSpec('heap','stable','strict').as_dict(),
              'best_source':compile_source(RouterSpec('heap','stable','strict')),
              'unique_candidates':6,'elapsed_seconds':1.,'best_train_gain':.1,
              'as_dict':lambda self:{'result':'mock_for_integration_only'}})()
    monkeypatch.setattr(impmod,'ASTSynthesisMCTS',FakeSearch)
    res=improver.run(simulations=20,approve=False,min_gain=0.)
    assert res.heldout_equivalent
    assert not res.promoted
    assert improver.active.read_text(encoding='utf-8')==before
    assert improver.staged.exists()
    assert guard.verify_ledger()


def test_default_automatic_swap_and_rollback(tmp_path,monkeypatch):
    guard=AlignmentGuard(tmp_path)
    improver=SearchDrivenImprover(guard)
    old=improver.active.read_text()
    import nexus_m.search_driven_improver as impmod
    class FakeSearch:
        def __init__(self,**kw):pass
        def search(self):
            return type('R',(),{'best_spec':RouterSpec('heap','stable','strict').as_dict(),
              'best_source':compile_source(RouterSpec('heap','stable','strict')),
              'unique_candidates':5,'elapsed_seconds':1.,'best_train_gain':.4,
              'as_dict':lambda self:{'result':'mock_for_integration_only'}})()
    monkeypatch.setattr(impmod,'ASTSynthesisMCTS',FakeSearch)
    outcome=improver.run(simulations=10,min_gain=0.)
    assert outcome.promoted
    assert improver.route([3.,3.,2.],2)==[0,1]
    assert improver.active.read_text()!=old
    assert improver.rollback()
    assert improver.active.read_text()==old
    assert guard.verify_ledger()


def test_cannot_edit_policy_file_even_with_manual_approval(tmp_path):
    guard=AlignmentGuard(tmp_path,ResourcePolicy.restricted())
    decision=guard.authorize('source_edit',{'relative_path':'nexus_m/alignment_and_value_core.py'},approved=True)
    assert not decision.allowed
    assert not guard.authorize('cloud_provision',{'agents':100},approved=True).allowed
    assert not guard.authorize('native_execute',{'cmd':'ls'},approved=True).allowed
