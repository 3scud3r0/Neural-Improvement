import ast
from pathlib import Path
import pytest
from nexus_m.function_evaluator import ProjectEvaluator
from nexus_m.sovereign_improver import SovereignImprover,automatic_contract
from nexus_m.repository_improver import rollback_function
from nexus_m.repository_transaction import RepositoryTransaction


@pytest.fixture
def controlled_clock(monkeypatch):
    """Only timing is controlled; generated-reference validation runs for real."""
    measure=ProjectEvaluator.measure
    def measured(self,source,**options):
        result=measure(self,source,**options)
        if result.passed:
            files=source.values() if isinstance(source,dict) else [source]
            result.seconds=sum(len(list(ast.walk(ast.parse(data)))) for data in files)/1000
        return result
    monkeypatch.setattr(ProjectEvaluator,'measure',measured)


@pytest.fixture
def project(tmp_path):
    root=tmp_path/'project';root.mkdir()
    (root/'calc.py').write_text('def f(x:int)->int:\n    return (((x*1)+0)*1)+0\n')
    (root/'other.py').write_text('from calc import f\ndef g(x:int)->int:\n    return ((x+0)*1)+0\n')
    return root


def test_automatic_oracle_and_synthesis_promote_without_proposal_files_or_test_contract(tmp_path,project,controlled_clock):
    original=(project/'calc.py').read_bytes()
    report=SovereignImprover(project).self_improve_cycle('calc.f',output=tmp_path/'out',max_fuel=1000,simulations=2)
    assert report['promoted']
    assert report['reference']['reference_is_specification'] and not report['reference']['intent_inferred']
    assert all(e['evidence']['status']=='proved' for e in report['proofs'])
    assert 'return x' in (project/'calc.py').read_text()
    assert rollback_function(project,tmp_path/'out')['rolled_back']
    assert (project/'calc.py').read_bytes()==original


def test_automatic_coordinated_bundle_validates_both_references_and_recovers(tmp_path,project,controlled_clock):
    originals={p.name:p.read_bytes() for p in project.glob('*.py')}
    result=SovereignImprover(project).self_improve_bundle(['calc.f','other.g'],output=tmp_path/'out',max_fuel=1000)
    assert result['promoted'] and len(result['evaluation']['files'])==2
    assert RepositoryTransaction(project,tmp_path/'out').recover()['state']=='rolled_back'
    assert all((project/name).read_bytes()==data for name,data in originals.items())


def test_unknown_auto_domain_reports_unsupported_not_fictional_promotion(tmp_path,project):
    (project/'calc.py').write_text('def f(x):\n    return x+0\n')
    before=(project/'calc.py').read_bytes()
    report=SovereignImprover(project).self_improve_cycle('calc.f',output=tmp_path/'out')
    assert report['status']=='unsupported_auto_domain' and not report['promoted']
    assert (project/'calc.py').read_bytes()==before


def test_no_change_and_no_approve_have_honest_status(tmp_path,project,controlled_clock):
    engine=SovereignImprover(project)
    report=engine.self_improve_cycle('calc.f',output=tmp_path/'preview',max_fuel=200,approve=False)
    assert not report['promoted'] and report['status']=='not_promoted'
    (project/'calc.py').write_text('def f(x:int)->int:\n    return x\n')
    report=engine.self_improve_cycle('calc.f',output=tmp_path/'unchanged',max_fuel=200)
    assert report['status']=='no_improvement' and not report['promoted']
