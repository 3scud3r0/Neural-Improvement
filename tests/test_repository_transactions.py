import json
from pathlib import Path
import subprocess
import sys
import pytest
from nexus_m.function_evaluator import FunctionContract
from nexus_m.repository_transaction import RepositoryTransaction
from nexus_m.whole_program_pdg import ProgramDependencyGraph


@pytest.fixture
def workspace(tmp_path):
    root=tmp_path/'project';package=root/'samplepkg';package.mkdir(parents=True)
    (package/'__init__.py').write_text('')
    (package/'maths.py').write_text('def calculate(x):\n    return x+1\n')
    (package/'caller.py').write_text('from .maths import calculate as compute\ndef run(x):\n    return compute(x)\n')
    (package/'top.py').write_text('import samplepkg.caller as api\ndef execute(x):\n    return api.run(x)\n')
    (root/'validate.py').write_text('from samplepkg.top import execute\nassert execute(-7)==-6\nassert execute(5)==6\n')
    (root/'bench.py').write_text('import os,json\nfrom pathlib import Path\nPath(os.environ["NEXUS_BENCHMARK_FILE"]).write_text(json.dumps({"seconds":1.0}))\n')
    return root


def contract():
    # Timings are a controlled fixture. These tests check coordinated correctness,
    # journaling and recovery rather than claiming a benchmark improvement.
    return FunctionContract(tests=[['{python}','validate.py']],heldout=[['{python}','validate.py']],
                            benchmark=['{python}','bench.py'],repetitions=1)


def coordinated(root):
    graph=ProgramDependencyGraph(root)
    return graph.create_multi_file_transaction({
        'samplepkg.maths.calculate':'def calculate(x, offset):\n    return x+offset\n',
        'samplepkg.caller.run':'def run(x):\n    return compute(x,1)\n'})


def test_graph_resolves_relative_imports_module_aliases_and_transitive_callers(workspace):
    graph=ProgramDependencyGraph(workspace)
    assert graph.get_impact_closure('samplepkg.maths.calculate')=={
        'samplepkg.maths.calculate','samplepkg.caller.run','samplepkg.top.execute'}
    # Full module paths avoid collisions between similarly named files.
    other=workspace/'other';other.mkdir()
    (other/'maths.py').write_text('def calculate(x):return x\n')
    graph.build_graph()
    assert 'other.maths.calculate' in graph.file_map
    with pytest.raises(ValueError):graph.resolve_symbol('calculate')
    report=graph.impact_report('samplepkg.maths.calculate')
    assert not report['complete']


def test_graph_resolves_self_methods_and_reports_dynamic_dispatch(workspace):
    (workspace/'objects.py').write_text('''class Engine:
    def inner(self,x): return x
    def outer(self,x): return self.inner(x)
def main(x):
    engine=Engine()
    return engine.outer(x)
def dynamic(callback,x): return callback(x)
''')
    graph=ProgramDependencyGraph(workspace)
    assert graph.get_impact_closure('objects.Engine.inner')=={
        'objects.Engine.inner','objects.Engine.outer','objects.main'}
    assert any(call['caller']=='objects.dynamic' for call in graph.unresolved)


def test_signature_change_requires_statically_affected_modules(workspace):
    graph=ProgramDependencyGraph(workspace)
    with pytest.raises(ValueError,match='affected module'):
        graph.create_multi_file_transaction({'samplepkg.maths.calculate':'def calculate(x,offset):return x+offset'})
    with pytest.raises(ValueError,match='affected module'):
        graph.create_multi_file_transaction({'samplepkg.maths.calculate':'async def calculate(x):return x+1'})
    # The transitive top-level caller also needs to be represented for this
    # deliberately conservative signature-change policy.
    patches={
        'samplepkg.maths.calculate':'def calculate(x, offset):return x+offset',
        'samplepkg.caller.run':'def run(x):return compute(x,1)',
        'samplepkg.top.execute':'def execute(x):return api.run(x)'}
    files=graph.create_multi_file_transaction(patches)
    assert len(files)==3 and all(b'def ' in source for source in files.values())


def valid_patches(root):
    graph=ProgramDependencyGraph(root)
    return graph.create_multi_file_transaction({
        'samplepkg.maths.calculate':'def calculate(x,offset):return x+offset',
        'samplepkg.caller.run':'def run(x):return compute(x,1)',
        'samplepkg.top.execute':'def execute(x):return api.run(x)'})


def test_complete_bundle_validates_commits_and_rolls_back(tmp_path,workspace):
    patches=valid_patches(workspace)
    original={name:(workspace/name).read_bytes() for name in patches}
    transaction=RepositoryTransaction(workspace,tmp_path/'artifacts')
    report=transaction.validate_and_apply(patches,contract())
    assert report['promoted'] and report['status']=='committed'
    assert all((workspace/name).read_bytes()==source for name,source in patches.items())
    assert transaction.recover()['state']=='rolled_back'
    assert all((workspace/name).read_bytes()==source for name,source in original.items())


def test_bundle_validation_failure_and_preview_leave_all_files_unchanged(tmp_path,workspace):
    patches=valid_patches(workspace);original={n:(workspace/n).read_bytes() for n in patches}
    broken=dict(patches);broken['samplepkg/caller.py']=b'def run(x):return None\n'
    rejected=RepositoryTransaction(workspace,tmp_path/'bad').validate_and_apply(broken,contract())
    assert not rejected['promoted']
    staged=RepositoryTransaction(workspace,tmp_path/'preview').validate_and_apply(patches,contract(),approve=False)
    assert staged['status']=='staged'
    assert all((workspace/n).read_bytes()==data for n,data in original.items())


def test_failure_during_second_install_restores_first_file(tmp_path,workspace,monkeypatch):
    import nexus_m.repository_transaction as module
    patches=valid_patches(workspace);old={n:(workspace/n).read_bytes() for n in patches}
    original=module._atomic_source;calls=[]
    def fail_second(path,data):
        calls.append(str(path))
        if len(calls)==2:raise OSError('injected replacement failure')
        return original(path,data)
    monkeypatch.setattr(module,'_atomic_source',fail_second)
    transaction=RepositoryTransaction(workspace,tmp_path/'output')
    with pytest.raises(OSError,match='injected'):
        transaction.validate_and_apply(patches,contract())
    assert json.loads(transaction.journal.read_text())['state']=='rolled_back'
    assert all((workspace/n).read_bytes()==data for n,data in old.items())


def test_process_crash_leaves_recoverable_journal(tmp_path,workspace):
    patches=valid_patches(workspace);old={n:(workspace/n).read_bytes() for n in patches}
    output=tmp_path/'crashed'
    encoded=json.dumps({n:v.decode() for n,v in patches.items()})
    code=f'''import json,os
import nexus_m.repository_transaction as module
root={str(workspace)!r}
patches={{n:v.encode() for n,v in json.loads({encoded!r}).items()}}
old={{n:(module.Path(root)/n).read_bytes() for n in patches}}
original=module._atomic_source
def crash(path,data):
    original(path,data)
    os._exit(17)
module._atomic_source=crash
module.RepositoryTransaction(root,{str(output)!r})._commit(patches,old)
'''
    child=subprocess.run([sys.executable,'-c',code],timeout=10)
    assert child.returncode==17
    assert any((workspace/n).read_bytes()!=data for n,data in old.items())
    transaction=RepositoryTransaction(workspace,output)
    assert transaction.recover()['state']=='rolled_back'
    assert all((workspace/n).read_bytes()==data for n,data in old.items())


def test_external_edit_blocks_recovery_without_partially_restoring_others(tmp_path,workspace):
    patches=valid_patches(workspace)
    transaction=RepositoryTransaction(workspace,tmp_path/'output')
    assert transaction.validate_and_apply(patches,contract())['promoted']
    path=workspace/'samplepkg/maths.py';path.write_text('# user edit\n'+path.read_text())
    before={n:(workspace/n).read_bytes() for n in patches}
    with pytest.raises(RuntimeError,match='external edit'):transaction.recover()
    assert all((workspace/n).read_bytes()==data for n,data in before.items())
    assert json.loads(transaction.journal.read_text())['state']=='recovery_required'


def test_transaction_refuses_escape_and_invalid_python(tmp_path,workspace):
    transaction=RepositoryTransaction(workspace,tmp_path/'output')
    with pytest.raises(ValueError):transaction.validate_and_apply({'../outside.py':b'x=1'},contract())
    with pytest.raises(SyntaxError):transaction.validate_and_apply({'samplepkg/maths.py':b'def broken('},contract())
