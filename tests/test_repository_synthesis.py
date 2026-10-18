"""General function synthesis against real copied packages and child commands."""
import ast
from copy import deepcopy
import json
from pathlib import Path
import random
import sys

import pytest

from nexus_m.function_ast import (
    FunctionTarget, Proposal, SourceProposals, find_function, function_names, mutations, parse_proposal)
from nexus_m.function_evaluator import FunctionContract, FunctionMeasurement, ProjectEvaluator
from nexus_m.function_mcts import FunctionMCTS
from nexus_m.repository_improver import RepositoryFunctionImprover, rollback_function


ORIGINAL = '''# Preserve module comments and context.
from math import floor

def aggregate(values):
    total = 0
    for value in values:
        total += value
    return total

# This unrelated function must stay byte-for-byte intact.
def untouched(x):
    return floor(x)
'''
FAST = 'def aggregate(values):\n    return sum(values)\n'
OVERFIT = 'def aggregate(values):\n    return 6 if values else 0\n'
BROKEN = 'def aggregate(values):\n    raise RuntimeError("bad candidate")\n'


@pytest.fixture(autouse=True)
def controlled_benchmark_clock(monkeypatch):
    """Exercise promotion decisions deterministically, keeping child tests real.

    Unit tests must not depend on host scheduling or CPU frequency. Only the
    successful timing value is replaced with a synthetic cost here. Actual
    performance is measured separately by the repository/attention examples.
    """
    measure = ProjectEvaluator.measure
    def controlled(self, source, **options):
        result = measure(self, source, **options)
        if result.passed:
            function = find_function(ast.parse(source), self.target.qualified_name)
            has_loop = any(isinstance(node, ast.For) for node in ast.walk(function))
            has_sum = any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                          and node.func.id == 'sum' for node in ast.walk(function))
            result.seconds = 1. if has_loop else .5 if has_sum else .1
        return result
    monkeypatch.setattr(ProjectEvaluator, 'measure', controlled)


@pytest.fixture
def project(tmp_path):
    root = tmp_path/'project'
    package = root/'package'
    package.mkdir(parents=True)
    (package/'__init__.py').write_text('')
    (package/'operations.py').write_text(ORIGINAL)
    (root/'validate.py').write_text(
        'import sys\nfrom package.operations import aggregate,untouched\n'
        'assert untouched(2.4)==2\n'
        'if sys.argv[1]=="train":\n'
        '    assert aggregate([1,2,3])==6\n    assert aggregate([])==0\n'
        'else:\n    assert aggregate([-4,7,2])==5\n    assert aggregate([.5,.25])==.75\n')
    (root/'bench.py').write_text(
        'import json,os,time\nfrom pathlib import Path\nfrom package.operations import aggregate\n'
        'values=list(range(1000))\nstart=time.process_time()\n'
        'for _ in range(1000): aggregate(values)\n'
        'Path(os.environ["NEXUS_BENCHMARK_FILE"]).write_text(json.dumps({"seconds":time.process_time()-start}))\n')
    return root


def contract(**options):
    values = dict(tests=[['{python}', 'validate.py', 'train']],
                  heldout=[['{python}', 'validate.py', 'heldout']],
                  benchmark=['{python}', 'bench.py'], repetitions=1, timeout_seconds=5.)
    values.update(options)
    return FunctionContract(**values)


def providers(tmp_path, *sources):
    paths = []
    for index, source in enumerate(sources):
        path = tmp_path/f'proposal{index}.py'
        path.write_text(source)
        paths.append(path)
    return [SourceProposals(paths)]


def test_arbitrary_function_promotes_after_rejecting_broken_and_overfit(tmp_path, project):
    output = tmp_path/'artifacts'
    engine = RepositoryFunctionImprover(project, 'package/operations.py', 'aggregate', contract(), output)
    report = engine.run(simulations=3, max_depth=1, min_gain=.05,
                        providers=providers(tmp_path, BROKEN, OVERFIT, FAST))
    assert report['promoted']
    assert report['unique_candidates'] == 3
    assert any(not candidate['measurement']['passed'] for candidate in report['candidates'])
    assert any(not validation['measurement']['passed'] for validation in report['independent_validations'])
    text = (project/'package/operations.py').read_text()
    assert 'return sum(values)' in text
    assert text[:text.index('def aggregate')] == ORIGINAL[:ORIGINAL.index('def aggregate')]
    assert text[text.index('# This unrelated'):] == ORIGINAL[ORIGINAL.index('# This unrelated'):]
    assert (output/'candidate.patch').is_file()
    assert (output/'report.json').is_file()
    assert rollback_function(project, output)['rolled_back']
    assert (project/'package/operations.py').read_text() == ORIGINAL


def test_built_in_mutation_discovers_non_router_reduction(tmp_path, project):
    engine = RepositoryFunctionImprover(project, 'package/operations.py', 'aggregate', contract(), tmp_path/'out')
    report = engine.run(simulations=1, min_gain=.05)
    assert report['promoted']
    assert report['candidates'][0]['label'] == 'loop-reduction-to-sum'
    assert 'sum(' in (project/'package/operations.py').read_text()


def test_preview_and_missing_heldout_never_modify_source(tmp_path, project):
    for i, heldout in enumerate(([['{python}', 'validate.py', 'heldout']], [])):
        engine = RepositoryFunctionImprover(project, 'package/operations.py', 'aggregate',
            contract(heldout=heldout), tmp_path/f'out-{i}')
        report = engine.run(simulations=1, approve=False if heldout else None,
                            providers=providers(tmp_path, FAST))
        assert not report['promoted']
        assert (project/'package/operations.py').read_text() == ORIGINAL
        assert (engine.output/'candidate_module.py').is_file()
        assert 'staged' in report['reason']


def test_candidate_timeout_does_not_hang_search_or_change_real_project(tmp_path, project):
    target = FunctionTarget.read(project, 'package/operations.py', 'aggregate')
    source = target.render(parse_proposal('def aggregate(values):\n    while True: pass\n'))
    with ProjectEvaluator(target, contract(timeout_seconds=.2)) as evaluator:
        measurement = evaluator.measure(source)
    assert not measurement.passed and 'timed out' in measurement.error
    assert (project/'package/operations.py').read_text() == ORIGINAL


def test_baseline_failure_prevents_search_and_promotion(tmp_path, project):
    (project/'validate.py').write_text('raise AssertionError("baseline broken")\n')
    engine = RepositoryFunctionImprover(project, 'package/operations.py', 'aggregate', contract(), tmp_path/'out')
    with pytest.raises(RuntimeError, match='baseline contract failed'):
        engine.run(simulations=1)
    assert (project/'package/operations.py').read_text() == ORIGINAL


def test_concurrent_edit_is_preserved_in_promotion_and_rollback(tmp_path, project):
    path = project/'package/operations.py'
    def editor(function, rng):
        path.write_text(ORIGINAL+'\n# concurrent edit\n')
        yield Proposal('sum', parse_proposal(FAST))
    engine = RepositoryFunctionImprover(project, 'package/operations.py', 'aggregate', contract(), tmp_path/'out')
    with pytest.raises(RuntimeError, match='source changed'):
        engine.run(simulations=1, providers=[editor])
    assert path.read_text().endswith('# concurrent edit\n')
    path.write_text(ORIGINAL)
    engine = RepositoryFunctionImprover(project, 'package/operations.py', 'aggregate', contract(), tmp_path/'out2')
    assert engine.run(simulations=1, providers=providers(tmp_path, FAST))['promoted']
    path.write_text(path.read_text()+'\n# later edit\n')
    with pytest.raises(RuntimeError, match='target changed'):
        rollback_function(project, tmp_path/'out2')
    assert path.read_text().endswith('# later edit\n')


def test_full_ast_accepts_different_signature_async_and_nested_helpers(tmp_path, project):
    target = FunctionTarget.read(project, 'package/operations.py', 'aggregate')
    candidate = parse_proposal('''async def aggregate(values, *, offset=0):
    import asyncio
    def helper(x):
        return x + offset
    await asyncio.sleep(0)
    return [helper(x) for x in values]
''')
    rendered = target.render(candidate)
    function = find_function(ast.parse(rendered), 'aggregate')
    assert isinstance(function, ast.AsyncFunctionDef)
    assert function.args.kwonlyargs[0].arg == 'offset'
    assert any(isinstance(node, ast.Import) for node in ast.walk(function))
    assert any(isinstance(node, ast.Await) for node in ast.walk(function))


def test_decorated_method_unicode_encoding_and_crlf_preserve_surroundings(tmp_path):
    raw = ('# coding: latin-1\r\n# olá\r\nclass Attention:\r\n'
           '    @staticmethod\r\n    def score(q, k): return "é"\r\n'
           '\r\n# fim\r\n').encode('latin-1')
    path = tmp_path/'attention.py'
    path.write_bytes(raw)
    target = FunctionTarget.read(tmp_path, path.name, 'Attention.score')
    candidate = parse_proposal('@staticmethod\ndef score(q,k):\n return q @ k.T\n')
    result = target.render(candidate)
    assert result.startswith(raw[:raw.index(b'    @staticmethod')])
    assert result.endswith(b'\r\n\r\n# fim\r\n')
    assert b'    @staticmethod\r\n    def score' in result
    assert list(function_names(ast.parse(result))) == ['Attention.score']


def test_target_discovery_handles_nested_and_conditional_definitions(tmp_path):
    source = 'if True:\n def f(x):\n  def inner(y): return y\n  return inner(x)\n'
    tree = ast.parse(source)
    assert list(function_names(tree)) == ['f', 'f.inner']
    assert find_function(tree, 'f.inner').name == 'inner'
    with pytest.raises(ValueError, match='ambiguous'):
        find_function(ast.parse('def f(): pass\ndef f(): pass'), 'f')
    (tmp_path/'test.py').write_text(source)
    with pytest.raises(ValueError):
        FunctionTarget.read(tmp_path/'other', '../test.py', 'f')


def test_mcts_composes_edits_and_caches_repeated_programs(tmp_path):
    path = tmp_path/'identity.py'
    path.write_text('def identity(x):\n    return (x+0)+0\n')
    target = FunctionTarget.read(tmp_path, path.name, 'identity')
    class Evaluator:
        calls = []
        def measure(self, source):
            self.calls.append(source)
            cost = sum(isinstance(n, ast.BinOp) for n in ast.walk(ast.parse(source))) + 1
            return FunctionMeasurement('unit', True, float(cost))
    def simplify(function, rng):
        new = deepcopy(function)
        expression = new.body[0].value
        if isinstance(expression, ast.BinOp):
            new.body[0].value = expression.left
            yield Proposal('drop-zero', new)
    evaluator = Evaluator()
    searcher = FunctionMCTS(target, evaluator, [simplify], simulations=6, max_depth=2)
    search = searcher.search()
    assert max(entry['depth'] for entry in search.trace) == 2
    assert min(c.measurement.seconds for c in search.candidates) == 1.
    assert len(evaluator.calls) == 3  # baseline + two unique edits
    assert any(entry['cached'] for entry in search.trace)
    assert searcher.root.visits == 6


def test_benchmark_cannot_pass_with_invalid_metric(tmp_path, project):
    (project/'bench.py').write_text('import os\nfrom pathlib import Path\n'
        'Path(os.environ["NEXUS_BENCHMARK_FILE"]).write_text(\'{"seconds": NaN}\')\n')
    target = FunctionTarget.read(project, 'package/operations.py', 'aggregate')
    with ProjectEvaluator(target, contract()) as evaluator:
        measurement = evaluator.measure(target.original)
    assert not measurement.passed and 'positive finite' in measurement.error


def test_invalid_ast_never_reaches_child_evaluation(tmp_path, project):
    engine = RepositoryFunctionImprover(project, 'package/operations.py', 'aggregate', contract(), tmp_path/'out')
    def broken(function, rng):
        function.body = [ast.Break()]
        yield Proposal('invalid-break', ast.fix_missing_locations(function))
    report = engine.run(simulations=1, providers=[broken])
    assert not report['promoted'] and report['unique_candidates'] == 0
    trace = json.loads((tmp_path/'out/search_trace.json').read_text())
    assert 'outside loop' in trace[0]['error']


def test_cli_search_functions_and_rollback(tmp_path, project, capsys):
    from nexus_m.autosynth_cli import main
    output = tmp_path/'out'
    config = tmp_path/'contract.json'
    config.write_text(json.dumps(vars(contract())))
    proposal = tmp_path/'fast.py'
    proposal.write_text(FAST)
    assert main(['functions', '--workspace', str(project), '--target', 'package/operations.py']) == 0
    assert json.loads(capsys.readouterr().out) == ['aggregate', 'untouched']
    assert main(['function-search', '--workspace', str(project), '--target', 'package/operations.py',
                 '--function', 'aggregate', '--contract', str(config), '--output', str(output),
                 '--proposals', str(proposal), '--no-mutations', '--simulations', '1']) == 0
    assert json.loads(capsys.readouterr().out)['promoted']
    assert main(['function-rollback', '--workspace', str(project), '--output', str(output)]) == 0
    assert json.loads(capsys.readouterr().out)['rolled_back']
    assert (project/'package/operations.py').read_text() == ORIGINAL
