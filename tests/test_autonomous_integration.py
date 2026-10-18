"""Automatic promotion, extended targets, optional NeuroOS lifecycle."""
import ast
import hashlib
import json
from pathlib import Path
import sys

import pytest

from nexus_m.alignment_and_value_core import AlignmentGuard, ResourcePolicy
from nexus_m.recursive_self_improvement import ManagedSelfImprover, SORT_TEMPLATE
from nexus_m.neuroos_bridge import NeuroOSBridge


def test_default_capabilities_quotas_and_project_paths(tmp_path):
    guard = AlignmentGuard(tmp_path)
    for action in ('network', 'native_execute', 'cloud_provision', 'custom_extension'):
        assert guard.authorize(action).allowed
    assert guard.authorize('mcts', {'simulations':10**9, 'depth':10**5}).allowed
    for target in ('nexus_m/model.py', 'nexus_m/alignment_and_value_core.py', 'configs/test.json'):
        assert guard.authorize('source_edit', {'relative_path':target}).allowed
    assert not guard.authorize('source_edit', {'relative_path':'../outside.py'}).allowed
    assert not guard.authorize('source_edit', {'relative_path':'nexus_m/model.py'}, approved=False).allowed
    assert guard.verify_ledger()


def test_plugin_outside_managed_promotes_and_runs_more_than_five_cycles(tmp_path):
    guard = AlignmentGuard(tmp_path)
    target = tmp_path / 'plugins/router.py'
    target.parent.mkdir()
    # A valid non-template plugin was previously rejected at load time.
    target.write_text('# custom plugin\n' + SORT_TEMPLATE, encoding='utf-8')
    engine = ManagedSelfImprover(guard, min_speedup=-100, target_path='plugins/router.py')
    reports = engine.optimize(7)
    assert len(reports) == 7 and all(r.promoted and r.equivalent for r in reports)
    assert target.with_suffix('.previous.py').exists()
    assert engine.route([2., 2., 3.], 2) == [2, 0]
    # None follows policy, False remains a useful explicit preview.
    before = target.read_text()
    assert not engine.step(approve=False).promoted
    assert target.read_text() == before


def test_continuous_iterator_has_optional_configured_stop(tmp_path):
    engine = ManagedSelfImprover(AlignmentGuard(tmp_path), max_cycles=7)
    assert len(list(engine.iter_optimize(rounds=0, approve=False))) == 7
    with pytest.raises(RuntimeError):
        engine.step()


def test_optional_policy_quotas_still_work(tmp_path):
    guard = AlignmentGuard(tmp_path, ResourcePolicy(max_mcts_simulations=3))
    assert not guard.authorize('mcts', {'simulations':4, 'depth':1}).allowed
    assert guard.authorize('mcts', {'simulations':3, 'depth':1}).allowed


def test_neuroos_lazy_lifecycle_and_workspace(tmp_path, monkeypatch):
    import nexus_m.neuroos_bridge as bridge
    model = tmp_path / 'model.gguf'
    model.touch()
    events = []
    class FakeNeuroOS:
        def __init__(self, **options):
            events.append(options)
        def generate(self, prompt, **options):
            return prompt + ':' + str(options['max_new_tokens'])
        def shutdown(self):
            events.append('closed')
    monkeypatch.setattr(bridge, 'load_neuroos', lambda: FakeNeuroOS)
    adapter = NeuroOSBridge(model, tmp_path / 'runtime')
    assert events == []
    assert adapter.generate('hello', max_new_tokens=3) == 'hello:3'
    assert events[0]['enable_api'] is False
    assert events[0]['forge_dir'] == str(tmp_path / 'runtime/forge')
    adapter.close()
    adapter.close()
    assert events[-1] == 'closed' and len(events) == 2


def test_neuroos_vendor_is_valid_and_matches_provenance():
    folder = Path(__file__).parents[1] / 'nexus_m/_vendor'
    source = (folder / 'neuroos.py').read_bytes()
    ast.parse(source)
    provenance = json.loads((folder / 'neuroos_provenance.json').read_text())
    assert hashlib.sha256(source).hexdigest() == provenance['vendored_sha256']
    assert 'Permission is hereby granted' in (folder / 'NEUROOS_LICENSE').read_text()


def test_evolution_text_generation_connects_bridge(tmp_path, monkeypatch):
    import nexus_m.neuroos_bridge as bridge
    from nexus_m.evolution_system import EvolutionSystem
    model = tmp_path / 'model.gguf'
    model.touch()
    closed = []
    class FakeNeuroOS:
        def __init__(self, **options): pass
        def generate(self, prompt, **options): return 'response for ' + prompt
        def shutdown(self): closed.append(True)
    monkeypatch.setattr(bridge, 'load_neuroos', lambda: FakeNeuroOS)
    with EvolutionSystem(tmp_path / 'run', agents=4, width=8) as system:
        assert system.generate_text('question', model) == 'response for question'
        with pytest.raises(ValueError):
            system.generate_text('question', tmp_path / 'different.gguf')
    assert closed == [True]
