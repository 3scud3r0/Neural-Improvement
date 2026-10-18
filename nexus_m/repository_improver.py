"""Repository-level function synthesis, held-out validation and source promotion."""
from __future__ import annotations

import ast
import difflib
import json
import math
import os
from pathlib import Path
import stat
import tempfile
import time

from .alignment_and_value_core import AlignmentGuard, ResourcePolicy
from .function_ast import FunctionTarget, ProposalProvider, digest, mutations
from .function_evaluator import FunctionContract, ProjectEvaluator
from .function_mcts import FunctionMCTS


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def _atomic_source(path: Path, source: bytes):
    mode = stat.S_IMODE(path.stat().st_mode)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f'.{path.name}.nexus-', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(source)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(mode)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class RepositoryFunctionImprover:
    def __init__(self, workspace: str | Path, target: str, function: str,
                 contract: FunctionContract, output: str | Path,
                 policy: ResourcePolicy | None = None):
        self.target = FunctionTarget.read(workspace, target, function)
        self.output = Path(output).resolve()
        if self.output == self.target.workspace or self.output in self.target.workspace.parents \
                or self.output == self.target.path.resolve() or self.output in self.target.path.resolve().parents:
            raise ValueError('output must be a separate artifact directory')
        self.guard = AlignmentGuard(self.target.workspace, policy)
        self.contract = contract

    def _unchanged(self, expected: str):
        path = self.target.path
        path.resolve().relative_to(self.target.workspace)
        if path.is_symlink() or digest(path.read_bytes()) != expected:
            raise RuntimeError('source changed since evaluation; refusing to overwrite it')

    def run(self, *, simulations=32, max_depth=3, max_children=16, min_gain=.05,
            providers: list[ProposalProvider] | None = None, seed=2026, approve=None):
        started = time.monotonic()
        if not math.isfinite(min_gain) or not 0 <= min_gain < 1:
            raise ValueError('min_gain must be in [0, 1)')
        if not self.guard.authorize('mcts', {'simulations':simulations, 'depth':max_depth}).allowed:
            raise PermissionError('MCTS rejected by configured policy')
        self._unchanged(digest(self.target.original))
        self.output.mkdir(parents=True, exist_ok=True)
        for name in ('candidate_module.py', 'candidate_function.py', 'candidate.patch',
                     'report.json', 'search_trace.json'):
            (self.output/name).unlink(missing_ok=True)
        with ProjectEvaluator(self.target, self.contract, self.output) as evaluator:
            search = FunctionMCTS(self.target, evaluator, providers if providers is not None else [mutations],
                                  simulations=simulations, max_depth=max_depth,
                                  max_children=max_children, seed=seed).search()
            ranked = sorted((item for item in search.candidates if item.measurement.passed),
                            key=lambda item:item.gain, reverse=True)
            selected = ranked[0] if ranked else None
            final, baseline_final, eligible = None, None, False
            validations = []
            reason = 'no candidate passed the search contract'
            if selected is not None:
                reason = 'no candidate reached the required measured gain'
            if selected is not None and not self.contract.heldout:
                reason = 'candidate staged; independent heldout commands required for promotion'
            elif selected is not None:
                for candidate in ranked:
                    if candidate.gain < min_gain:
                        continue
                    if baseline_final is None:
                        baseline_final = evaluator.measure(self.target.original, heldout=True)
                        if not baseline_final.passed:
                            reason = 'original source failed heldout contract'
                            break
                    measured = evaluator.measure(candidate.source, heldout=True)
                    final_gain = ((baseline_final.seconds - measured.seconds) / baseline_final.seconds
                                  if measured.passed else None)
                    validations.append(dict(source_sha256=digest(candidate.source),
                                            measurement=measured.as_dict(), gain=final_gain))
                    if measured.passed and final_gain >= min_gain:
                        selected, final, eligible = candidate, measured, True
                        reason = 'search and independent validation passed the gain threshold'
                        break
                    reason = 'candidate failed independent tests or benchmark gain threshold'

        self._unchanged(digest(self.target.original))
        report = dict(target=self.target.relative_path, function=self.target.qualified_name,
                      original_sha256=digest(self.target.original),
                      selected_sha256=digest(selected.source) if selected else None,
                      selected_label=selected.label if selected else None,
                      baseline=search.baseline.as_dict(),
                      baseline_final=baseline_final.as_dict() if baseline_final else None,
                      final=final.as_dict() if final else None,
                      independent_validations=validations, promoted=False, reason=reason,
                      simulations=len(search.trace), unique_candidates=len(search.candidates),
                      search_elapsed_seconds=search.elapsed_seconds,
                      verification_scope='supplied tests and benchmarks, not universal equivalence',
                      contract=vars(self.contract), candidates=[])
        candidates_folder = self.output/'candidates'
        candidates_folder.mkdir(exist_ok=True)
        for candidate in search.candidates:
            sha = digest(candidate.source)
            (candidates_folder/f'{sha}.py').write_bytes(candidate.source)
            report['candidates'].append(dict(label=candidate.label, gain=candidate.gain,
                                              measurement=candidate.measurement.as_dict()))
        if selected is not None:
            (self.output/'candidate_module.py').write_bytes(selected.source)
            (self.output/'candidate_function.py').write_text(ast.unparse(selected.function)+'\n', encoding='utf-8')
            patch = difflib.unified_diff(self.target.text.splitlines(keepends=True),
                                        selected.source.decode(self.target.encoding).splitlines(keepends=True),
                                        fromfile=self.target.relative_path, tofile=self.target.relative_path)
            (self.output/'candidate.patch').write_bytes(''.join(patch).encode('utf-8'))
        _write_json(self.output/'search_trace.json', search.trace)
        allowed = self.guard.policy.auto_promote_source_changes if approve is None else approve
        if eligible and allowed:
            details = {'relative_path':self.target.relative_path}
            if not self.guard.authorize('source_edit', details, approved=allowed).allowed:
                report['reason'] = 'source promotion denied by configured policy'
            else:
                self._unchanged(report['original_sha256'])
                backup = self.output/f"original-{report['original_sha256']}.py"
                backup.write_bytes(self.target.original)
                # Persist recovery information before changing the target.
                promotion = dict(workspace=str(self.target.workspace), target=self.target.relative_path,
                                 original_sha256=report['original_sha256'],
                                 promoted_sha256=digest(selected.source), backup=backup.name, rolled_back=False)
                _write_json(self.output/'promotion.json', promotion)
                _atomic_source(self.target.path, selected.source)
                report['promoted'] = True
                report['reason'] = 'promoted tested source atomically; takes effect in new processes'
        elif eligible:
            report['reason'] = 'validated candidate staged; caller disabled promotion'
        report['audit_ledger_verified'] = self.guard.verify_ledger()
        report['elapsed_seconds'] = time.monotonic()-started
        report['final_gain'] = ((baseline_final.seconds-final.seconds)/baseline_final.seconds
                                if final is not None else None)
        _write_json(self.output/'report.json', report)
        return report


def rollback_function(workspace: str | Path, output: str | Path, *, approve=None,
                      policy: ResourcePolicy | None = None):
    workspace, output = Path(workspace).resolve(), Path(output).resolve()
    meta = json.loads((output/'promotion.json').read_text(encoding='utf-8'))
    if meta['workspace'] != str(workspace) or meta['rolled_back']:
        raise ValueError('promotion belongs to another workspace or was already rolled back')
    guard = AlignmentGuard(workspace, policy)
    path = workspace/meta['target']
    path.resolve().relative_to(workspace)
    if path.is_symlink() or digest(path.read_bytes()) != meta['promoted_sha256']:
        raise RuntimeError('target changed after promotion; refusing to overwrite it')
    backup = (output/meta['backup']).resolve()
    backup.relative_to(output)
    source = backup.read_bytes()
    if digest(source) != meta['original_sha256']:
        raise RuntimeError('backup hash mismatch')
    if not guard.authorize('source_edit', {'relative_path':meta['target']}, approved=approve).allowed:
        raise PermissionError('rollback disabled by caller or policy')
    _atomic_source(path, source)
    meta['rolled_back'] = True
    _write_json(output/'promotion.json', meta)
    return dict(rolled_back=True, target=str(path), restored_sha256=digest(source))
