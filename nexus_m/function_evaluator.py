"""Evaluate arbitrary Python function edits in fresh project copies.

This is process separation, not an OS sandbox. Test/benchmark commands and
proposal providers are trusted configuration. Wrap the runner in CrossJail or
a disposable VM when stronger containment is needed.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tempfile
import time

from crossjail import COPY_IGNORE, _kill_tree
from .function_ast import FunctionTarget, digest


@dataclass
class SnapshotTarget:
    workspace: Path
    relative_path: str
    original: bytes
    @property
    def path(self):return self.workspace/self.relative_path


@dataclass
class FunctionContract:
    tests: list[list[str]]
    benchmark: list[str]
    heldout: list[list[str]] = field(default_factory=list)
    repetitions: int = 3
    timeout_seconds: float = 60.
    benchmark_timeout_seconds: float = 60.
    seed: int = 2026

    def __post_init__(self):
        if type(self.seed) is not int:
            raise ValueError('seed must be an integer')
        if not self.tests or not self.benchmark:
            raise ValueError('contract requires tests and a benchmark command')
        for command in [*self.tests, *self.heldout, self.benchmark]:
            if not isinstance(command, list) or not command or not all(
                    isinstance(arg, str) and arg for arg in command):
                raise ValueError('commands must be nonempty argument arrays; no shell strings')
        if (type(self.repetitions) is not int or self.repetitions < 1
                or not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0
                or not math.isfinite(self.benchmark_timeout_seconds) or self.benchmark_timeout_seconds <= 0):
            raise ValueError('repetitions and timeouts must be positive')

    @classmethod
    def read(cls, path: str | Path):
        return cls(**json.loads(Path(path).read_text(encoding='utf-8')))


@dataclass
class FunctionMeasurement:
    source_sha256: str
    passed: bool
    seconds: float | None
    error: str | None = None
    commands: list[dict] = field(default_factory=list)
    def as_dict(self):
        return asdict(self)


def _environment(work: Path, scratch: Path, seed: int):
    secret = ('TOKEN', 'SECRET', 'PASSWORD', 'CREDENTIAL', 'API_KEY', 'ACCESS_KEY',
              'PRIVATE_KEY', 'SSH_AUTH_SOCK')
    env = {key: value for key, value in os.environ.items()
           if not any(term in key.upper() for term in secret)
           and key not in ('PYTHONPATH', 'PYTHONSTARTUP', 'PYTHONHOME', 'LD_PRELOAD',
                           'DYLD_INSERT_LIBRARIES')}
    # Use the active interpreter's installed dependencies, including user-site
    # packages. This intentionally isn't CrossJail's stricter environment.
    env.update(PYTHONPATH=str(work), PYTHONDONTWRITEBYTECODE='1',
               PYTHONHASHSEED=str(seed % 2**32), NEXUS_EVALUATION_SEED=str(seed),
               TMPDIR=str(scratch), TMP=str(scratch), TEMP=str(scratch))
    return env


def run_command(command: list[str], cwd: Path, env: dict, timeout: float) -> dict:
    argv = [sys.executable if arg == '{python}' else arg for arg in command]
    kwargs = {'start_new_session': True} if os.name == 'posix' else {
        'creationflags': subprocess.CREATE_NEW_PROCESS_GROUP}
    start = time.monotonic()
    with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
        proc = subprocess.Popen(argv, cwd=cwd, env=env, stdout=out, stderr=err, **kwargs)
        timed_out = False
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill_tree(proc)
            proc.wait(timeout=10)
        except BaseException:
            _kill_tree(proc)
            proc.wait(timeout=10)
            raise
        finally:
            if os.name == 'posix':
                _kill_tree(proc)
        streams = []
        for stream in (out, err):
            size = stream.tell()
            stream.seek(max(0, size - 8192))
            streams.append(stream.read(8192).decode('utf-8', errors='replace'))
    return dict(command=argv, returncode=proc.returncode, timed_out=timed_out,
                duration_seconds=time.monotonic()-start, stdout=streams[0], stderr=streams[1])


class ProjectEvaluator:
    def __init__(self, target: FunctionTarget, contract: FunctionContract,
                 output: Path | None = None):
        self.target, self.contract = target, contract
        self.output = Path(output).resolve() if output else None
        self._temporary = tempfile.TemporaryDirectory(prefix='nexus_function_snapshot_')
        self.snapshot = Path(self._temporary.name) / 'project'
        self._closed = False
        try:
            def ignore(folder, names):
                ignored = shutil.ignore_patterns(*COPY_IGNORE, 'runs', '*.egg-info')(folder, names)
                for name in set(names) - set(ignored):
                    path = Path(folder) / name
                    if self.output and path.resolve() == self.output:
                        ignored.append(name)
                    elif path.is_symlink():
                        raise ValueError(f'project snapshots do not follow symlinks: {path}')
                return ignored
            shutil.copytree(target.workspace, self.snapshot, ignore=ignore)
            if target.path.read_bytes() != target.original:
                raise RuntimeError('target changed while taking snapshot')
            # Reject a symlink alias whose lexical copy was excluded or ambiguous.
            copied = self.snapshot / target.relative_path
            if not copied.is_file() or copied.read_bytes() != target.original:
                raise ValueError('target must be included unchanged in the project snapshot')
        except BaseException:
            self.close()
            raise

    def close(self):
        if not self._closed:
            self._closed = True
            self._temporary.cleanup()

    def __enter__(self): return self
    def __exit__(self, *args): self.close()

    def measure(self, source: bytes | dict[str,bytes], *, heldout=False) -> FunctionMeasurement:
        sources=source if isinstance(source,dict) else {self.target.relative_path:source}
        identity=(json.dumps({name:digest(data) for name,data in sorted(sources.items())},sort_keys=True).encode()
                  if isinstance(source,dict) else source)
        result = FunctionMeasurement(digest(identity), False, None)
        try:
            with tempfile.TemporaryDirectory(prefix='nexus_function_trial_') as folder:
                root = Path(folder)
                work, scratch = root/'work', root/'tmp'
                scratch.mkdir()
                shutil.copytree(self.snapshot, work)
                for name,data in sources.items():
                    path=(work/name).resolve()
                    path.relative_to(work)
                    if not path.is_file() or not isinstance(data,bytes):raise ValueError('patches must replace existing project files with bytes')
                    if path.suffix=='.py':compile(data,str(path),'exec')
                    path.write_bytes(data)
                env = _environment(work, scratch, self.contract.seed + (1 if heldout else 0))
                tests = self.contract.tests + (self.contract.heldout if heldout else [])
                for command in tests:
                    record = run_command(command, work, env, self.contract.timeout_seconds)
                    result.commands.append(record)
                    if record['timed_out'] or record['returncode'] != 0:
                        result.error = 'test command timed out' if record['timed_out'] else 'test command failed'
                        return result
                timings = []
                for index in range(self.contract.repetitions):
                    metric = scratch/f'metric-{index}.json'
                    env['NEXUS_BENCHMARK_FILE'] = str(metric)
                    record = run_command(self.contract.benchmark, work, env,
                                         self.contract.benchmark_timeout_seconds)
                    result.commands.append(record)
                    if record['timed_out'] or record['returncode'] != 0:
                        result.error = 'benchmark timed out' if record['timed_out'] else 'benchmark command failed'
                        return result
                    data = json.loads(metric.read_text(encoding='utf-8'))
                    seconds = data['seconds']
                    if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds <= 0:
                        raise ValueError('benchmark must report a positive finite seconds value')
                    timings.append(float(seconds))
                result.seconds = statistics.median(timings)
                result.passed = True
        except (OSError, ValueError, SyntaxError, KeyError, TypeError, subprocess.SubprocessError) as exc:
            result.error = str(exc)
        return result
