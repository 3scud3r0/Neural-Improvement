#!/usr/bin/env python3
"""Cross-platform experiment runner using only the Python standard library.

Copies a project, scrubs the environment, records runs and kills process groups
on timeout. Linux network isolation uses unshare; macOS uses sandbox-exec.
If network isolation cannot be established, execution is refused. --network
explicitly permits networking. File copies, cwd and Python guards are NOT an
OS filesystem sandbox: payloads can access host files with user permissions.
Use a disposable VM for hostile code. No symlink/hardlink copy modes are offered.
"""
from __future__ import annotations

import argparse
import ast
from dataclasses import asdict, dataclass, field
import fnmatch
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from typing import Sequence
import uuid

CROSSJAIL_VERSION = '1.0.0'
DEFAULT_TIMEOUT = 600
DEFAULT_MEM_MB = 4096
META_FILENAME = 'crossjail_meta.json'
AUDIT_FILENAME = 'crossjail_audit.jsonl'
COPY_IGNORE = ('.git', '__pycache__', '.venv', 'venv', 'node_modules',
               '.pytest_cache', '.mypy_cache', '.ruff_cache', '*.pyc',
               '.env', '.env.*', '*.pem', '*.key', '.ssh', '.aws', '.azure',
               '.netrc', '_netrc', '.pypirc', '.git-credentials')
IS_WINDOWS = os.name == 'nt'
IS_POSIX = os.name == 'posix'


@dataclass
class JailPolicy:
    allow_network: bool = False
    max_timeout_sec: float = DEFAULT_TIMEOUT
    max_memory_mb: int = DEFAULT_MEM_MB
    max_cpu_sec: int = 0
    max_output_bytes: int = 1024 * 1024
    env_passthrough: list[str] = field(default_factory=lambda: ['PATH', 'LANG', 'LC_ALL', 'TERM'])
    extra_env: dict[str, str] = field(default_factory=dict)
    denied_modules: list[str] = field(default_factory=lambda: [
        'ctypes', 'cffi', 'socket', 'subprocess', 'multiprocessing', 'os',
        'sys', 'shutil', 'pathlib', 'tempfile', 'importlib'])

    def __post_init__(self):
        if self.max_timeout_sec <= 0:
            raise ValueError('timeout must be positive')
        if min(self.max_memory_mb, self.max_cpu_sec, self.max_output_bytes) < 0:
            raise ValueError('resource values must be nonnegative')


@dataclass
class JailMeta:
    jail_id: str
    root: str
    src: str
    created_at: float
    platform: str
    policy: dict
    last_run: dict | None = None


@dataclass
class RunResult:
    returncode: int
    stdout: str
    stderr: str
    duration_sec: float
    timed_out: bool
    peak_rss_mb: float | None = None
    audit_path: str | None = None
    capabilities: dict = field(default_factory=dict)


def _inside(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def _atomic_json(path: Path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)


class AuditLog:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()

    def write(self, event, **details):
        with self._lock, self.path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(dict(ts=time.time(), event=event, **details),
                                    ensure_ascii=False) + '\n')


def copy_tree_safe(src: Path, dst: Path, mode='copy'):
    if mode != 'copy':
        raise ValueError('only independent file copies are supported')
    src, dst = src.resolve(), dst.resolve()
    if _inside(dst, src) or _inside(src, dst):
        raise ValueError('source and destination must not overlap')

    def ignore(folder, names):
        ignored = [name for name in names if any(fnmatch.fnmatch(name, p) for p in COPY_IGNORE)]
        for name in set(names) - set(ignored):
            if (Path(folder) / name).is_symlink():
                raise ValueError(f'symlinks are not copied: {Path(folder) / name}')
        return ignored

    shutil.copytree(src, dst, ignore=ignore)


def scrub_env(policy: JailPolicy, root: Path) -> dict[str, str]:
    # A fixed set excludes credentials even if callers add names to passthrough.
    permitted = {'PATH', 'LANG', 'LC_ALL', 'TERM', 'SYSTEMROOT', 'WINDIR',
                 'COMSPEC', 'PATHEXT', 'SYSTEMDRIVE'}
    env = {k: v for k, v in os.environ.items()
           if k.upper() in permitted and (IS_WINDOWS or k in policy.env_passthrough)}
    forbidden = {'PYTHONPATH', 'PYTHONHOME', 'PYTHONSTARTUP', 'LD_PRELOAD',
                 'LD_LIBRARY_PATH', 'DYLD_INSERT_LIBRARIES', 'DYLD_LIBRARY_PATH'}
    for key, value in policy.extra_env.items():
        if key.upper() in forbidden or any(term in key.upper() for term in (
                'KEY', 'TOKEN', 'SECRET', 'PASSWORD', 'CREDENTIAL', 'SSH')):
            raise ValueError(f'environment override not permitted: {key}')
        env[key] = value
    for name in ('home', 'tmp'):
        (root / name).mkdir(exist_ok=True)
    env.update(HOME=str(root / 'home'), USERPROFILE=str(root / 'home'),
               TMPDIR=str(root / 'tmp'), TMP=str(root / 'tmp'), TEMP=str(root / 'tmp'),
               PYTHONNOUSERSITE='1', CROSSJAIL='1', CROSSJAIL_VERSION=CROSSJAIL_VERSION)
    return env


def _network_prefix(policy: JailPolicy, env: dict) -> tuple[list[str], str]:
    if policy.allow_network:
        return [], 'allowed'
    if sys.platform.startswith('linux') and shutil.which('unshare'):
        prefix = [shutil.which('unshare'), '--user', '--map-root-user', '--net', '--']
    elif sys.platform == 'darwin' and shutil.which('sandbox-exec'):
        prefix = [shutil.which('sandbox-exec'), '-p',
                  '(version 1) (allow default) (deny network*)']
    else:
        raise RuntimeError('network isolation unavailable; use a VM or explicitly pass --network')
    probe = subprocess.run(prefix + [sys.executable, '-I', '-c', 'pass'], env=env,
                           capture_output=True, timeout=5)
    if probe.returncode:
        raise RuntimeError('network isolation refused by this OS; use a VM or explicitly pass --network')
    return prefix, 'linux-network-namespace' if sys.platform.startswith('linux') else 'macos-sandbox-exec'


def _child(policy_data: str, command: list[str]) -> int:
    """Bootstrap limits in a fresh interpreter, avoiding threaded preexec_fn."""
    policy = JailPolicy(**json.loads(policy_data))
    if IS_POSIX:
        import resource
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        if policy.max_cpu_sec:
            resource.setrlimit(resource.RLIMIT_CPU, (policy.max_cpu_sec,) * 2)
        if policy.max_memory_mb and sys.platform.startswith('linux'):
            resource.setrlimit(resource.RLIMIT_AS, (policy.max_memory_mb * 1024**2,) * 2)
    # No blanket ImportError fallback: if a requested limit fails, do not launch.
    os.execvpe(command[0], command, os.environ)
    return 127


def _kill_tree(proc):
    if IS_POSIX:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    else:
        subprocess.run(['taskkill', '/PID', str(proc.pid), '/T', '/F'],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
        if proc.poll() is None:
            proc.kill()


class CrossJail:
    def __init__(self, root: Path, policy: JailPolicy | None = None):
        if Path(root).is_symlink():
            raise ValueError('jail root must not be a symlink')
        self.root = Path(root).resolve()
        self.meta_path = self.root / META_FILENAME
        # Read-only operations on missing roots must not create directories.
        self.meta = None
        if self.meta_path.is_file():
            self.meta = JailMeta(**json.loads(self.meta_path.read_text(encoding='utf-8')))
            if self.meta.root != str(self.root):
                raise ValueError('metadata root mismatch')
            uuid.UUID(self.meta.jail_id)
        self.policy = policy or (JailPolicy(**self.meta.policy) if self.meta else JailPolicy())
        self.audit = AuditLog(self.root / AUDIT_FILENAME)

    def _validate(self):
        if self.meta is None or not (self.root / 'work').is_dir():
            raise ValueError('not an initialized CrossJail root')
        if (self.root / 'work').is_symlink():
            raise ValueError('work directory must not be a symlink')
        src = Path(self.meta.src).resolve()
        if _inside(src, self.root) or _inside(self.root, src):
            raise ValueError('jail root overlaps source')

    @classmethod
    def init(cls, root: Path, src: Path, policy: JailPolicy | None = None, force=False):
        root, src = Path(root), Path(src).resolve()
        if root.is_symlink():
            raise ValueError('jail root must not be a symlink')
        root = root.resolve()
        if not src.is_dir():
            raise FileNotFoundError(src)
        if _inside(root, src) or _inside(src, root):
            raise ValueError('source and jail root must not overlap')
        policy = policy or JailPolicy()
        if root.exists():
            if not force:
                raise FileExistsError(root)
            cls(root).destroy(force=True)  # only delete a validated jail
        root.mkdir(parents=True)
        try:
            copy_tree_safe(src, root / 'work')
            meta = JailMeta(str(uuid.uuid4()), str(root), str(src), time.time(),
                            platform.platform(), asdict(policy))
            _atomic_json(root / META_FILENAME, asdict(meta))
            jail = cls(root)
            jail.audit.write('init', jail_id=meta.jail_id, src=str(src))
            return jail
        except BaseException:
            shutil.rmtree(root)
            raise

    def destroy(self, force=False):
        self._validate()
        if not force:
            raise ValueError('destroy requires --force')
        self.audit.write('destroy')
        shutil.rmtree(self.root)

    def status(self):
        return dict(root=str(self.root), exists=self.root.exists(), version=CROSSJAIL_VERSION,
                    meta=asdict(self.meta) if self.meta else None,
                    capabilities=self.capabilities())

    def capabilities(self):
        return dict(filesystem_sandbox=False, environment_scrubbed=True,
                    independent_copy=True, wall_timeout=True,
                    posix_cpu_limit=IS_POSIX,
                    linux_address_space_limit=sys.platform.startswith('linux'),
                    windows_resource_limits=False,
                    network_requested='allowed' if self.policy.allow_network else 'isolated',
                    network_enforcement='probed on each run')

    def run(self, cmd: Sequence[str], timeout: float | None = None,
            cwd_relative='work', capture=True) -> RunResult:
        self._validate()
        if isinstance(cmd, str) or not cmd:
            raise ValueError('cmd must be a nonempty sequence of arguments')
        timeout = self.policy.max_timeout_sec if timeout is None else timeout
        if timeout <= 0:
            raise ValueError('timeout must be positive')
        work = (self.root / cwd_relative).resolve()
        if not _inside(work, (self.root / 'work').resolve()):
            raise ValueError('cwd must remain inside work')
        if not work.is_dir():
            raise FileNotFoundError(work)
        env = scrub_env(self.policy, self.root)
        capabilities = self.capabilities()
        start = time.monotonic()
        timed_out, returncode, stdout, stderr = False, 125, '', ''
        self.audit.write('run_start', cmd=list(cmd), timeout=timeout, cwd=str(work))
        proc = None
        try:
            prefix, capabilities['network_enforcement'] = _network_prefix(self.policy, env)
            bootstrap = (list(cmd) if IS_WINDOWS else
                         [sys.executable, '-I', str(Path(__file__).resolve()),
                          '--_child', json.dumps(asdict(self.policy)), *cmd])
            kwargs = dict(cwd=work, env=env)
            if IS_POSIX:
                kwargs['start_new_session'] = True
            elif IS_WINDOWS:
                kwargs['creationflags'] = subprocess.CREATE_NEW_PROCESS_GROUP
            # Disk spools bound parent memory; disk consumption remains an OS concern.
            with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
                proc = subprocess.Popen(prefix + bootstrap,
                                        stdout=out if capture else None,
                                        stderr=err if capture else None, **kwargs)
                try:
                    returncode = proc.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    _kill_tree(proc)
                    returncode = proc.wait(timeout=10)
                finally:
                    # Clean surviving children even when the group leader exits first.
                    if IS_POSIX:
                        _kill_tree(proc)
                if capture:
                    for stream, name in ((out, 'stdout'), (err, 'stderr')):
                        stream.seek(0)
                        data = stream.read(self.policy.max_output_bytes + 1)
                        truncated = len(data) > self.policy.max_output_bytes
                        value = data[:self.policy.max_output_bytes].decode('utf-8', errors='replace')
                        if truncated:
                            value += '\n[crossjail] output truncated\n'
                        if name == 'stdout':
                            stdout = value
                        else:
                            stderr = value
        except KeyboardInterrupt:
            if proc is not None:
                _kill_tree(proc)
                proc.wait(timeout=10)
            returncode, stderr = 130, '[crossjail] interrupted'
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
            if proc is not None and proc.poll() is None:
                _kill_tree(proc)
                proc.wait(timeout=10)
            stderr = f'[crossjail] launch failed: {exc}'
        result = RunResult(returncode, stdout, stderr, time.monotonic() - start,
                           timed_out, audit_path=str(self.audit.path), capabilities=capabilities)
        self.audit.write('run_end', returncode=returncode, duration_sec=result.duration_sec,
                         timed_out=timed_out, capabilities=capabilities)
        self.meta.last_run = dict(returncode=returncode, timed_out=timed_out,
                                  duration_sec=result.duration_sec, ts=time.time())
        _atomic_json(self.meta_path, asdict(self.meta))
        return result


def exec_once(src: Path, cmd: Sequence[str], policy=None, timeout=None) -> RunResult:
    with tempfile.TemporaryDirectory(prefix='crossjail_') as container:
        jail = CrossJail.init(Path(container) / 'jail', src, policy)
        result = jail.run(cmd, timeout=timeout)
        # The ephemeral audit is deleted with the temporary root.
        result.audit_path = None
        return result


def restricted_exec(code: str, policy: JailPolicy, timeout=30) -> RunResult:
    """Small-snippet convenience; AST filtering is NOT a security boundary."""
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                modules = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or '']
                if any(m.split('.')[0] in policy.denied_modules for m in modules):
                    raise ValueError('denied module referenced')
    except (SyntaxError, ValueError) as exc:
        return RunResult(1, '', str(exc), 0., False)
    # No multiprocessing closure, so this also works with Windows spawn.
    with tempfile.TemporaryDirectory(prefix='crossjail_snippet_') as folder:
        src = Path(folder) / 'src'
        src.mkdir()
        (src / 'snippet.txt').write_text(code, encoding='utf-8')
        (src / 'run.py').write_text(
            "import builtins\nfrom pathlib import Path\n"
            "names = 'abs min max sum len range enumerate zip print str int float list dict set tuple bool ValueError'.split()\n"
            "ns = {'__builtins__': {k: getattr(builtins,k) for k in names}}\n"
            "exec(compile(Path('snippet.txt').read_text(encoding='utf-8'), '<snippet>', 'exec'), ns, ns)\n",
            encoding='utf-8')
        return exec_once(src, [sys.executable, 'run.py'], policy, timeout)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', action='version', version=CROSSJAIL_VERSION)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('init', 'run', 'exec', 'status', 'destroy'):
        p = sub.add_parser(name)
        if name != 'exec':
            p.add_argument('--root', type=Path, required=True)
        if name in ('init', 'exec'):
            p.add_argument('--src', type=Path, required=True)
            p.add_argument('--network', action='store_true', help='explicitly permit networking')
            p.add_argument('--no-network', action='store_false', dest='network')
            p.add_argument('--mem-mb', type=int, default=DEFAULT_MEM_MB)
            p.add_argument('--cpu-sec', type=int, default=0)
        if name in ('init', 'run', 'exec'):
            p.add_argument('--timeout', type=float, default=None if name == 'run' else DEFAULT_TIMEOUT)
        if name in ('run', 'exec'):
            p.add_argument('--cmd', help='quoted command; use -- argv for exact arguments')
            p.add_argument('--no-capture', action='store_true')
            p.add_argument('argv', nargs=argparse.REMAINDER)
        if name in ('init', 'destroy'):
            p.add_argument('--force', action='store_true')
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command in ('init', 'exec'):
            policy = JailPolicy(allow_network=args.network, max_timeout_sec=args.timeout,
                                max_memory_mb=args.mem_mb, max_cpu_sec=args.cpu_sec)
        if args.command == 'init':
            CrossJail.init(args.root, args.src, policy, args.force)
            return 0
        if args.command in ('run', 'exec'):
            exact = args.argv[1:] if args.argv[:1] == ['--'] else args.argv
            if bool(exact) == bool(args.cmd):
                parser.error('supply either --cmd or -- argv')
            if IS_WINDOWS and args.cmd:
                parser.error('on Windows use -- python script.py "argument with spaces"')
            cmd = exact or shlex.split(args.cmd)
            if args.command == 'run':
                result = CrossJail(args.root).run(cmd, args.timeout, capture=not args.no_capture)
            else:
                if args.no_capture:
                    parser.error('--no-capture is supported for persistent run only')
                result = exec_once(args.src, cmd, policy, args.timeout)
            sys.stdout.write(result.stdout)
            sys.stderr.write(result.stderr)
            return 124 if result.timed_out else (128 - result.returncode if result.returncode < 0 else result.returncode)
        jail = CrossJail(args.root)
        if args.command == 'status':
            print(json.dumps(jail.status(), indent=2, ensure_ascii=False))
        else:
            jail.destroy(force=args.force)
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f'[crossjail] {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--_child':
        raise SystemExit(_child(sys.argv[2], sys.argv[3:]))
    raise SystemExit(main())
