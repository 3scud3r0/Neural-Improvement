"""Exercise containment claims and failure modes with real child processes."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

import crossjail
from crossjail import CrossJail, JailPolicy


@pytest.fixture
def project(tmp_path):
    src = tmp_path / 'project'
    src.mkdir()
    (src / 'data.txt').write_text('original', encoding='utf-8')
    return src


def make_jail(tmp_path, project, **options):
    return CrossJail.init(tmp_path / 'jail', project,
                          JailPolicy(allow_network=True, **options))


def test_copy_scrubbing_policy_reload_and_audit(tmp_path, project, monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'test-secret')
    monkeypatch.setenv('PYTHONPATH', '/host/private')
    (project / '.env').write_text('SECRET=test-secret')
    jail = make_jail(tmp_path, project, max_timeout_sec=2, max_memory_mb=128)
    assert not (jail.root / 'work/.env').exists()
    loaded = CrossJail(jail.root)
    assert loaded.policy.allow_network and loaded.policy.max_memory_mb == 128
    code = ("import os,json; from pathlib import Path; "
            "Path('data.txt').write_text('changed'); "
            "print(json.dumps({k:os.environ.get(k) for k in "
            "['OPENAI_API_KEY','PYTHONPATH','HOME','CROSSJAIL']}))")
    result = loaded.run([sys.executable, '-c', code])
    assert result.returncode == 0, result.stderr
    env = json.loads(result.stdout)
    assert env['OPENAI_API_KEY'] is None and env['PYTHONPATH'] is None
    assert env['HOME'] == str(jail.root / 'home') and env['CROSSJAIL'] == '1'
    assert (project / 'data.txt').read_text() == 'original'
    assert (jail.root / 'work/data.txt').read_text() == 'changed'
    events = [json.loads(line)['event'] for line in jail.audit.path.read_text().splitlines()]
    assert events == ['init', 'run_start', 'run_end']
    assert CrossJail(jail.root).meta.last_run['returncode'] == 0


def test_missing_status_does_not_create_root(tmp_path):
    root = tmp_path / 'absent'
    assert not CrossJail(root).status()['exists']
    assert not root.exists()
    with pytest.raises(ValueError):
        CrossJail(root).run([sys.executable, '-c', 'pass'])
    assert not root.exists()


def test_lifecycle_refuses_overlap_and_unmanaged_deletion(tmp_path, project):
    with pytest.raises(ValueError):
        CrossJail.init(project / 'jail', project)
    with pytest.raises(ValueError):
        CrossJail.init(tmp_path, project, force=True)
    unrelated = tmp_path / 'unrelated'
    unrelated.mkdir()
    keep = unrelated / 'keep.txt'
    keep.write_text('keep')
    with pytest.raises(ValueError):
        CrossJail.init(unrelated, project, force=True)
    assert keep.read_text() == 'keep'
    jail = make_jail(tmp_path, project)
    with pytest.raises(ValueError):
        jail.destroy()
    jail.destroy(force=True)
    assert not jail.root.exists() and project.is_dir()


def test_reject_cwd_escape_and_symlink_copy(tmp_path, project):
    jail = make_jail(tmp_path, project)
    for cwd in ('../project', str(project)):
        with pytest.raises(ValueError):
            jail.run([sys.executable, '-c', 'pass'], cwd_relative=cwd)
    try:
        (project / 'link').symlink_to(tmp_path, target_is_directory=True)
    except OSError:
        pytest.skip('symlink creation unavailable')
    with pytest.raises(ValueError):
        CrossJail.init(tmp_path / 'other-jail', project)
    assert not (tmp_path / 'other-jail').exists()


def test_network_failure_refuses_payload(tmp_path, project, monkeypatch):
    jail = CrossJail.init(tmp_path / 'jail', project)
    def denied(*args):
        raise RuntimeError('network isolation unavailable')
    monkeypatch.setattr(crossjail, '_network_prefix', denied)
    result = jail.run([sys.executable, '-c', "open('executed','w').close()"])
    assert result.returncode == 125
    assert 'network isolation unavailable' in result.stderr
    assert not (jail.root / 'work/executed').exists()


def test_real_network_policy_enforced_or_refused(tmp_path, project):
    jail = CrossJail.init(tmp_path / 'jail', project)
    result = jail.run([sys.executable, '-c',
                       "import socket; s=socket.socket(); s.settimeout(.2); "
                       "print(s.connect_ex(('1.1.1.1',443)))"])
    if result.returncode == 125:
        assert 'network isolation' in result.stderr
    else:
        assert result.returncode == 0, result.stderr
        assert int(result.stdout.strip()) != 0
        assert result.capabilities['network_enforcement'] != 'allowed'


def test_timeout_and_bounded_output(tmp_path, project):
    jail = make_jail(tmp_path, project, max_output_bytes=64)
    result = jail.run([sys.executable, '-c', "print('x'*1000)"])
    assert result.returncode == 0
    assert 'output truncated' in result.stdout
    result = jail.run([sys.executable, '-c', 'import time; time.sleep(30)'], timeout=.2)
    assert result.timed_out and result.duration_sec < 5


@pytest.mark.skipif(os.name != 'posix', reason='POSIX process group regression')
def test_timeout_kills_descendant(tmp_path, project):
    jail = make_jail(tmp_path, project)
    marker = jail.root / 'work/escaped.txt'
    descendant = "import time; from pathlib import Path; time.sleep(1); Path('escaped.txt').touch()"
    code = (f"import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',{descendant!r}]); "
            "print('ready',flush=True); time.sleep(30)")
    result = jail.run([sys.executable, '-c', code], timeout=.4)
    assert 'ready' in result.stdout and result.timed_out
    time.sleep(1.1)
    assert not marker.exists()


@pytest.mark.skipif(not sys.platform.startswith('linux'), reason='Linux address space limit')
def test_memory_limit_applies_to_payload(tmp_path, project):
    jail = make_jail(tmp_path, project, max_memory_mb=96)
    result = jail.run([sys.executable, '-c', 'data=bytearray(256*1024*1024)'])
    assert result.returncode != 0
    assert 'MemoryError' in result.stderr


def test_snippets_spawn_portably_and_capture_output():
    policy = JailPolicy(allow_network=True)
    result = crossjail.restricted_exec('print(sum(range(10)))', policy)
    assert result.returncode == 0 and result.stdout.strip() == '45'
    assert result.audit_path is None
    result = crossjail.restricted_exec('from os import system', policy)
    assert result.returncode == 1 and 'denied module' in result.stderr
    result = crossjail.restricted_exec('while True: pass', policy, timeout=.2)
    assert result.timed_out


def test_cli_exact_arguments_and_exit_status(tmp_path, project):
    runner = Path(crossjail.__file__)
    result = subprocess.run([sys.executable, str(runner), 'exec', '--src', str(project),
                             '--network', '--', sys.executable, '-c',
                             'import sys; print(sys.argv[1]); sys.exit(7)', 'a b'],
                            capture_output=True, text=True)
    assert result.returncode == 7, result.stderr
    assert result.stdout.strip() == 'a b'


def test_secret_and_loader_environment_overrides_rejected(tmp_path, project):
    jail = make_jail(tmp_path, project)
    for key in ('OPENAI_API_KEY', 'LD_PRELOAD', 'PYTHONPATH'):
        jail.policy.extra_env = {key: 'test-value'}
        with pytest.raises(ValueError):
            jail.run([sys.executable, '-c', 'pass'])
