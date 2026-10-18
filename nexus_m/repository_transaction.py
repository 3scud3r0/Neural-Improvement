"""Validated multi-file changes with a write-ahead recovery journal.

Individual replacements are atomic; a set of paths cannot be made atomically
visible by os.replace. Cooperating writers are locked and failures are rolled
back. Other readers may see intermediate states. Recovery never overwrites an
unrecognized external edit.
"""
from __future__ import annotations
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import tempfile
from .alignment_and_value_core import AlignmentGuard
from .function_ast import digest
from .function_evaluator import ProjectEvaluator,SnapshotTarget
from .repository_improver import _atomic_source


@contextmanager
def _lock(workspace):
    path=workspace/'.nexus_transaction.lock'
    with path.open('a+b') as stream:
        if os.name=='posix':
            import fcntl
            fcntl.flock(stream.fileno(),fcntl.LOCK_EX)
        else:
            import msvcrt
            stream.seek(0);stream.write(b'0');stream.flush();stream.seek(0)
            msvcrt.locking(stream.fileno(),msvcrt.LK_LOCK,1)
        try:yield
        finally:
            if os.name=='posix':fcntl.flock(stream.fileno(),fcntl.LOCK_UN)
            else:
                stream.seek(0);msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)


def _journal(path,value):
    temp=None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent,prefix='.journal-',delete=False) as stream:
            temp=Path(stream.name)
            stream.write((json.dumps(value,indent=2)+'\n').encode());stream.flush();os.fsync(stream.fileno())
        os.replace(temp,path)
        if os.name=='posix':
            descriptor=os.open(path.parent,os.O_RDONLY)
            try:os.fsync(descriptor)
            finally:os.close(descriptor)
    finally:
        if temp:temp.unlink(missing_ok=True)


def _backup(path,data):
    with path.open('wb') as stream:
        stream.write(data);stream.flush();os.fsync(stream.fileno())


class RepositoryTransaction:
    def __init__(self,workspace,output,policy=None):
        self.workspace=Path(workspace).resolve();self.output=Path(output).resolve()
        if self.output==self.workspace or self.output in self.workspace.parents:
            raise ValueError('transaction output must not be the workspace or its ancestor')
        self.guard=AlignmentGuard(self.workspace,policy)
        self.journal=self.output/'transaction.json'

    def _path(self,relative):
        raw=self.workspace/relative
        path=raw.resolve();path.relative_to(self.workspace)
        if raw.is_symlink() or not path.is_file() or path.suffix!='.py':raise ValueError('transaction targets must be existing Python files, not symlinks')
        if self.output==path or self.output in path.parents:raise ValueError('target must not be inside transaction output')
        return path

    def _commit(self,patches,originals,*,approve=None):
        self.output.mkdir(parents=True,exist_ok=True)
        with _lock(self.workspace):
            if self.journal.exists():
                previous=json.loads(self.journal.read_text())
                if previous['state'] not in ('rolled_back','rejected'):
                    raise RuntimeError('use a new output directory or recover/rollback the prior transaction')
            entries=[]
            for index,(name,new) in enumerate(sorted(patches.items())):
                path=self._path(name);old=originals[name]
                if path.read_bytes()!=old:raise RuntimeError(f'source changed during validation: {name}')
                if not self.guard.authorize('source_edit',{'relative_path':name},approved=approve).allowed:
                    raise PermissionError(f'promotion denied: {name}')
                backup=f'backup-{index}.py';candidate=f'candidate-{index}.py'
                _backup(self.output/backup,old);_backup(self.output/candidate,new)
                entries.append(dict(path=name,original_sha256=digest(old),candidate_sha256=digest(new),
                                    backup=backup,candidate=candidate))
            record=dict(workspace=str(self.workspace),state='prepared',files=entries)
            _journal(self.journal,record)
            try:
                record['state']='applying';_journal(self.journal,record)
                for entry in entries:
                    path=self._path(entry['path'])
                    if digest(path.read_bytes())!=entry['original_sha256']:raise RuntimeError('concurrent file edit')
                    _atomic_source(path,patches[entry['path']])
                record['state']='committed';_journal(self.journal,record)
            except BaseException:
                self._restore(record)
                raise
        return record

    def _restore(self,record):
        # First preflight EVERY file. An external edit blocks the whole restore.
        restores=[]
        try:
            for entry in record['files']:
                path=self._path(entry['path']);current=digest(path.read_bytes())
                if current not in (entry['original_sha256'],entry['candidate_sha256']):
                    raise RuntimeError(f'external edit blocks recovery: {entry["path"]}')
                backup=(self.output/entry['backup']).resolve();backup.relative_to(self.output)
                data=backup.read_bytes()
                if digest(data)!=entry['original_sha256']:raise RuntimeError('transaction backup hash mismatch')
                if current!=entry['original_sha256']:restores.append((path,data))
            for path,data in reversed(restores):_atomic_source(path,data)
            record['state']='rolled_back';_journal(self.journal,record)
        except BaseException:
            record['state']='recovery_required';_journal(self.journal,record)
            raise

    def recover(self):
        with _lock(self.workspace):
            record=json.loads(self.journal.read_text())
            if record['workspace']!=str(self.workspace):raise ValueError('journal belongs to another workspace')
            if record['state']=='rolled_back':return record
            for entry in record['files']:
                if not self.guard.authorize('source_edit',{'relative_path':entry['path']}).allowed:
                    raise PermissionError('recovery denied by policy')
            self._restore(record)
            return record

    def validate_and_apply(self,patches,contract,*,approve=None,min_gain=None):
        if not patches:raise ValueError('empty transaction')
        if min_gain is not None and (not math.isfinite(min_gain) or not 0<=min_gain<1):raise ValueError('min_gain must be in [0,1)')
        originals={};canonical={}
        for name,data in patches.items():
            path=self._path(name);relative=str(path.relative_to(self.workspace))
            if relative in canonical:raise ValueError('duplicate canonical target')
            if not isinstance(data,bytes):raise TypeError('transaction sources must be bytes')
            compile(data,str(path),'exec')
            originals[relative]=path.read_bytes();canonical[relative]=data
        first=next(iter(originals));target=SnapshotTarget(self.workspace,first,originals[first])
        if not contract.heldout:raise ValueError('independent heldout commands required for multi-file promotion')
        with ProjectEvaluator(target,contract,self.output) as evaluator:
            baseline=evaluator.measure(originals,heldout=True)
            candidate=evaluator.measure(canonical,heldout=True) if baseline.passed else None
        report=dict(status='rejected',promoted=False,files=sorted(canonical),
                    baseline=baseline.as_dict(),candidate=candidate.as_dict() if candidate else None,
                    visibility='per-file atomic with recovery; not global multi-file atomic visibility')
        if baseline.passed and candidate.passed:
            gain=(baseline.seconds-candidate.seconds)/baseline.seconds
            report['gain']=gain
            allowed=self.guard.policy.auto_promote_source_changes if approve is None else approve
            if min_gain is not None and gain<min_gain:report['reason']='gain below threshold'
            elif not allowed:report['status']='staged';report['reason']='promotion disabled by caller or policy'
            else:
                self._commit(canonical,originals,approve=approve)
                report['status']='committed';report['promoted']=True
        self.output.mkdir(parents=True,exist_ok=True)
        _journal(self.output/'report.json',report)
        return report
