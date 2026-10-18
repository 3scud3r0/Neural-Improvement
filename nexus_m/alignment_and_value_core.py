"""Enforceable local operating constraints and tamper-evident action ledger.

This module is NOT a solution to AI alignment or a proof of human values. It
provides bounded permissions, reproducible audits, manual promotion gates,
resource quotas and explicit denials. External process isolation requires
separate OS/container controls; a Python policy cannot be incorruptible.
"""
from __future__ import annotations
from dataclasses import dataclass,asdict
from pathlib import Path
import hashlib,json,os,time

@dataclass(frozen=True)
class ResourcePolicy:
    max_mcts_simulations:int=100_000
    max_mcts_depth:int=1024
    max_agents:int=1000
    max_neurons_per_agent:int=10_000
    max_parameters:int=25_000_000
    max_disk_bytes:int=2_000_000_000
    max_wall_seconds:float=3600.
    allow_network:bool=False
    allow_cloud_provision:bool=False
    allow_native_code_execution:bool=False
    auto_promote_source_changes:bool=False

@dataclass
class Decision:
    allowed:bool
    action:str
    reason:str
    needs_approval:bool=False
    def as_dict(self):return asdict(self)


class AlignmentGuard:
    """Local capability guard, not a value-learning or formal safety theorem."""
    def __init__(self,workspace:str|Path,policy:ResourcePolicy|None=None):
        self.workspace=Path(workspace).resolve()
        self.workspace.mkdir(parents=True,exist_ok=True)
        self.policy=policy or ResourcePolicy()
        self.audit=self.workspace/'audit_chain.jsonl'
        self._last_hash=self._read_last_hash()

    def _read_last_hash(self):
        prev='0'*64
        if not self.audit.exists():return prev
        for line in self.audit.read_text(encoding='utf-8').splitlines():
            record=json.loads(line)
            payload={k:v for k,v in record.items() if k!='hash'}
            actual=hashlib.sha256(json.dumps(payload,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
            if payload['prev_hash']!=prev or actual!=record['hash']:
                raise RuntimeError('audit history inconsistency')
            prev=actual
        return prev

    def _log(self,decision:Decision,details:dict|None=None):
        entry={'timestamp':time.time(),'decision':decision.as_dict(),'details':details or {},
               'prev_hash':self._last_hash}
        entry['hash']=hashlib.sha256(json.dumps(entry,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        with self.audit.open('a',encoding='utf-8') as f:f.write(json.dumps(entry,ensure_ascii=False,sort_keys=True)+'\n')
        self._last_hash=entry['hash']

    def authorize(self,action:str,details:dict|None=None,approved:bool=False)->Decision:
        d=details or {};p=self.policy
        if action=='mcts':
            valid=(1<=int(d.get('simulations',0))<=p.max_mcts_simulations and
                   1<=int(d.get('depth',0))<=p.max_mcts_depth)
            decision=Decision(valid,action,'within search budget' if valid else 'search budget exceeded')
        elif action=='topology':
            valid=(2<=int(d.get('agents',0))<=p.max_agents and
                   4<=int(d.get('width',0))<=p.max_neurons_per_agent and
                   int(d.get('parameters',0))<=p.max_parameters)
            decision=Decision(valid,action,'within topology budget' if valid else 'topology budget exceeded')
        elif action=='network':
            decision=Decision(p.allow_network and approved,action,
                              'network access requires configuration and approval',True)
        elif action=='cloud_provision':
            decision=Decision(p.allow_cloud_provision and approved,action,
                              'cloud provisioning requires opt-in and human approval',True)
        elif action=='native_execute':
            decision=Decision(p.allow_native_code_execution and approved,action,
                              'native execution requires explicit permissions',True)
        elif action in ('source_edit','hot_swap'):
            try:
                target=(self.workspace/str(d.get('relative_path',''))).resolve()
                relative=target.relative_to(self.workspace)
                valid=len(relative.parts)>=2 and relative.parts[0]=='managed' and relative.suffix=='.py'
            except ValueError:valid=False
            allow=valid and (approved or p.auto_promote_source_changes)
            decision=Decision(allow,action,
                  'allowlisted managed source and promotion approval required',not allow and valid)
        elif action=='disk_write':
            valid=0<=int(d.get('bytes',-1))<=p.max_disk_bytes
            decision=Decision(valid,action,'disk write budget' if valid else 'disk budget exceeded')
        else:
            decision=Decision(False,action,'unknown capability')
        self._log(decision,d)
        return decision

    def verify_ledger(self)->bool:
        self._read_last_hash()
        return True

    def invariant_report(self)->dict:
        return {'enforced': ['workspace-limited source promotion','manual gate by default',
                             'bounded search','bounded topology','audit hash chain'],
                'not_guaranteed':['human value alignment','recursive safety proofs',
                                  'hostile process containment','tamper resistance against OS administrators'],
                'policy':asdict(self.policy),'audit_integrity':self.verify_ledger()}
