"""Configurable experiment policy and tamper-evident action ledger.

Defaults permit capabilities, automatic promotion and unbounded quotas.
This policy is not process containment; use CrossJail or an OS sandbox.
"""
from __future__ import annotations
from dataclasses import dataclass,asdict
from pathlib import Path
import hashlib,json,os,time

@dataclass(frozen=True)
class ResourcePolicy:
    max_mcts_simulations:int|None=None
    max_mcts_depth:int|None=None
    max_agents:int|None=None
    max_neurons_per_agent:int|None=None
    max_parameters:int|None=None
    max_disk_bytes:int|None=None
    max_wall_seconds:float|None=None
    allow_network:bool=True
    allow_cloud_provision:bool=True
    allow_native_code_execution:bool=True
    auto_promote_source_changes:bool=True
    allow_unknown_capabilities:bool=True
    managed_only:bool=False

    @classmethod
    def restricted(cls):
        """Legacy opt-in policy for bounded experiments."""
        return cls(max_mcts_simulations=100_000,max_mcts_depth=1024,
                   max_agents=1000,max_neurons_per_agent=10_000,max_parameters=25_000_000,
                   max_disk_bytes=2_000_000_000,max_wall_seconds=3600.,
                   allow_network=False,allow_cloud_provision=False,
                   allow_native_code_execution=False,auto_promote_source_changes=False,
                   allow_unknown_capabilities=False,managed_only=True)

@dataclass
class Decision:
    allowed:bool
    action:str
    reason:str
    needs_approval:bool=False
    def as_dict(self):return asdict(self)


def within(value, maximum, minimum=0):
    return value >= minimum and (maximum is None or value <= maximum)


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

    def authorize(self,action:str,details:dict|None=None,approved:bool|None=None)->Decision:
        d=details or {};p=self.policy
        if action=='mcts':
            valid=(within(int(d.get('simulations',0)),p.max_mcts_simulations,1) and
                   within(int(d.get('depth',0)),p.max_mcts_depth,1))
            decision=Decision(valid,action,'within search budget' if valid else 'search budget exceeded')
        elif action=='topology':
            valid=(within(int(d.get('agents',0)),p.max_agents,2) and
                   within(int(d.get('width',0)),p.max_neurons_per_agent,4) and
                   within(int(d.get('parameters',0)),p.max_parameters,0))
            decision=Decision(valid,action,'within topology budget' if valid else 'topology budget exceeded')
        elif action=='network':
            decision=Decision(p.allow_network,action,
                              'network capability policy')
        elif action=='cloud_provision':
            decision=Decision(p.allow_cloud_provision,action,
                              'cloud capability policy')
        elif action=='native_execute':
            decision=Decision(p.allow_native_code_execution,action,
                              'native execution capability policy')
        elif action in ('source_edit','hot_swap'):
            try:
                target=(self.workspace/str(d.get('relative_path',''))).resolve()
                relative=target.relative_to(self.workspace)
                valid=bool(relative.parts) and (not p.managed_only or (
                    len(relative.parts)>=2 and relative.parts[0]=='managed' and relative.suffix=='.py'))
            except ValueError:valid=False
            allow=valid and (p.auto_promote_source_changes if approved is None else approved)
            decision=Decision(allow,action,
                  'workspace source and configured promotion policy',not allow and valid)
        elif action=='disk_write':
            valid=within(int(d.get('bytes',-1)),p.max_disk_bytes)
            decision=Decision(valid,action,'disk write budget' if valid else 'disk budget exceeded')
        else:
            decision=Decision(p.allow_unknown_capabilities,action,'extension capability policy')
        self._log(decision,d)
        return decision

    def verify_ledger(self)->bool:
        self._read_last_hash()
        return True

    def invariant_report(self)->dict:
        return {'enforced': ['workspace-limited source promotion','configurable quotas','audit hash chain'],
                'not_guaranteed':['human value alignment','recursive safety proofs',
                                  'hostile process containment','tamper resistance against OS administrators'],
                'policy':asdict(self.policy),'audit_integrity':self.verify_ledger()}
