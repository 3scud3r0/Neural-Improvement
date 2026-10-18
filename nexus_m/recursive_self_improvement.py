"""Router optimization with configurable cycles, plugin targets and promotion.

External plugins execute ordinary Python; containment is the caller's responsibility.
"""
from __future__ import annotations
from dataclasses import dataclass,asdict
from pathlib import Path
import hashlib,importlib.util,json,random,statistics,time
from .alignment_and_value_core import AlignmentGuard

SORT_TEMPLATE='''"""Managed router: deterministic top-k (sorting)."""\ndef route(scores, k):\n    if not 0 <= k <= len(scores):\n        raise ValueError("invalid k")\n    return sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:k]\n'''

HEAP_TEMPLATE='''"""Managed router: deterministic top-k (heap)."""\nimport heapq\ndef route(scores, k):\n    if not 0 <= k <= len(scores):\n        raise ValueError("invalid k")\n    return heapq.nsmallest(k, range(len(scores)), key=lambda i: (-scores[i], i))\n'''

@dataclass
class ImprovementReport:
    proposed:str
    baseline_median_ms:float
    candidate_median_ms:float
    equivalent:bool
    relative_gain:float
    promoted:bool
    reason:str
    source_path:str
    def as_dict(self):return asdict(self)

class ManagedSelfImprover:
    def __init__(self,guard:AlignmentGuard,min_speedup:float=0.05,*,max_cycles:int|None=None,
                 target_path:str="managed/router.py"):
        if max_cycles is not None and max_cycles<1:raise ValueError("max_cycles must be positive")
        self.max_cycles=max_cycles
        self.guard=guard
        self.managed=self.guard.workspace/'managed'
        self.managed.mkdir(parents=True,exist_ok=True)
        self.path=(guard.workspace/target_path).resolve()
        self.path.relative_to(guard.workspace)
        self.path.parent.mkdir(parents=True,exist_ok=True)
        if not self.path.exists():
            self.path.write_text(SORT_TEMPLATE,encoding='utf-8')
        self.min_speedup=min_speedup
        self.rounds=0
        self._route=self._load(self.path)

    @staticmethod
    def _load(path:Path):
        # Execute fresh source rather than cached bytecode after a hot swap.
        namespace={"__file__":str(path),"__name__":"nexus_router_plugin"}
        exec(compile(path.read_text(encoding='utf-8'),str(path),'exec'),namespace)
        route=namespace.get('route')
        if not callable(route):raise TypeError('plugin must define route(scores, k)')
        return route

    def route(self,scores:list[float],k:int):return self._route(scores,k)

    @staticmethod
    def _benchmark(fn,batches):
        trials=[]
        for _ in range(3):
            start=time.perf_counter()
            for scores,k in batches:fn(scores,k)
            trials.append((time.perf_counter()-start)*1000)
        return statistics.median(trials)

    def step(self,approve:bool|None=None,seed:int=2026,batches:int=160)->ImprovementReport:
        self.rounds+=1
        if self.max_cycles is not None and self.rounds>self.max_cycles:
            raise RuntimeError('configured improvement cycle limit reached')
        approve=self.guard.policy.auto_promote_source_changes if approve is None else approve
        active=self.path.read_text(encoding='utf-8')
        candidate=HEAP_TEMPLATE if active==SORT_TEMPLATE else SORT_TEMPLATE
        rng=random.Random(seed)
        examples=[([rng.uniform(-5,5) for _ in range(rng.choice((32,100,1000)))],4)
                  for _ in range(max(10,min(400,batches)))]
        # Explicit edge-cases, ties, finite values, degenerate k.
        correctness=[([1.,1.,0.],2),([-4.,-5.,-3.],1),([0.],0),([0.],1)]
        temp=self.managed/'candidate_router.py'
        temp.write_text(candidate,encoding='utf-8')
        try:
            proposed=self._load(temp)
            equivalent=all(self._route(values,k)==proposed(values,k)
                           for values,k in examples+correctness)
            baseline=self._benchmark(self._route,examples)
            alt=self._benchmark(proposed,examples)
            gain=(baseline-alt)/max(baseline,1e-12)
            promotion=False;reason='candidate failed equivalence tests'
            if equivalent:
                if gain<self.min_speedup:
                    reason=f'gain {gain:.3f} < required {self.min_speedup:.3f}'
                elif not approve:
                    reason='candidate is eligible but manual promotion not approved'
                else:
                    rel=str(self.path.relative_to(self.guard.workspace))
                    if not self.guard.authorize('source_edit',{'relative_path':rel},approved=approve).allowed:
                        reason='source edit rejected by policy'
                    elif not self.guard.authorize('hot_swap',{'relative_path':rel},approved=approve).allowed:
                        reason='hot swap rejected by policy'
                    else:
                        backup=self.path.with_suffix('.previous.py')
                        backup.write_text(active,encoding='utf-8')
                        replacement=self.path.with_suffix('.next.py')
                        replacement.write_text(candidate,encoding='utf-8')
                        try:
                            replacement.replace(self.path) # atomic within filesystem
                            self._route=self._load(self.path) # runtime function swap
                            assert all(self.route(values,k)==self._load(backup)(values,k)
                                       for values,k in correctness)
                            promotion=True;reason='passed equivalence, benchmark and promotion policy'
                        except Exception:
                            self.path.write_text(active,encoding='utf-8')
                            self._route=self._load(self.path)
                            raise
            report=ImprovementReport('heap' if candidate==HEAP_TEMPLATE else 'sort',
                         baseline,alt,equivalent,gain,promotion,reason,str(self.path))
            with (self.managed/'improvement_reports.jsonl').open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(report.as_dict(),ensure_ascii=False)+'\n')
            return report
        finally:
            temp.unlink(missing_ok=True)

    def iter_optimize(self,rounds:int=0,approve:bool|None=None):
        """Yield reports; rounds=0 continues until interrupted or max_cycles."""
        if rounds<0:raise ValueError('rounds must be nonnegative')
        index=0
        while rounds==0 or index<rounds:
            if self.max_cycles is not None and self.rounds>=self.max_cycles:break
            yield self.step(approve=approve,seed=2026+index)
            index+=1

    def optimize(self,rounds:int=1,approve:bool|None=None)->list[ImprovementReport]:
        if rounds<1:raise ValueError('use iter_optimize for continuous runs')
        return list(self.iter_optimize(rounds,approve))
