"""Profile the managed router to create reproducible bottleneck observations."""
from __future__ import annotations
import cProfile,pstats,io,random,time
from dataclasses import dataclass,asdict
from .ast_program_synthesis import RouterSpec,compile_source
from .search_driven_improver import SearchDrivenImprover

@dataclass
class ProfileReport:
    calls:int
    total_ms:float
    mean_microseconds:float
    hotspots:list[dict]
    recommendation:str
    def as_dict(self):return asdict(self)

def profile(improver:SearchDrivenImprover,*,iterations:int=100,seed:int=2026)->ProfileReport:
    if not 1<=iterations<=10000:raise ValueError('iterations must be bounded')
    rng=random.Random(seed)
    batches=[([rng.uniform(-10.,10.) for _ in range(rng.choice((256,1000,2000)))],4)
             for _ in range(iterations)]
    profiler=cProfile.Profile()
    t=time.perf_counter()
    profiler.enable()
    for scores,k in batches:improver.route(scores,k)
    profiler.disable()
    elapsed=(time.perf_counter()-t)*1000
    stats=pstats.Stats(profiler)
    rows=[]
    for (filename,line,func),(cc,nc,total,cumulative,callers) in stats.stats.items():
        if func in ('route','sorted','nsmallest','<lambda>','min'):
            rows.append({'function':func,'calls':nc,
                        'self_ms':total*1000,'cumulative_ms':cumulative*1000})
    rows.sort(key=lambda x:x['cumulative_ms'],reverse=True)
    recommendation=('Try top-k selective search (heap or selection) rather than complete sorting'
                    if any(r['function']=='sorted' for r in rows) else
                    'Benchmark alternative stable tie-breaking and indexing')
    return ProfileReport(iterations,elapsed,elapsed*1000/iterations,rows[:12],recommendation)
