"""Validate generated Python AST, measure candidates inside bounded child processes."""
from __future__ import annotations
from dataclasses import dataclass,asdict
from pathlib import Path
import json,math,random,subprocess,sys
from .ast_program_synthesis import RouterSpec,compile_source,validate_generated_source,reference_route

@dataclass
class CandidateMeasurement:
    spec:dict
    equivalent:bool
    baseline_ms:float|None
    candidate_ms:float|None
    improvement:float
    error:str|None
    mismatches:list
    def as_dict(self):return asdict(self)


def make_cases(seed:int=2026,checks:int=65,timing:int=45,large:bool=False):
    rng=random.Random(seed)
    cases=[]
    for v,k in [([],0),([],1),([0.],0),([0.],1),([1.,1.,0.],2),
                ([0.,1.,1.,1.],3),([5.,5.,5.,5.],4),([-4.,-5.,-3.],1),
                ([1.,2.],-1),([1.,2.],3),([2.0,-2.0,1.0],1)]:
        cases.append([v,k])
    for _ in range(checks):
        n=rng.choice([0,1,2,3,5,8,12,32,64,128])
        scores=[float(rng.choice([-3,-2,-1,0,1,2,3]) if rng.random()<.45
                        else round(rng.uniform(-5,5),4)) for _ in range(n)]
        k=rng.choice([0,1,2,3,n,max(0,n-1),n+1,-1])
        cases.append([scores,k])
    benchmarking=[]
    for _ in range(timing):
        n=rng.choice([400,800,1600,3200] if large else [120,300,600,1200])
        scores=[rng.uniform(-100,100) for _ in range(n)]
        benchmarking.append([scores,rng.choice([1,2,4,8])])
    return cases,benchmarking


def _set_limits():
    # Linux/Unix defense-in-depth. Not a substitute for an OS sandbox.
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_CPU,(6,7))
        resource.setrlimit(resource.RLIMIT_AS,(512*1024**2,512*1024**2))
        resource.setrlimit(resource.RLIMIT_FSIZE,(8*1024**2,8*1024**2))
    except (ImportError,OSError,ValueError):pass


def evaluate(spec:RouterSpec,*,seed:int=2026,large:bool=False,
             timeout:float=8.0,checks:int=65,timing:int=45,repetitions:int=3,
             baseline_spec:RouterSpec|None=None)->CandidateMeasurement:
    source=compile_source(spec)
    validate_generated_source(source)
    base=compile_source(baseline_spec or RouterSpec())
    check_cases,timing_cases=make_cases(seed,checks,timing,large)
    payload=json.dumps({'baseline':base,'candidate':source,'checks':check_cases,
                        'timing':timing_cases,'repetitions':repetitions})
    try:
        result=subprocess.run([sys.executable,'-I',str(Path(__file__).with_name('ast_runtime_harness.py'))],
                 input=payload,text=True,capture_output=True,timeout=timeout,
                 cwd=Path(__file__).parent,preexec_fn=_set_limits if sys.platform!='win32' else None)
        if result.returncode!=0:
            raise RuntimeError('child exit '+str(result.returncode)+': '+result.stderr[-500:])
        data=json.loads(result.stdout)
        a,b=data.get('baseline_ms'),data.get('candidate_ms')
        gain=(a-b)/a if a and b is not None and a>0 else 0.
        return CandidateMeasurement(spec.as_dict(),data['equivalent'],a,b,gain,None,data['mismatches'])
    except (subprocess.TimeoutExpired,ValueError,RuntimeError,json.JSONDecodeError) as exc:
        return CandidateMeasurement(spec.as_dict(),False,None,None,0.,str(exc),[])
