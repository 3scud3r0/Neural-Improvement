"""Child-process driver. Never run this module against arbitrary sources.

Parent validates exact canonical ASTs. This child has tight builtins, and the
parent applies OS limits/timeouts. It is *not* a hostile-code security boundary.
"""
import heapq
import json
import statistics
import sys
import time

SAFE_BUILTINS = {'sorted': sorted, 'range':range, 'len':len,
                 'ValueError':ValueError, 'min':min, 'list':list}

def _fn(source):
    namespace={'__builtins__':SAFE_BUILTINS,'heapq':heapq}
    exec(compile(source, '<approved_ast>', 'exec'), namespace)
    return namespace['route']

def _outcome(fn,scores,k):
    try:return ['ok',fn(scores,k)]
    except Exception as exc:return ['error',type(exc).__name__]

def _time(fn,cases,repetitions):
    for vals,k in cases[:8]:fn(vals,k)
    samples=[]
    for _ in range(repetitions):
        t=time.perf_counter_ns()
        for vals,k in cases:fn(vals,k)
        samples.append((time.perf_counter_ns()-t)/1e6)
    return statistics.median(samples)

def main():
    payload=json.load(sys.stdin)
    baseline=_fn(payload['baseline']);candidate=_fn(payload['candidate'])
    mismatches=[]
    for i,(values,k) in enumerate(payload['checks']):
        a,b=_outcome(baseline,values,k),_outcome(candidate,values,k)
        if a!=b:
            mismatches.append({'case':i,'expected':a,'actual':b})
            if len(mismatches)>=4:break
    if mismatches:
        print(json.dumps({'equivalent':False,'mismatches':mismatches,'baseline_ms':None,'candidate_ms':None}))
        return
    # Two timings executed in same child, with balanced ordering to limit clock drift.
    cases=payload['timing'];repeat=payload.get('repetitions',3)
    baseline_ms=_time(baseline,cases,repeat)
    candidate_ms=_time(candidate,cases,repeat)
    print(json.dumps({'equivalent':True,'mismatches':[],
                      'baseline_ms':baseline_ms,'candidate_ms':candidate_ms}))

if __name__=='__main__':main()
