"""Bounded multi-process search worker coordinator (one machine, no cloud).

A thin distributed-systems demonstration: immutable requests, independent
MCTS workers, deterministic seeds, aggregation and time/resource budgets.
Not a production cluster scheduler or a hardened code-execution sandbox.
"""
from __future__ import annotations
from dataclasses import dataclass,asdict
from concurrent.futures import ProcessPoolExecutor,as_completed,TimeoutError as FuturesTimeout
import time,multiprocessing
from .hierarchical_abstraction_search import HierarchicalMCTS,evaluate_macro_program

@dataclass(frozen=True)
class WorkerJob:
    examples:tuple[tuple[int,int],...]
    simulations:int
    depth:int
    seed:int


def _worker(job:WorkerJob):
    result=HierarchicalMCTS(simulations=job.simulations,max_depth=job.depth,
                            seed=job.seed).search(list(job.examples))
    return {'seed':job.seed,'result':result.as_dict()}


class DistributedSearchCoordinator:
    def __init__(self,workers=2,timeout_seconds=30.,max_simulations_per_worker=10000):
        if not 1<=workers<=4:raise ValueError('only 1..4 local worker processes allowed')
        if not 1<=timeout_seconds<=300:raise ValueError('invalid wall time limit')
        self.workers=workers
        self.timeout_seconds=timeout_seconds
        self.max_simulations_per_worker=max_simulations_per_worker

    def search(self,examples:list[tuple[int,int]],simulations_per_worker=350,
               depth=6,seed=2026):
        if not 1<=simulations_per_worker<=self.max_simulations_per_worker:
            raise ValueError('simulation quota exceeded')
        if not 1<=depth<=50 or not 1<=len(examples)<=100:raise ValueError('budget exceeded')
        job_examples=tuple(tuple(map(int,pair)) for pair in examples)
        jobs=[WorkerJob(job_examples,simulations_per_worker,depth,seed+i) for i in range(self.workers)]
        start=time.perf_counter()
        pool=ProcessPoolExecutor(max_workers=self.workers,
                                 mp_context=multiprocessing.get_context("spawn"))
        try:
            futures=[pool.submit(_worker,job) for job in jobs]
            reports=[]
            for future in as_completed(futures,timeout=self.timeout_seconds):
                reports.append(future.result())
        except FuturesTimeout as exc:
            for f in futures:f.cancel()
            pool.shutdown(wait=False,cancel_futures=True)
            raise TimeoutError('distributed search timed out; workers may still need OS cleanup') from exc
        else:
            pool.shutdown(wait=True)
        solved=[r for r in reports if r['result']['found']]
        winner=min(solved,key=lambda item:(len(item['result']['macro_program']),
                           item['result']['simulations'],item['seed'])) if solved else None
        return {'found':bool(winner),'winner':winner,'worker_reports':sorted(reports,key=lambda x:x['seed']),
                'total_simulations':sum(r['result']['simulations'] for r in reports),
                'elapsed_seconds':time.perf_counter()-start,
                'executed_on':'local_multiprocessing_only'}
