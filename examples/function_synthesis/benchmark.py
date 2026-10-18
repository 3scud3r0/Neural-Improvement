"""Measure calls inside the process, excluding Python/PyTorch startup time."""
import argparse
import json
import math
import os
from pathlib import Path
import random
import statistics
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('target', choices=('attention', 'mcts'))
    args = parser.parse_args()
    samples = []
    seed = int(os.environ.get('NEXUS_EVALUATION_SEED', '2026'))
    if args.target == 'attention':
        import torch
        from examples.function_synthesis.attention import attention
        torch.set_num_threads(1)
        generator = torch.Generator().manual_seed(seed)
        q, k, v = [torch.randn(2, 4, 128, 32, generator=generator) for _ in range(3)]
        with torch.no_grad():
            for _ in range(10): attention(q, k, v)
            for _ in range(5):
                start = time.perf_counter()
                for _ in range(120): attention(q, k, v)
                samples.append(time.perf_counter()-start)
    else:
        from nexus_m.mcts import SharedMCTS, SearchNode
        search = SharedMCTS(model=None)
        rng = random.Random(seed)
        parent = SearchNode((), visits=200)
        parent.children = {str(i): SearchNode((str(i),), visits=rng.randrange(100),
            prior=rng.random(), score_sum=rng.random()*10) for i in range(64)}
        for _ in range(100): search._select(parent)
        for _ in range(5):
            start = time.process_time()
            for _ in range(5000): search._select(parent)
            samples.append(time.process_time()-start)
    Path(os.environ['NEXUS_BENCHMARK_FILE']).write_text(
        json.dumps({'seconds':statistics.median(samples)}), encoding='utf-8')


if __name__ == '__main__': main()
