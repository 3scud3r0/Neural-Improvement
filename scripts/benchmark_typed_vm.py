"""Measure the checked stack workload; no projected CUDA throughput."""
import argparse
import json
from pathlib import Path
import platform
import statistics
import time
import torch
from nexus_m.gpu_program_evaluator import GPUBatchEvaluator
from nexus_m.triton_kernel_compiler import TritonKernelCompiler


def scalar(program,x):
    stack=[]
    for token in program:
        if token=='X':stack.append(x)
        elif token=='C0':stack.append(0)
        elif token=='C1':stack.append(1)
        elif token=='C2':stack.append(2)
        else:
            b,a=stack.pop(),stack.pop()
            value=a+b if token=='ADD' else a*b
            if not -(2**63)<=value<2**63:raise OverflowError('checked scalar overflow')
            stack.append(value)
    return stack[0]


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',default='auto')
    parser.add_argument('--programs',type=int,default=512)
    parser.add_argument('--inputs',type=int,default=128)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--threads',type=int,default=1)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args(argv)
    if min(args.programs,args.inputs,args.repeats,args.threads)<1:parser.error('positive budgets required')
    torch.set_num_threads(args.threads)
    vm=GPUBatchEvaluator(args.device)
    templates=[['X','C2','MUL','C1','ADD'],['X','C1','ADD','C2','MUL'],['X','X','MUL'],['X','C0','ADD']]
    programs=[templates[i%4] for i in range(args.programs)]
    values=[i%2000-1000 for i in range(args.inputs)]
    inputs=torch.tensor(values,dtype=torch.int64,device=vm.device);expected=2*inputs+1
    start=time.perf_counter();matrix=vm.compile_bytecode_matrix(programs)
    if vm.device.type=='cuda':torch.cuda.synchronize(vm.device)
    compile_seconds=time.perf_counter()-start
    vm.evaluate_batch(matrix,expected,inputs) # warmup
    elapsed=[]
    for _ in range(args.repeats):
        result=vm.evaluate_batch(matrix,expected,inputs);elapsed.append(result.execution_time_ms/1000)
    scalar_times=[]
    scalar_full_times=[]
    for _ in range(args.repeats):
        start=time.perf_counter()
        correct=[i for i,p in enumerate(programs) if all(scalar(p,x)==2*x+1 for x in values)]
        scalar_times.append(time.perf_counter()-start)
        start=time.perf_counter();full_correct=[]
        for index,program in enumerate(programs):
            matches=True
            for x in values:matches &= scalar(program,x)==2*x+1
            if matches:full_correct.append(index)
        scalar_full_times.append(time.perf_counter()-start)
    if correct!=result.surviving_program_indices.cpu().tolist():raise AssertionError('scalar/vector disagreement')
    if correct!=full_correct:raise AssertionError('full scalar disagreement')
    seconds=statistics.median(elapsed)
    report=dict(device=str(vm.device),python=platform.python_version(),torch=torch.__version__,
        cuda_available=torch.cuda.is_available(),triton=TritonKernelCompiler.status(),
        programs=args.programs,inputs=args.inputs,program_input_pairs=result.total_evaluations,
        distinct_programs=len({tuple(program) for program in programs}),threads=args.threads,
        instruction_slots=result.instruction_slots,survivors=len(correct),compile_seconds=compile_seconds,
        vector_seconds_median=seconds,program_input_pairs_per_second=result.total_evaluations/seconds,
        scalar_short_circuit_seconds_median=statistics.median(scalar_times),
        scalar_full_seconds_median=statistics.median(scalar_full_times),
        chunks=result.chunks,estimated_peak_chunk_bytes=result.estimated_peak_chunk_bytes,
        note='CPU/CUDA timings are measured only on the reported device. Scalar verifier may short-circuit; no universal throughput claim.')
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())
