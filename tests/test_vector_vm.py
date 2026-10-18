import random
import pytest
import torch
from nexus_m.gpu_program_evaluator import (
    GPUBatchEvaluator,ERROR_STACK,ERROR_OVERFLOW,ERROR_DIV_ZERO,MIN_INT)
from nexus_m.gpu_falsifier import GPUAdversarialFalsifier
from nexus_m.triton_kernel_compiler import TritonKernelCompiler


def scalar(program,x,depth=16):
    stack=[]
    for token in program:
        if token=='NOP':continue
        if token in ('X','C0','C1','C2','C-1'):
            if len(stack)>=depth:return None,ERROR_STACK
            stack.append({'X':x,'C0':0,'C1':1,'C2':2,'C-1':-1}[token]);continue
        if len(stack)<2:return None,ERROR_STACK
        b,a=stack.pop(),stack.pop()
        if token=='DIV' and b==0:return None,ERROR_DIV_ZERO
        value={'ADD':lambda:a+b,'SUB':lambda:a-b,'MUL':lambda:a*b,'DIV':lambda:a//b,
               'MIN':lambda:min(a,b),'MAX':lambda:max(a,b)}[token]()
        if not -(2**63)<=value<2**63:return None,ERROR_OVERFLOW
        stack.append(value)
    return (stack[0],0) if len(stack)==1 else (None,ERROR_STACK)


def test_all_opcodes_match_checked_python_including_nops_and_exceptions():
    rng=random.Random(5)
    inputs=[MIN_INT,MIN_INT+1,-100,-3,-1,0,1,2,100,2**63-1]+[rng.randrange(-(2**62),2**62) for _ in range(20)]
    programs=[['X'],['X','NOP'],['C0'],['C-1'],[],['ADD'],['X','X'],['X','C0','DIV']]
    for operation in ('ADD','SUB','MUL','DIV','MIN','MAX'):
        programs.extend([['X','C-1',operation],['C-1','X',operation],['X','X',operation]])
    vm=GPUBatchEvaluator('cpu',program_chunk=3,input_chunk=7)
    run=vm.raw_outputs(vm.compile_bytecode_matrix(programs),torch.tensor(inputs,dtype=torch.int64))
    for p,program in enumerate(programs):
        for i,x in enumerate(inputs):
            value,error=scalar(program,x)
            assert run.errors[p,i].item()==error,(program,x,value,error)
            if not error:assert run.outputs[p,i].item()==value,(program,x)


def test_regression_non_push_lanes_do_not_overwrite_stack_or_accept_zero_division():
    vm=GPUBatchEvaluator('cpu')
    programs=[['X','C2','MUL','C1','ADD'],['X','C0','ADD'],['X','C0','DIV']]
    inputs=torch.tensor([-3,0,4],dtype=torch.int64)
    result=vm.evaluate_batch(vm.compile_bytecode_matrix(programs),2*inputs+1,inputs)
    assert result.surviving_program_indices.tolist()==[0]
    zero=vm.evaluate_batch(vm.compile_bytecode_matrix([['X','C0','DIV']]),torch.zeros_like(inputs),inputs)
    assert zero.surviving_program_indices.numel()==0


def test_structural_overflow_unknown_and_truncated_programs_rejected():
    vm=GPUBatchEvaluator('cpu',stack_depth=1)
    inputs=torch.tensor([0],dtype=torch.int64)
    run=vm.raw_outputs(vm.compile_bytecode_matrix([['X','X','ADD']]),inputs)
    assert run.errors.item()==ERROR_STACK
    with pytest.raises(ValueError):vm.compile_bytecode_matrix([['X','INVALID']])
    with pytest.raises(ValueError):vm.compile_bytecode_matrix([['X','C1','ADD']],max_len=2)
    with pytest.raises(ValueError):vm.evaluate_batch(torch.tensor([[99]],dtype=torch.int64),inputs,inputs)
    with pytest.raises(ValueError):vm.evaluate_batch(vm.compile_bytecode_matrix([['X']]),torch.tensor([],dtype=torch.int64),torch.tensor([],dtype=torch.int64))


def test_chunked_comparison_preserves_global_indices_and_witnesses():
    vm=GPUBatchEvaluator('cpu',max_memory_bytes=8192,program_chunk=2,input_chunk=3)
    programs=[['X'],['X','C1','ADD'],['X','NOP'],['X','C2','MUL']]
    inputs=torch.arange(-8,9,dtype=torch.int64)
    result=vm.evaluate_batch(vm.compile_bytecode_matrix(programs),inputs,inputs)
    assert result.surviving_program_indices.tolist()==[0,2]
    assert result.first_mismatch_indices.tolist()==[-1,0,-1,0]
    assert result.chunks>1 and result.estimated_peak_chunk_bytes<=8192
    assert result.total_evaluations==len(programs)*len(inputs)
    assert result.evaluations_per_second==pytest.approx(result.total_evaluations/(result.execution_time_ms/1000))
    with pytest.raises(ValueError):vm.raw_outputs(vm.compile_bytecode_matrix([['X']]*100),inputs)


def test_falsifier_uses_full_reference_semantics_and_reproducible_cases():
    a=GPUAdversarialFalsifier('cpu',seed=42)
    b=GPUAdversarialFalsifier('cpu',seed=42)
    assert torch.equal(a.generate_adversarial_tensor(20),b.generate_adversarial_tensor(20))
    assert len(a.generate_adversarial_tensor(3))==3
    # Reference uses subtraction, floor division and a negative constant.
    reference=['X','C1','SUB','C-1','DIV']
    result=a.falsify_candidates(reference,[reference,['X'],['X','C0','DIV']],batch_size=30)
    assert result.tolist()==[0]
    undefined=['X','C0','DIV']
    assert a.falsify_candidates(undefined,[undefined,['C0']],batch_size=4).tolist()==[0]
    with pytest.raises(ValueError,match='reference'):
        a.falsify_candidates(['ADD'],[['C0']],batch_size=4)


def test_affine_fallback_is_labeled_and_checks_overflow():
    runner=TritonKernelCompiler.compile_affine_transform(2,1,backend='torch')
    values=torch.tensor([-5,0,5],dtype=torch.int64)
    assert runner.backend=='torch' and torch.equal(runner(values),2*values+1)
    with pytest.raises(OverflowError):runner(torch.tensor([2**63-1],dtype=torch.int64))
    with pytest.raises(ValueError):runner(values.float())
    if not TritonKernelCompiler.status()['available']:
        with pytest.raises(RuntimeError):TritonKernelCompiler.compile_affine_transform(2,1,backend='triton')
    import ast
    from nexus_m.typed_program import IntExpr
    affine=IntExpr.from_ast(ast.parse('x+x+1',mode='eval').body)
    exported=TritonKernelCompiler.compile_expression(affine,backend='torch')
    assert torch.equal(exported(values),values*2+1)
    with pytest.raises(ValueError):TritonKernelCompiler.compile_expression(IntExpr('mul',(IntExpr('var',value='x'),)*2))


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA hardware unavailable')
def test_cuda_vm_matches_cpu_and_reports_synchronized_elapsed_time():
    programs=[['X','C2','MUL','C1','ADD'],['X','C-1','DIV'],['X','C0','DIV']]
    x=torch.tensor([MIN_INT,-7,0,11,2**63-1],dtype=torch.int64)
    cpu=GPUBatchEvaluator('cpu');gpu=GPUBatchEvaluator('cuda')
    a=cpu.raw_outputs(cpu.compile_bytecode_matrix(programs),x)
    b=gpu.raw_outputs(gpu.compile_bytecode_matrix(programs),x.cuda())
    assert torch.equal(a.outputs,b.outputs.cpu()) and torch.equal(a.errors,b.errors.cpu())
    result=gpu.evaluate_batch(gpu.compile_bytecode_matrix([['X']]),x.cuda(),x.cuda())
    assert result.execution_time_ms>0 and result.surviving_program_indices.tolist()==[0]


@pytest.mark.skipif(not TritonKernelCompiler.status()['available'],reason='Triton/CUDA unavailable')
def test_triton_affine_matches_checked_torch_on_non_power_of_two_size():
    values=torch.arange(-777,888,dtype=torch.int64,device='cuda')
    runner=TritonKernelCompiler.compile_affine_transform(-3,7,backend='triton')
    assert torch.equal(runner(values),values*-3+7)
