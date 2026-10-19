"""Finite adversarial input batches using the SAME checked VM for both sides."""
from __future__ import annotations
import torch
from .gpu_program_evaluator import GPUBatchEvaluator


class GPUAdversarialFalsifier:
    def __init__(self,device='auto',*,seed=2026,**vm_options):
        self.evaluator=GPUBatchEvaluator(device=device,**vm_options)
        self.device=self.evaluator.device
        self.generator=torch.Generator(device=self.device).manual_seed(seed)
        self.last_result=None

    def generate_adversarial_tensor(self,batch_size=16384):
        if batch_size<1:raise ValueError('batch_size must be positive')
        extremes=torch.tensor([0,1,-1,2,-2,10,-10,255,-256,32767,-32768,
                               10**6,-10**6,2**63-1,-2**63],dtype=torch.int64,device=self.device)
        if batch_size<=len(extremes):return extremes[:batch_size]
        samples=torch.randint(-10**6,10**6,(batch_size-len(extremes),),
            dtype=torch.int64,device=self.device,generator=self.generator)
        return torch.cat((extremes,samples))

    def falsify_candidates(self,reference_tokens,candidates_tokens,batch_size=16384):
        inputs=self.generate_adversarial_tensor(batch_size)
        reference=self.evaluator.raw_outputs(self.evaluator.compile_bytecode_matrix([reference_tokens]),inputs)
        from .gpu_program_evaluator import ERROR_STACK
        if bool((reference.errors==ERROR_STACK).any()):raise ValueError('reference has invalid bytecode stack structure')
        matrix=self.evaluator.compile_bytecode_matrix(candidates_tokens)
        result=self.evaluator.evaluate_batch(matrix,reference.outputs[0],inputs,reference_errors=reference.errors[0])
        self.last_result=dict(inputs=inputs,evaluation=result,
            scope='checked int64 outcomes on this finite input batch, not universal equivalence')
        return result.surviving_program_indices

    def _extract_raw_output(self,program_matrix,inputs):
        result=self.evaluator.raw_outputs(program_matrix,inputs)
        if program_matrix.shape[0]!=1:raise ValueError('one reference program required')
        return result.outputs[0]
