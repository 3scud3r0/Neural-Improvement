"""Chunked tensor stack VM with checked int64 semantics on CPU or CUDA.

This accelerates a small integer bytecode, not arbitrary Python ASTs. PyTorch
operations are not assumed fused or stored in GPU shared memory. Overflow,
division by zero and structural errors are explicit invalid outcomes.
"""
from __future__ import annotations
from dataclasses import dataclass
import time
import torch

OP_NOP,OP_PUSH_X,OP_CONST_0,OP_CONST_1,OP_CONST_2,OP_CONST_NEG1=range(6)
OP_ADD,OP_SUB,OP_MUL,OP_FLOORDIV,OP_MIN,OP_MAX=range(6,12)
OP_MAP={'NOP':0,'X':1,'C0':2,'C1':3,'C2':4,'C-1':5,
        'ADD':6,'SUB':7,'MUL':8,'DIV':9,'MIN':10,'MAX':11}
ERROR_NONE,ERROR_DIV_ZERO,ERROR_OVERFLOW,ERROR_STACK=0,1,2,3
MIN_INT=-(2**63)


@dataclass
class VMExecution:
    outputs: torch.Tensor
    errors: torch.Tensor
    @property
    def valid(self):return self.errors==0


@dataclass
class GPUEvaluationResult:
    surviving_program_indices: torch.Tensor
    execution_time_ms: float
    total_evaluations: int
    evaluations_per_second: float
    first_mismatch_indices: torch.Tensor
    chunks: int
    estimated_peak_chunk_bytes: int
    instruction_slots: int
    backend: str


class GPUBatchEvaluator:
    def __init__(self,device='auto',stack_depth=16,*,max_memory_bytes=128*1024**2,
                 program_chunk=256,input_chunk=1024):
        if device=='auto':device='cuda' if torch.cuda.is_available() else 'cpu'
        self.device=torch.device(device)
        if self.device.type not in ('cpu','cuda'):raise ValueError('supported VM devices are CPU and CUDA')
        if self.device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA is unavailable')
        if self.device.type=='cuda' and self.device.index is None:self.device=torch.device('cuda',torch.cuda.current_device())
        if min(stack_depth,max_memory_bytes,program_chunk,input_chunk)<1:raise ValueError('positive VM budgets required')
        self.stack_depth=stack_depth;self.max_memory_bytes=max_memory_bytes
        self.program_chunk=program_chunk;self.input_chunk=input_chunk
        # Conservative working-set estimate; allocator/kernel overhead is extra.
        self.bytes_per_cell=8*(stack_depth+40)
        if max_memory_bytes<self.bytes_per_cell:raise ValueError('VM memory budget is too small')

    def compile_bytecode_matrix(self,tokenized_programs,max_len=None):
        programs=[list(p) for p in tokenized_programs]
        length=max((len(p) for p in programs),default=0) if max_len is None else max_len
        if length<0:raise ValueError('max_len must be nonnegative')
        rows=[]
        for program in programs:
            if any(not isinstance(token,str) for token in program):raise ValueError('bytecode tokens must be strings')
            if len(program)>length:raise ValueError('program exceeds max_len; truncation is forbidden')
            unknown=set(program)-set(OP_MAP)
            if unknown:raise ValueError(f'unknown tokens: {sorted(unknown)}')
            rows.append([OP_MAP[token] for token in program]+[OP_NOP]*(length-len(program)))
        if not programs:return torch.empty((0,length),dtype=torch.int16,device=self.device)
        return torch.tensor(rows,dtype=torch.int16,device=self.device).reshape(len(rows),length)

    def _validate(self,matrix,inputs):
        if matrix.ndim!=2 or matrix.dtype not in (torch.int16,torch.int32,torch.int64):raise ValueError('bytecode must be a 2D integer tensor')
        if inputs.ndim!=1 or inputs.dtype!=torch.int64 or inputs.numel()==0:raise ValueError('inputs must be a nonempty int64 vector')
        if matrix.device!=self.device or inputs.device!=self.device:raise ValueError('VM tensors must be on the configured device')
        if matrix.numel() and bool(((matrix<0)|(matrix>OP_MAX)).any()):raise ValueError('unknown opcode')

    def _dimensions(self,n_inputs,reserved=0):
        cells=(self.max_memory_bytes-reserved)//self.bytes_per_cell
        if cells<1:raise ValueError('insufficient working memory after reserving raw outputs')
        inputs=min(self.input_chunk,n_inputs,cells)
        programs=min(self.program_chunk,max(1,cells//inputs))
        return programs,inputs

    def _blocks(self,matrix,inputs,reserved=0):
        pchunk,ichunk=self._dimensions(len(inputs),reserved)
        for p in range(0,matrix.shape[0],pchunk):
            for i in range(0,len(inputs),ichunk):
                yield p,i,matrix[p:p+pchunk],inputs[i:i+ichunk]

    def _run_chunk(self,matrix,inputs):
        p,length=matrix.shape;m=len(inputs)
        stack=torch.zeros((self.stack_depth,p,m),dtype=torch.int64,device=self.device)
        pointer=torch.zeros(p,dtype=torch.int64,device=self.device)
        errors=torch.zeros((p,m),dtype=torch.int8,device=self.device)
        x=inputs.unsqueeze(0).expand(p,m)
        def index(position):return position.clamp(0,self.stack_depth-1).view(1,p,1).expand(1,p,m)
        def write(position,mask,values):
            indices=index(position)
            old=stack.gather(0,indices)
            stack.scatter_(0,indices,torch.where(mask.view(1,p,1),values.unsqueeze(0),old))
        for step in range(length):
            op=matrix[:,step]
            push=(op>=OP_PUSH_X)&(op<=OP_CONST_NEG1)
            binary=(op>=OP_ADD)&(op<=OP_MAX)
            bad_stack=(push&(pointer>=self.stack_depth))|(binary&(pointer<2))
            errors=torch.where((errors==0)&bad_stack.unsqueeze(1),ERROR_STACK,errors)
            can_push=push&(pointer<self.stack_depth)
            values=torch.zeros_like(x)
            values=torch.where((op==OP_PUSH_X).unsqueeze(1),x,values)
            values=torch.where((op==OP_CONST_1).unsqueeze(1),1,values)
            values=torch.where((op==OP_CONST_2).unsqueeze(1),2,values)
            values=torch.where((op==OP_CONST_NEG1).unsqueeze(1),-1,values)
            # Non-push lanes preserve the existing stack slot, including NOPs.
            write(pointer,can_push,values)
            pointer=torch.where(can_push,pointer+1,pointer)
            a=stack.gather(0,index(pointer-2)).squeeze(0)
            b=stack.gather(0,index(pointer-1)).squeeze(0)
            add=a+b;sub=a-b;mul=a*b
            add_over=((a^add)&(b^add))<0
            sub_over=((a^b)&(a^sub))<0
            mul_bad_div=(mul==MIN_INT)&(b==-1)
            mul_denominator=torch.where((b==0)|mul_bad_div,1,b)
            mul_over=(b!=0)&((torch.div(mul,mul_denominator,rounding_mode='floor')!=a)|mul_bad_div)
            div_over=(a==MIN_INT)&(b==-1)
            denominator=torch.where((b==0)|div_over,1,b)
            divided=torch.div(a,denominator,rounding_mode='floor')
            result=torch.zeros_like(a)
            for opcode,value in ((OP_ADD,add),(OP_SUB,sub),(OP_MUL,mul),
                                 (OP_FLOORDIV,divided),(OP_MIN,torch.minimum(a,b)),(OP_MAX,torch.maximum(a,b))):
                result=torch.where((op==opcode).unsqueeze(1),value,result)
            overflow=((op==OP_ADD).unsqueeze(1)&add_over)|((op==OP_SUB).unsqueeze(1)&sub_over)|(
                (op==OP_MUL).unsqueeze(1)&mul_over)|((op==OP_FLOORDIV).unsqueeze(1)&div_over)
            zero=(op==OP_FLOORDIV).unsqueeze(1)&(b==0)
            errors=torch.where((errors==0)&overflow,ERROR_OVERFLOW,errors)
            errors=torch.where((errors==0)&zero,ERROR_DIV_ZERO,errors)
            can_binary=binary&(pointer>=2)
            write(pointer-2,can_binary,torch.where(errors==0,result,0))
            pointer=torch.where(can_binary,pointer-1,pointer)
        errors=torch.where((errors==0)&(pointer!=1).unsqueeze(1),ERROR_STACK,errors)
        outputs=stack.gather(0,index(pointer-1)).squeeze(0)
        return VMExecution(outputs,errors)

    @torch.inference_mode()
    def raw_outputs(self,program_matrix,inputs):
        self._validate(program_matrix,inputs)
        size=program_matrix.shape[0]*len(inputs)
        output_bytes=size*9
        if output_bytes>self.max_memory_bytes//2:raise ValueError('raw result exceeds memory budget; use streaming evaluate_batch')
        outputs=torch.empty((program_matrix.shape[0],len(inputs)),dtype=torch.int64,device=self.device)
        errors=torch.empty_like(outputs,dtype=torch.int8)
        for p,i,matrix,values in self._blocks(program_matrix,inputs,output_bytes):
            run=self._run_chunk(matrix,values)
            outputs[p:p+len(matrix),i:i+len(values)]=run.outputs
            errors[p:p+len(matrix),i:i+len(values)]=run.errors
        return VMExecution(outputs,errors)

    @torch.inference_mode()
    def evaluate_batch(self,program_matrix,reference_outputs,inputs,*,reference_errors=None):
        self._validate(program_matrix,inputs)
        if reference_outputs.shape!=inputs.shape or reference_outputs.dtype!=torch.int64 or reference_outputs.device!=self.device:
            raise ValueError('reference outputs must match the int64 input vector shape and device')
        if reference_errors is None:reference_errors=torch.zeros_like(inputs,dtype=torch.int8)
        if reference_errors.shape!=inputs.shape or reference_errors.device!=self.device or reference_errors.dtype not in (torch.int8,torch.int16,torch.int32,torch.int64):raise ValueError('invalid reference errors')
        if bool(((reference_errors<0)|(reference_errors>ERROR_STACK)).any()):raise ValueError('unknown reference error')
        if self.device.type=='cuda':torch.cuda.synchronize(self.device)
        started=time.perf_counter()
        count=program_matrix.shape[0]
        passed=torch.ones(count,dtype=torch.bool,device=self.device)
        mismatch=torch.full((count,),-1,dtype=torch.int64,device=self.device)
        chunks=0;peak=0
        for p,i,matrix,values in self._blocks(program_matrix,inputs):
            run=self._run_chunk(matrix,values);chunks+=1
            expected=reference_outputs[i:i+len(values)].unsqueeze(0)
            expected_errors=reference_errors[i:i+len(values)].unsqueeze(0)
            matches=(run.errors==expected_errors)&((expected_errors!=0)|(run.outputs==expected))
            good=matches.all(dim=1)
            first=(~matches).to(torch.int64).argmax(dim=1)+i
            old=mismatch[p:p+len(matrix)]
            mismatch[p:p+len(matrix)]=torch.where((old<0)&~good,first,old)
            passed[p:p+len(matrix)]&=good
            peak=max(peak,len(matrix)*len(values)*self.bytes_per_cell)
        survivors=torch.nonzero(passed).flatten()
        if self.device.type=='cuda':torch.cuda.synchronize(self.device)
        elapsed=(time.perf_counter()-started)*1000
        pairs=count*len(inputs)
        return GPUEvaluationResult(survivors,elapsed,pairs,pairs/max(elapsed/1000,1e-12),
            mismatch,chunks,peak,pairs*program_matrix.shape[1],f'torch-vectorized-{self.device.type}')
