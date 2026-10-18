"""Optional affine int64 kernel; CPU fallback is labeled, never called CUDA."""
from __future__ import annotations
import torch
try:
    import triton
    import triton.language as tl
    HAS_TRITON=True
except ImportError:
    HAS_TRITON=False


if HAS_TRITON:
    @triton.jit
    def _affine(X,Y,N:tl.constexpr,MULT:tl.constexpr,ADD:tl.constexpr,BLOCK:tl.constexpr):
        offsets=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
        values=tl.load(X+offsets,offsets<N,other=0).to(tl.int64)
        result=values*tl.full((),MULT,tl.int64)+tl.full((),ADD,tl.int64)
        tl.store(Y+offsets,result,offsets<N)


class TritonKernelCompiler:
    @staticmethod
    def compile_expression(expression,*,variable='x',backend='auto'):
        """Export a proved affine IR as a separate tensor callable, not a scalar replacement."""
        polynomial=expression.polynomial()
        if any(term not in ((),(variable,)) for term in polynomial):
            raise ValueError('native exporter currently supports only one-variable affine polynomials')
        return TritonKernelCompiler.compile_affine_transform(polynomial.get((variable,),0),polynomial.get((),0),backend=backend)

    @staticmethod
    def status():
        return dict(triton_installed=HAS_TRITON,cuda_available=torch.cuda.is_available(),
                    available=HAS_TRITON and torch.cuda.is_available(),scope='contiguous CUDA int64 affine transforms')

    @staticmethod
    def compile_affine_transform(multiplier,addend,*,backend='auto'):
        if backend not in ('auto','torch','triton'):raise ValueError('unknown affine backend')
        if any(type(v) is not int or not -(2**63)<=v<2**63 for v in (multiplier,addend)):
            raise ValueError('affine constants must fit int64')
        available=TritonKernelCompiler.status()['available']
        if backend=='triton' and not available:raise RuntimeError('Triton/CUDA unavailable')
        chosen='triton' if backend=='triton' or backend=='auto' and available else 'torch'
        def runner(x):
            if x.dtype!=torch.int64 or not x.is_contiguous():raise ValueError('contiguous int64 tensor required')
            if chosen=='triton' and x.device.type!='cuda':raise ValueError('Triton runner requires a CUDA tensor')
            if x.numel():
                low,high=int(x.min().item()),int(x.max().item())
                products=(low*multiplier,high*multiplier)
                if any(not -(2**63)<=v<2**63 for v in (*products,*(v+addend for v in products))):
                    raise OverflowError('affine intermediate/result exceeds int64')
            if chosen=='torch':return x*multiplier+addend
            y=torch.empty_like(x)
            if x.numel():_affine[(triton.cdiv(x.numel(),256),)](x,y,x.numel(),multiplier,addend,BLOCK=256)
            return y
        runner.backend=chosen
        return runner
