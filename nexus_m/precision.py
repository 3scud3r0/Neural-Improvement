"""Adaptive numeric precision: FP32/FP64 nets, FP64 search, exact DSL,
Decimal arbitrary-precision sandbox for fractional calculations.

Extra decimal digits never imply additional learned information.
"""
from __future__ import annotations
from decimal import Decimal, localcontext
import torch


def network_dtype(precision: str):
    if precision in ('float32','adaptive'):return torch.float32
    if precision=='float64': return torch.float64
    raise ValueError('Choose float32, float64 or adaptive; arbitrary precision is in exact verifier')


def high_precision_decimal(value: str, digits=80) -> str:
    if not 16<=digits<=1000: raise ValueError('digits must be 16..1000')
    with localcontext() as ctx:
        ctx.prec=digits
        return format(Decimal(value)*Decimal(1), 'f')


def decimal_matmul_2x2(a,b,digits=80):
    """Small illustrative exact-decimal numeric kernel, NOT a PyTorch trainable layer."""
    if len(a)!=2 or len(b)!=2 or any(len(r)!=2 for r in a+b):
        raise ValueError('Expected two 2x2 matrices')
    with localcontext() as ctx:
        ctx.prec=digits
        aa=[[Decimal(str(v)) for v in row] for row in a]
        bb=[[Decimal(str(v)) for v in row] for row in b]
        return [[str(sum(aa[i][k]*bb[k][j] for k in range(2))) for j in range(2)] for i in range(2)]


def describe_precision(mode):
    return {'float32':'~7 significant decimal digits in neural tensors',
            'float64':'~16 significant decimal digits in neural tensors',
            'adaptive':'neural FP32 + search statistics FP64 + exact integer DSL + optional Decimal' }[mode]
