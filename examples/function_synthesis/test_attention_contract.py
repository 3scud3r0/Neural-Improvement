"""Forward and gradient comparisons, with different training/heldout shapes."""
import math
import pytest
import torch
from examples.function_synthesis.attention import attention


def reference(query, key, value):
    return torch.softmax((query @ key.transpose(-2, -1))/math.sqrt(query.size(-1)), dim=-1) @ value


@pytest.mark.parametrize('shape', [(1, 2, 8, 8, 4), (2, 3, 11, 7, 8)])
def test_training_attention(shape):
    batch, heads, nq, nk, dim = shape
    generator = torch.Generator().manual_seed(2026)
    q = torch.randn(batch, heads, nq, dim, generator=generator)
    k = torch.randn(batch, heads, nk, dim, generator=generator)
    v = torch.randn(batch, heads, nk, dim, generator=generator)
    torch.testing.assert_close(attention(q, k, v), reference(q, k, v), atol=1e-6, rtol=1e-5)


def test_heldout_noncontiguous_and_gradients():
    torch.manual_seed(113)
    q = torch.randn(2, 4, 23, 16, dtype=torch.float64).transpose(1, 2).requires_grad_()
    k = torch.randn(2, 4, 23, 16, dtype=torch.float64).transpose(1, 2).requires_grad_()
    v = torch.randn(2, 4, 23, 16, dtype=torch.float64).transpose(1, 2).requires_grad_()
    actual, expected = attention(q, k, v), reference(q, k, v)
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-8)
    grads_a = torch.autograd.grad(actual.square().sum(), (q, k, v), retain_graph=True)
    grads_b = torch.autograd.grad(expected.square().sum(), (q, k, v))
    for a, b in zip(grads_a, grads_b):
        torch.testing.assert_close(a, b, atol=1e-9, rtol=1e-7)


def test_heldout_different_sequence_and_value_dimensions():
    torch.manual_seed(991)
    q, k, v = torch.randn(3, 9, 12), torch.randn(3, 17, 12), torch.randn(3, 17, 7)
    torch.testing.assert_close(attention(q, k, v), reference(q, k, v), atol=1e-6, rtol=1e-5)
