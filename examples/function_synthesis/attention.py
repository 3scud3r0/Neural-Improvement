"""Explicit PyTorch attention baseline for whole-function synthesis experiments."""
import math
import torch


def attention(query, key, value):
    scores = (query @ key.transpose(-2, -1)) / math.sqrt(query.size(-1))
    return torch.softmax(scores, dim=-1) @ value
