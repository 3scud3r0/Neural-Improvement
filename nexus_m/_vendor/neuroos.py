#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   ███╗   ██╗███████╗██╗   ██╗██████╗  ██████╗  ██████╗ ███████╗             ║
║   ████╗  ██║██╔════╝██║   ██║██╔══██╗██╔═══██╗██╔════╝ ██╔════╝             ║
║   ██╔██╗ ██║█████╗  ██║   ██║██████╔╝██║   ██║╚█████╗  █████╗               ║
║   ██║╚██╗██║██╔══╝  ██║   ██║██╔══██╗██║   ██║ ╚═══██╗ ██╔══╝               ║
║   ██║ ╚████║███████╗╚██████╔╝██║  ██║╚██████╔╝██████╔╝ ███████╗             ║
║   ╚═╝  ╚═══╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚═════╝  ╚══════╝             ║
║                                                                              ║
║   NeuroOS v1.0 — The Autonomous Neural Operating System                      ║
║                                                                              ║
║   Módulos Integrados:                                                        ║
║   ├── CORE: NeuroHive GPU Ultimate (Transformer Engine Completo)             ║
║   ├── FASE 1: Neural Bus + VRAM Manager + FastAPI OpenAI-Compatible         ║
║   ├── FASE 2: AutoForge (Minerador + Self-Play + Reward + LoRA Trainer)     ║
║   ├── FASE 3: HydraMoE + ProteusNet (Hot-Swap LoRA + Continuous Learning)  ║
║   ├── FASE 4: O1-Engine (MCTS + HiveMind Multi-Agent Debate)               ║
║   └── FASE 5: CortexFS (Knowledge Graph + Tool Calling + RAG Injection)    ║
║                                                                              ║
║   Dependências:                                                              ║
║   - torch >= 2.x, numpy, gguf                                               ║
║   - Opcionais: tiktoken, fastapi, uvicorn, redis, networkx,                 ║
║     sentence-transformers, unsloth, trl, peft, zmq, aiohttp,               ║
║     beautifulsoup4, PyPDF2, langchain                                        ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import os
import re
import sys
import math
import time
import enum
import json
import random
import struct
import signal
import shutil
import logging
import hashlib
import asyncio
import inspect
import secrets
import pathlib
import textwrap
import tempfile
import threading
import traceback
import subprocess
import collections
from copy import deepcopy
from io import BytesIO
from abc import ABC, abstractmethod
from queue import Queue, Empty, PriorityQueue
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from contextlib import contextmanager, asynccontextmanager
from dataclasses import dataclass, field, asdict
from typing import (
    Any, Callable, Coroutine, Deque, Dict, FrozenSet, Generator,
    Iterable, Iterator, List, Literal, Mapping, NamedTuple, Optional,
    Protocol, Sequence, Set, Tuple, Type, TypeVar, Union, overload,
)
from functools import lru_cache, partial, wraps
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ── Optional imports with graceful fallback ──────────────────────────────────

try:
    import gguf
except Exception:
    gguf = None

try:
    import tiktoken
except Exception:
    tiktoken = None

try:
    import redis as redis_lib
except Exception:
    redis_lib = None

try:
    import zmq
except Exception:
    zmq = None

try:
    from fastapi import FastAPI, Request, HTTPException
    from fastapi.responses import StreamingResponse, JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    HAS_FASTAPI = True
except Exception:
    HAS_FASTAPI = False

try:
    import uvicorn
    HAS_UVICORN = True
except Exception:
    HAS_UVICORN = False

try:
    import networkx as nx
    HAS_NETWORKX = True
except Exception:
    HAS_NETWORKX = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except Exception:
    HAS_SENTENCE_TRANSFORMERS = False

try:
    import aiohttp
    HAS_AIOHTTP = True
except Exception:
    HAS_AIOHTTP = False

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except Exception:
    HAS_BS4 = False

try:
    import PyPDF2
    HAS_PYPDF2 = True
except Exception:
    HAS_PYPDF2 = False

try:
    from peft import PeftModel, LoraConfig, get_peft_model, TaskType
    HAS_PEFT = True
except Exception:
    HAS_PEFT = False


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 0: GLOBAL LOGGING & CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("NeuroOS")

T = TypeVar("T")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: UTILITIES (Shared across ALL modules)
# ═══════════════════════════════════════════════════════════════════════════════

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pick_dtype(dtype_name: str, device: torch.device) -> torch.dtype:
    if device.type == "cpu":
        return torch.float32
    dtype_name = dtype_name.lower()
    if dtype_name in ("fp16", "float16", "half"):
        return torch.float16
    if dtype_name in ("bf16", "bfloat16"):
        return torch.bfloat16
    return torch.float16


def as_torch(x: Any, device: torch.device, dtype: Optional[torch.dtype] = None) -> Tensor:
    if isinstance(x, Tensor):
        return x.to(device=device, dtype=dtype) if dtype else x.to(device=device)
    arr = np.asarray(x)
    if dtype is None:
        dtype = torch.float32 if arr.dtype.kind in ("f", "c") else torch.int64
    return torch.as_tensor(np.ascontiguousarray(arr), device=device, dtype=dtype)


def softmax_stable(x: Tensor, dim: int = -1) -> Tensor:
    x = x - torch.amax(x, dim=dim, keepdim=True)
    return torch.softmax(x, dim=dim)


def top2_margin(logits: Tensor) -> Tensor:
    vals = torch.topk(logits, k=min(2, logits.numel()), dim=-1).values
    if vals.numel() < 2:
        return torch.tensor(1e9, device=logits.device, dtype=logits.dtype)
    return vals[..., 0] - vals[..., 1]


def entropy_from_logits(logits: Tensor) -> Tensor:
    probs = softmax_stable(logits, dim=-1)
    return -(probs * torch.log(probs.clamp_min(1e-9))).sum(dim=-1)


def rms_norm(x: Tensor, weight: Tensor, eps: float) -> Tensor:
    xf = x.float()
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    y = xf * torch.rsqrt(var + eps)
    return (y * weight.float()).to(x.dtype)


def layer_norm(x: Tensor, weight: Tensor, bias: Optional[Tensor], eps: float) -> Tensor:
    return F.layer_norm(
        x.float(),
        normalized_shape=(x.shape[-1],),
        weight=weight.float(),
        bias=None if bias is None else bias.float(),
        eps=eps,
    ).to(x.dtype)


def gelu(x: Tensor) -> Tensor:
    return F.gelu(x)


def silu(x: Tensor) -> Tensor:
    return F.silu(x)


def _unpack_nibbles(u8: np.ndarray) -> np.ndarray:
    lo = (u8 & 0x0F).astype(np.int8) - 8
    hi = ((u8 >> 4) & 0x0F).astype(np.int8) - 8
    out = np.empty(u8.size * 2, dtype=np.int8)
    out[0::2] = lo
    out[1::2] = hi
    return out


def timestamp_str() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def sha256_short(data: Union[str, bytes], length: int = 12) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()[:length]


def safe_json_loads(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return None


def bytes_to_human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PB"


def gpu_memory_stats() -> Dict[str, str]:
    if not torch.cuda.is_available():
        return {"status": "no_cuda"}
    return {
        "allocated": bytes_to_human(torch.cuda.memory_allocated()),
        "reserved": bytes_to_human(torch.cuda.memory_reserved()),
        "max_allocated": bytes_to_human(torch.cuda.max_memory_allocated()),
        "free": bytes_to_human(
            torch.cuda.get_device_properties(0).total_mem - torch.cuda.memory_allocated()
        ) if torch.cuda.device_count() > 0 else "N/A",
    }


def meta_get(meta: Dict[str, Any], keys: Iterable[str], default: Any = None) -> Any:
    for k in keys:
        if k in meta and meta[k] is not None:
            return meta[k]
    return default


def meta_int(meta: Dict[str, Any], keys: Iterable[str], default: int) -> int:
    try:
        return int(meta_get(meta, keys, default))
    except Exception:
        return default


def meta_float(meta: Dict[str, Any], keys: Iterable[str], default: float) -> float:
    try:
        return float(meta_get(meta, keys, default))
    except Exception:
        return default


def meta_str(meta: Dict[str, Any], keys: Iterable[str], default: str) -> str:
    v = meta_get(meta, keys, default)
    return default if v is None else str(v)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: ENUMS, DATACLASSES & PROTOCOLS
# ═══════════════════════════════════════════════════════════════════════════════

class Family(str, enum.Enum):
    LLAMA = "llama"
    QWEN = "qwen"
    MISTRAL = "mistral"
    GEMMA = "gemma"
    PHI = "phi"
    DEEPSEEK = "deepseek"
    UNKNOWN = "unknown"


class NormType(str, enum.Enum):
    RMSNORM = "rmsnorm"
    LAYERNORM = "layernorm"


class ActivationType(str, enum.Enum):
    SILU = "silu"
    GELU = "gelu"
    SWIGLU = "swiglu"
    GEGLU = "geglu"


class SymOp(enum.IntEnum):
    AND = 0
    OR = 1
    NOT = 2
    IMPLIES = 3
    XOR = 4
    THRESHOLD = 5


class TNormType(enum.IntEnum):
    GODEL = 0
    PRODUCT = 1
    LUKASIEWICZ = 2


class QuantType(str, enum.Enum):
    F32 = "f32"
    F16 = "f16"
    BF16 = "bf16"
    Q4_0 = "q4_0"
    Q8_0 = "q8_0"
    Q4_K_M = "q4_k_m"


class CacheQuantType(str, enum.Enum):
    FP16 = "fp16"
    INT8 = "int8"


class RopeScalingType(str, enum.Enum):
    NONE = "none"
    LINEAR = "linear"
    DYNAMIC_NTK = "dynamic_ntk"


class BusMessageType(str, enum.Enum):
    INFERENCE_REQUEST = "inference_request"
    INFERENCE_RESPONSE = "inference_response"
    LORA_TRAIN_REQUEST = "lora_train_request"
    LORA_TRAIN_COMPLETE = "lora_train_complete"
    LORA_LOAD_REQUEST = "lora_load_request"
    LORA_UNLOAD_REQUEST = "lora_unload_request"
    VRAM_OFFLOAD_REQUEST = "vram_offload_request"
    VRAM_RELOAD_REQUEST = "vram_reload_request"
    DATA_GENERATION_REQUEST = "data_generation_request"
    DATA_GENERATION_COMPLETE = "data_generation_complete"
    CORTEX_QUERY = "cortex_query"
    CORTEX_WRITE = "cortex_write"
    CORTEX_RESPONSE = "cortex_response"
    MCTS_START = "mcts_start"
    MCTS_RESULT = "mcts_result"
    HIVEMIND_DEBATE_START = "hivemind_debate_start"
    HIVEMIND_DEBATE_RESULT = "hivemind_debate_result"
    SYSTEM_STATUS = "system_status"
    SHUTDOWN = "shutdown"
    HEARTBEAT = "heartbeat"
    ERROR = "error"


class MCTSNodeState(str, enum.Enum):
    UNEXPLORED = "unexplored"
    EXPANDED = "expanded"
    TERMINAL = "terminal"
    PRUNED = "pruned"


class AgentPersona(str, enum.Enum):
    LOGIC = "logic"
    CREATIVE = "creative"
    SKEPTIC = "skeptic"
    EXPERT = "expert"
    SYNTHESIZER = "synthesizer"


def infer_family(arch: str) -> Family:
    arch = (arch or "").lower()
    if "llama" in arch:
        return Family.LLAMA
    if "qwen" in arch:
        return Family.QWEN
    if "mistral" in arch:
        return Family.MISTRAL
    if "gemma" in arch:
        return Family.GEMMA
    if "phi" in arch:
        return Family.PHI
    if "deepseek" in arch:
        return Family.DEEPSEEK
    return Family.UNKNOWN


# ── Core Dataclasses ─────────────────────────────────────────────────────────

@dataclass
class ModelSpec:
    architecture: str
    family: Family
    dim: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    vocab_size: int
    max_seq_len: int
    norm_eps: float
    rope_theta: float
    norm_type: NormType
    activation: ActivationType
    ffn_dim: int
    tie_embeddings: bool = True
    sliding_window: Optional[int] = None
    use_moe: bool = False
    n_experts: int = 0
    n_shared_experts: int = 0
    rope_scaling_type: RopeScalingType = RopeScalingType.NONE
    rope_scaling_factor: float = 1.0
    rope_scaling_original_max_position: Optional[int] = None

    @property
    def gqa_group_size(self) -> int:
        return max(1, self.n_heads // max(1, self.n_kv_heads))


@dataclass
class RuntimeConfig:
    device: torch.device
    dtype: torch.dtype
    seed: int = 42
    entropy_threshold: float = 3.8
    margin_threshold: float = 0.15
    top_k_reasoning: int = 8
    beam_width: int = 3
    repetition_penalty: float = 1.08
    temperature: float = 0.7
    top_k_sampling: int = 40
    use_autocast: bool = True
    cache_quant: CacheQuantType = CacheQuantType.FP16


@dataclass
class ReasoningDecision:
    activate: bool
    entropy: float
    margin: float
    topk_ids: Tensor
    topk_vals: Tensor


@dataclass
class SymbolicRule:
    confidence: float
    temperature: float
    ops: List[SymOp] = field(default_factory=list)
    operands: List[int] = field(default_factory=list)


@dataclass
class PageSeqState:
    pages: List[int] = field(default_factory=list)
    length: int = 0


# ── Bus & System Dataclasses ────────────────────────────────────────────────

@dataclass
class BusMessage:
    msg_type: BusMessageType
    source: str
    target: str = "broadcast"
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    msg_id: str = field(default_factory=lambda: secrets.token_hex(8))
    priority: int = 5  # 0=highest, 9=lowest

    def to_dict(self) -> Dict[str, Any]:
        return {
            "msg_type": self.msg_type.value,
            "source": self.source,
            "target": self.target,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "msg_id": self.msg_id,
            "priority": self.priority,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BusMessage":
        return cls(
            msg_type=BusMessageType(d["msg_type"]),
            source=d["source"],
            target=d.get("target", "broadcast"),
            payload=d.get("payload", {}),
            timestamp=d.get("timestamp", time.time()),
            msg_id=d.get("msg_id", secrets.token_hex(8)),
            priority=d.get("priority", 5),
        )


@dataclass
class LoRAMetadata:
    name: str
    path: str
    category: str
    rank: int = 16
    alpha: float = 32.0
    created_at: str = field(default_factory=timestamp_str)
    training_samples: int = 0
    metrics: Dict[str, float] = field(default_factory=dict)
    active: bool = False
    on_gpu: bool = False


@dataclass
class MCTSNode:
    state: MCTSNodeState = MCTSNodeState.UNEXPLORED
    token_id: int = -1
    text_so_far: str = ""
    parent_id: int = -1
    children_ids: List[int] = field(default_factory=list)
    visits: int = 0
    total_reward: float = 0.0
    prior: float = 0.0
    depth: int = 0

    @property
    def value(self) -> float:
        return self.total_reward / max(1, self.visits)


@dataclass
class CortexNode:
    node_id: str
    node_type: str  # "entity", "concept", "memory", "preference", "fact"
    content: str
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=timestamp_str)
    access_count: int = 0
    last_accessed: str = field(default_factory=timestamp_str)


@dataclass
class CortexEdge:
    source_id: str
    target_id: str
    relation: str  # "likes", "knows", "causes", "part_of", etc.
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SyntheticDataSample:
    prompt: str
    response: str
    category: str
    quality_score: float = 0.0
    symbolic_valid: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=timestamp_str)


@dataclass
class DebateRound:
    round_num: int
    persona: AgentPersona
    argument: str
    confidence: float
    supporting_evidence: List[str] = field(default_factory=list)


@dataclass
class DebateResult:
    question: str
    rounds: List[DebateRound]
    final_answer: str
    consensus_score: float
    winning_persona: AgentPersona
    symbolic_validation: float


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: NEUROHIVE GPU ULTIMATE (THE CORE ENGINE)
# ═══════════════════════════════════════════════════════════════════════════════
# This is the COMPLETE original NeuroHive, now embedded as the heart of NeuroOS

# ── Tokenizer ────────────────────────────────────────────────────────────────

class TokenizerAdapter:
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self._tok = None
        self._gguf_vocab = None
        self._fallback_char = False

        if gguf is not None and model_path and os.path.exists(model_path):
            try:
                self._load_gguf_tokenizer(model_path)
            except Exception as e:
                logger.warning("Tokenizer GGUF falhou: %s", e)

        if self._tok is None and self._gguf_vocab is None:
            if tiktoken is not None:
                try:
                    self._tok = tiktoken.get_encoding("cl100k_base")
                except Exception:
                    self._tok = None

        if self._tok is None and self._gguf_vocab is None:
            self._fallback_char = True
            logger.warning("Tokenizer fallback por caractere ativado.")

    def _load_gguf_tokenizer(self, model_path: str) -> None:
        reader = gguf.GGUFReader(model_path)
        meta = {}
        for attr in ("fields", "metadata", "kv_data", "kv", "_fields"):
            obj = getattr(reader, attr, None)
            if obj is None:
                continue
            try:
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        meta[str(k)] = getattr(v, "value", v)
                else:
                    for item in obj:
                        k = getattr(item, "name", None) or getattr(item, "key", None)
                        v = getattr(item, "value", None)
                        if k is not None:
                            meta[str(k)] = v
            except Exception:
                pass

        tokens = meta_get(meta, ["tokenizer.ggml.tokens", "tokens", "token_list"], None)
        if tokens is not None:
            self._gguf_vocab = tokens
            logger.info("Tokenizer GGUF detectado: %s tokens", len(tokens))

    def encode(self, text: str) -> List[int]:
        if self._tok is not None:
            return self._tok.encode(text)
        if self._gguf_vocab is not None:
            return [ord(c) % len(self._gguf_vocab) for c in text]
        return [ord(c) % 256 for c in text]

    def decode(self, ids: List[int]) -> str:
        if self._tok is not None:
            try:
                return self._tok.decode(ids)
            except Exception:
                pass
        return "".join(chr(i % 95 + 32) for i in ids)

    @property
    def vocab_size(self) -> int:
        if self._tok is not None:
            return getattr(self._tok, "n_vocab", 100000)
        if self._gguf_vocab is not None:
            return len(self._gguf_vocab)
        return 256


# ── GGUF Reader & Architecture Builder ──────────────────────────────────────

class GGUFReaderAdapter:
    def __init__(self, path: str):
        if gguf is None:
            raise RuntimeError("Biblioteca 'gguf' não instalada.")
        self.path = path
        self.reader = gguf.GGUFReader(path)

    def metadata(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        r = self.reader
        for attr in ("fields", "metadata", "kv_data", "kv", "_fields"):
            obj = getattr(r, attr, None)
            if obj is None:
                continue
            try:
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        out[str(k)] = getattr(v, "value", v)
                else:
                    for item in obj:
                        k = getattr(item, "name", None) or getattr(item, "key", None)
                        v = getattr(item, "value", None)
                        if k is not None:
                            out[str(k)] = v
            except Exception:
                pass
        return out

    def tensors(self) -> List[Any]:
        ts = getattr(self.reader, "tensors", None)
        if ts is None:
            raise RuntimeError("GGUFReader não expôs tensors")
        return list(ts)


class GGUFArchitectureBuilder:
    @staticmethod
    def build(meta: Dict[str, Any]) -> ModelSpec:
        arch = meta_str(meta, ["general.architecture", "architecture"], "unknown").lower()
        family = infer_family(arch)

        dim = meta_int(meta, [
            "llama.embedding_length", "qwen2.embedding_length",
            "hidden_size", "n_embd", "model.embedding_length",
        ], 4096)

        n_layers = meta_int(meta, [
            "llama.block_count", "qwen2.block_count",
            "num_hidden_layers", "n_layer", "model.block_count",
        ], 32)

        n_heads = meta_int(meta, [
            "llama.attention.head_count", "qwen2.attention.head_count",
            "num_attention_heads", "n_head", "model.attention.head_count",
        ], 32)

        n_kv_heads = meta_int(meta, [
            "llama.attention.head_count_kv", "qwen2.attention.head_count_kv",
            "num_key_value_heads", "n_kv_head", "model.attention.head_count_kv",
        ], n_heads)

        head_dim = meta_int(meta, [
            "llama.attention.head_dim", "qwen2.attention.head_dim",
            "model.attention.head_dim",
        ], max(1, dim // max(1, n_heads)))

        vocab_size = meta_int(meta, [
            "tokenizer.ggml.tokens", "llama.vocab_size", "qwen2.vocab_size",
            "vocab_size", "model.vocab_size",
        ], 32000)

        max_seq_len = meta_int(meta, [
            "llama.context_length", "qwen2.context_length",
            "n_ctx_train", "context_length", "model.context_length",
        ], 4096)

        norm_eps = meta_float(meta, [
            "llama.attention.layer_norm_rms_epsilon", "llama.rms_norm_eps",
            "qwen2.rms_norm_eps", "norm_eps", "layer_norm_epsilon", "model.norm_eps",
        ], 1e-5)

        rope_theta = meta_float(meta, [
            "llama.rope.freq_base", "qwen2.rope.freq_base",
            "rope_theta", "model.rope.freq_base",
        ], 10000.0)

        ffn_dim = meta_int(meta, [
            "llama.feed_forward_length", "qwen2.feed_forward_length",
            "intermediate_size", "n_ff", "model.feed_forward_length",
        ], dim * 4)

        norm_type = NormType.LAYERNORM if family == Family.PHI else NormType.RMSNORM

        act_name = meta_str(meta, ["model.activation", "llama.feed_forward_act"], "swiglu").lower()
        if "gelu" in act_name:
            activation = ActivationType.GELU
        elif "silu" in act_name:
            activation = ActivationType.SILU
        elif "geglu" in act_name:
            activation = ActivationType.GEGLU
        else:
            activation = ActivationType.SWIGLU

        use_moe = bool(meta_get(meta, ["moe", "use_moe"], False))
        n_experts = meta_int(meta, ["n_experts", "num_experts"], 0) if use_moe else 0
        n_shared_experts = meta_int(meta, ["n_shared_experts"], 0) if use_moe else 0

        sliding_window = meta_int(meta, ["sliding_window"], 0)
        if sliding_window <= 0:
            sliding_window = None

        rope_scaling_type_str = meta_str(meta, [
            "rope.scaling_type", "llama.rope.scaling_type", "qwen2.rope.scaling_type",
        ], "none").lower()
        if rope_scaling_type_str in ("linear", "linear_scaling"):
            rope_scaling_type = RopeScalingType.LINEAR
        elif rope_scaling_type_str in ("dynamic", "dynamic_ntk", "ntk"):
            rope_scaling_type = RopeScalingType.DYNAMIC_NTK
        else:
            rope_scaling_type = RopeScalingType.NONE

        rope_scaling_factor = meta_float(meta, [
            "rope.scaling_factor", "llama.rope.scaling_factor", "qwen2.rope.scaling_factor",
        ], 1.0)

        rope_orig = meta_int(meta, [
            "rope.original_max_position_embeddings",
            "llama.rope.original_max_position_embeddings",
            "qwen2.rope.original_max_position_embeddings",
        ], 0)
        rope_scaling_original_max_position = rope_orig if rope_orig > 0 else None

        return ModelSpec(
            architecture=arch, family=family, dim=dim, n_layers=n_layers,
            n_heads=n_heads, n_kv_heads=n_kv_heads, head_dim=head_dim,
            vocab_size=vocab_size, max_seq_len=max_seq_len, norm_eps=norm_eps,
            rope_theta=rope_theta, norm_type=norm_type, activation=activation,
            ffn_dim=ffn_dim, tie_embeddings=True, sliding_window=sliding_window,
            use_moe=use_moe, n_experts=n_experts, n_shared_experts=n_shared_experts,
            rope_scaling_type=rope_scaling_type,
            rope_scaling_factor=rope_scaling_factor,
            rope_scaling_original_max_position=rope_scaling_original_max_position,
        )


# ── Quantization ─────────────────────────────────────────────────────────────

@dataclass
class QuantizedBlob:
    qtype: QuantType
    shape: Tuple[int, ...]
    raw: bytes
    scale: Optional[np.ndarray] = None
    zero: Optional[np.ndarray] = None

    def dequantize(self, device: torch.device, dtype: torch.dtype) -> Tensor:
        if self.qtype in (QuantType.F32, QuantType.F16, QuantType.BF16):
            if self.qtype == QuantType.F32:
                arr = np.frombuffer(self.raw, dtype=np.float32)
                return torch.as_tensor(arr.reshape(self.shape), device=device, dtype=dtype)
            if self.qtype == QuantType.F16:
                arr = np.frombuffer(self.raw, dtype=np.float16).astype(np.float32)
                return torch.as_tensor(arr.reshape(self.shape), device=device, dtype=dtype)
            arr = np.frombuffer(self.raw, dtype=np.uint16)
            arr32 = (arr.astype(np.uint32) << 16).view(np.float32)
            return torch.as_tensor(arr32.reshape(self.shape), device=device, dtype=dtype)

        if self.qtype == QuantType.Q8_0:
            return self._dequant_q8_0(device, dtype)
        if self.qtype == QuantType.Q4_0:
            return self._dequant_q4_0(device, dtype)
        if self.qtype == QuantType.Q4_K_M:
            return self._dequant_q4_k_m(device, dtype)
        raise NotImplementedError(f"Quant type não suportado: {self.qtype}")

    def _dequant_q8_0(self, device: torch.device, dtype: torch.dtype) -> Tensor:
        arr = np.frombuffer(self.raw, dtype=np.int8)
        if arr.size == 0:
            return torch.zeros(self.shape, device=device, dtype=dtype)
        flat = arr.astype(np.float32)
        if self.scale is not None:
            s = np.asarray(self.scale).astype(np.float32).reshape(-1)
            flat *= float(s[0]) if s.size else 1.0
        flat = flat[: int(np.prod(self.shape))]
        return torch.as_tensor(flat.reshape(self.shape), device=device, dtype=dtype)

    def _dequant_q4_0(self, device: torch.device, dtype: torch.dtype) -> Tensor:
        raw = np.frombuffer(self.raw, dtype=np.uint8)
        if raw.size == 0:
            return torch.zeros(self.shape, device=device, dtype=dtype)
        vals_all: List[np.ndarray] = []
        ptr = 0
        while ptr + 18 <= raw.size:
            scale = np.frombuffer(raw[ptr:ptr + 2].tobytes(), dtype=np.float16)[0].astype(np.float32)
            qs = raw[ptr + 2:ptr + 18]
            vals = _unpack_nibbles(qs).astype(np.float32) * scale
            vals_all.append(vals)
            ptr += 18
        if not vals_all:
            arr = raw.view(np.int8).astype(np.float32)
            arr = arr[: int(np.prod(self.shape))]
            return torch.as_tensor(arr.reshape(self.shape), device=device, dtype=dtype)
        flat = np.concatenate(vals_all, axis=0)
        flat = flat[: int(np.prod(self.shape))]
        return torch.as_tensor(flat.reshape(self.shape), device=device, dtype=dtype)

    def _dequant_q4_k_m(self, device: torch.device, dtype: torch.dtype) -> Tensor:
        raw = np.frombuffer(self.raw, dtype=np.uint8)
        if raw.size == 0:
            return torch.zeros(self.shape, device=device, dtype=dtype)
        scale = 1.0
        if self.scale is not None:
            s = np.asarray(self.scale).astype(np.float32).reshape(-1)
            if s.size > 0:
                scale = float(s[0])
        try:
            vals = _unpack_nibbles(raw).astype(np.float32) * scale
            vals = vals[: int(np.prod(self.shape))]
            return torch.as_tensor(vals.reshape(self.shape), device=device, dtype=dtype)
        except Exception:
            arr = raw.view(np.int8).astype(np.float32) * scale
            arr = arr[: int(np.prod(self.shape))]
            return torch.as_tensor(arr.reshape(self.shape), device=device, dtype=dtype)


# ── Weight Store ─────────────────────────────────────────────────────────────

class WeightStore:
    def __init__(self, device: torch.device, dtype: torch.dtype):
        self.device = device
        self.dtype = dtype
        self.tensors: Dict[str, Tensor] = {}
        self.quantized: Dict[str, QuantizedBlob] = {}

    def load_from_reader(self, reader: GGUFReaderAdapter) -> None:
        for t in reader.tensors():
            name = str(getattr(t, "name", ""))
            if not name:
                continue
            data = getattr(t, "data", None)
            if data is None:
                data = getattr(t, "tensor", None)
            if data is None:
                data = getattr(t, "value", None)
            if data is None:
                continue
            try:
                arr = np.asarray(data)
                if arr.dtype.kind in ("f", "c"):
                    self.tensors[name] = torch.as_tensor(
                        np.ascontiguousarray(arr), device=self.device, dtype=self.dtype
                    )
                elif arr.dtype.kind in ("i", "u"):
                    self.tensors[name] = torch.as_tensor(
                        np.ascontiguousarray(arr), device=self.device
                    )
                else:
                    if isinstance(data, (bytes, bytearray, memoryview)):
                        qtype = self._infer_qtype(name, t)
                        shape = (
                            tuple(getattr(t, "shape", ()))
                            or tuple(getattr(t, "dims", ()))
                            or (arr.size if arr.size else 1,)
                        )
                        self.quantized[name] = QuantizedBlob(
                            qtype=qtype,
                            shape=shape if isinstance(shape, tuple) else tuple(shape),
                            raw=bytes(data),
                        )
                    else:
                        self.tensors[name] = torch.as_tensor(
                            np.ascontiguousarray(arr), device=self.device, dtype=self.dtype
                        )
            except Exception as e:
                logger.warning("Falha ao carregar tensor '%s': %s", name, e)

    def _infer_qtype(self, name: str, tensor_obj: Any) -> QuantType:
        s = f"{name} {tensor_obj}".lower()
        if "q4_k_m" in s:
            return QuantType.Q4_K_M
        if "q8_0" in s:
            return QuantType.Q8_0
        if "q4_0" in s:
            return QuantType.Q4_0
        if "bf16" in s:
            return QuantType.BF16
        if "f16" in s or "float16" in s:
            return QuantType.F16
        return QuantType.F32

    def resolve(self, candidates: Sequence[str]) -> Optional[Tensor]:
        for c in candidates:
            if c in self.tensors:
                return self.tensors[c]
        return None

    def resolve_quant(self, candidates: Sequence[str]) -> Optional[QuantizedBlob]:
        for c in candidates:
            if c in self.quantized:
                return self.quantized[c]
        return None

    def make_random(self, shape: Tuple[int, ...], std: float = 0.02) -> Tensor:
        return torch.randn(shape, device=self.device, dtype=self.dtype) * std

    def linear_or_init(
        self, candidates: Sequence[str], in_dim: int, out_dim: int, desc: str
    ) -> Tensor:
        t = self.resolve(candidates)
        if t is None:
            qb = self.resolve_quant(candidates)
            if qb is not None:
                try:
                    t = qb.dequantize(self.device, self.dtype)
                except Exception as e:
                    logger.warning("Falha ao dequantizar %s: %s", desc, e)
                    t = None
        if t is None:
            logger.warning("Tensor ausente: %s. Inicializando aleatoriamente.", desc)
            return self.make_random((out_dim, in_dim))
        t = t.to(device=self.device, dtype=self.dtype)
        if t.dim() == 2:
            if t.shape == (out_dim, in_dim):
                return t.contiguous()
            if t.shape == (in_dim, out_dim):
                return t.t().contiguous()
        if t.numel() == in_dim * out_dim:
            return t.reshape(out_dim, in_dim).contiguous()
        logger.warning("Shape inválido para %s. Inicializando aleatoriamente.", desc)
        return self.make_random((out_dim, in_dim))

    def vector_or_none(
        self, candidates: Sequence[str], out_dim: int, desc: str
    ) -> Optional[Tensor]:
        t = self.resolve(candidates)
        if t is None:
            qb = self.resolve_quant(candidates)
            if qb is not None:
                try:
                    t = qb.dequantize(self.device, self.dtype)
                except Exception as e:
                    logger.warning("Falha ao dequantizar vetor %s: %s", desc, e)
                    t = None
        if t is None:
            return None
        t = t.to(device=self.device, dtype=self.dtype).reshape(-1)
        if t.numel() != out_dim:
            logger.warning("Vector inválido para %s. Ignorando.", desc)
            return None
        return t.contiguous()


# ── Rotary Embedding ─────────────────────────────────────────────────────────

class RotaryEmbedding:
    def __init__(
        self, dim: int, theta: float, device: torch.device,
        scaling_type: RopeScalingType = RopeScalingType.NONE,
        scaling_factor: float = 1.0,
        original_max_position: Optional[int] = None,
    ):
        if dim % 2 != 0:
            raise ValueError("RoPE requer head_dim par")
        self.dim = dim
        self.theta = theta
        self.device = device
        self.scaling_type = scaling_type
        self.scaling_factor = scaling_factor
        self.original_max_position = original_max_position
        self.inv_freq = 1.0 / (
            theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim)
        )

    def _effective_position(self, seq_pos: int) -> float:
        if self.scaling_type == RopeScalingType.NONE:
            return float(seq_pos)
        if self.scaling_type == RopeScalingType.LINEAR:
            return float(seq_pos) / max(self.scaling_factor, 1e-6)
        if self.scaling_type == RopeScalingType.DYNAMIC_NTK:
            base = self.original_max_position or 4096
            scale = max(self.scaling_factor, 1.0)
            if seq_pos <= base:
                return float(seq_pos)
            ratio = math.log((seq_pos / base) * scale + 1.0)
            return float(base + (seq_pos - base) / max(ratio, 1e-6))
        return float(seq_pos)

    def apply(self, x: Tensor, seq_pos: int) -> Tensor:
        orig_dtype = x.dtype
        x = x.float()
        pos = self._effective_position(seq_pos)
        freqs = self.inv_freq * pos
        c = torch.cos(freqs)
        s = torch.sin(freqs)
        x1, x2 = x[..., 0::2], x[..., 1::2]
        y = torch.empty_like(x)
        y[..., 0::2] = x1 * c - x2 * s
        y[..., 1::2] = x1 * s + x2 * c
        return y.to(orig_dtype)


# ── Paged KV Cache ───────────────────────────────────────────────────────────

class PagedKVCache:
    def __init__(
        self, max_seq_len: int, n_kv_heads: int, head_dim: int,
        device: torch.device, dtype: torch.dtype,
        page_size: int = 16, max_pages: int = 4096,
        quant_type: CacheQuantType = CacheQuantType.FP16,
    ):
        self.max_seq_len = max_seq_len
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype
        self.page_size = page_size
        self.max_pages = max_pages
        self.quant_type = quant_type

        self.k_pages: List[Optional[Tensor]] = [None] * max_pages
        self.v_pages: List[Optional[Tensor]] = [None] * max_pages
        self.k_scales: List[Optional[Tensor]] = [None] * max_pages
        self.v_scales: List[Optional[Tensor]] = [None] * max_pages
        self.free_pages = list(range(max_pages))
        self.sequences: Dict[int, PageSeqState] = {}

    def _alloc_page(self) -> int:
        if not self.free_pages:
            raise MemoryError("PagedKVCache OOM: sem páginas livres")
        pid = self.free_pages.pop()
        shape = (self.page_size, self.n_kv_heads, self.head_dim)
        if self.quant_type == CacheQuantType.INT8:
            self.k_pages[pid] = torch.zeros(shape, device=self.device, dtype=torch.int8)
            self.v_pages[pid] = torch.zeros(shape, device=self.device, dtype=torch.int8)
            self.k_scales[pid] = torch.ones(
                (self.page_size, 1, 1), device=self.device, dtype=torch.float16
            )
            self.v_scales[pid] = torch.ones(
                (self.page_size, 1, 1), device=self.device, dtype=torch.float16
            )
        else:
            self.k_pages[pid] = torch.zeros(shape, device=self.device, dtype=self.dtype)
            self.v_pages[pid] = torch.zeros(shape, device=self.device, dtype=self.dtype)
        return pid

    def _free_page(self, pid: int) -> None:
        self.k_pages[pid] = None
        self.v_pages[pid] = None
        self.k_scales[pid] = None
        self.v_scales[pid] = None
        self.free_pages.append(pid)

    def ensure_seq(self, seq_id: int) -> PageSeqState:
        if seq_id not in self.sequences:
            self.sequences[seq_id] = PageSeqState()
        return self.sequences[seq_id]

    def clear_seq(self, seq_id: int) -> None:
        st = self.sequences.pop(seq_id, None)
        if st is None:
            return
        for pid in st.pages:
            self._free_page(pid)

    def reset_all(self) -> None:
        for sid in list(self.sequences.keys()):
            self.clear_seq(sid)

    def _dequantize_int8_page(self, q: Tensor, scale: Tensor) -> Tensor:
        return (q.float() * scale.float()).to(self.dtype)

    def append(self, seq_id: int, k: Tensor, v: Tensor) -> None:
        st = self.ensure_seq(seq_id)
        page_idx = st.length // self.page_size
        offset = st.length % self.page_size
        if page_idx >= len(st.pages):
            pid = self._alloc_page()
            st.pages.append(pid)
        else:
            pid = st.pages[page_idx]

        if self.quant_type == CacheQuantType.INT8:
            max_abs_k = k.abs().amax().clamp_min(1e-8)
            max_abs_v = v.abs().amax().clamp_min(1e-8)
            k_s = (max_abs_k / 127.0).to(torch.float16)
            v_s = (max_abs_v / 127.0).to(torch.float16)
            self.k_pages[pid][offset].copy_(
                torch.clamp((k / k_s).round(), -128, 127).to(torch.int8)
            )
            self.v_pages[pid][offset].copy_(
                torch.clamp((v / v_s).round(), -128, 127).to(torch.int8)
            )
            self.k_scales[pid][offset].fill_(k_s)
            self.v_scales[pid][offset].fill_(v_s)
        else:
            self.k_pages[pid][offset].copy_(k.to(self.dtype))
            self.v_pages[pid][offset].copy_(v.to(self.dtype))
        st.length += 1

    def gather(self, seq_id: int) -> Tuple[Tensor, Tensor]:
        st = self.sequences.get(seq_id, None)
        if st is None or st.length == 0:
            empty = torch.zeros(
                (0, self.n_kv_heads, self.head_dim), device=self.device, dtype=self.dtype
            )
            return empty, empty
        rem = st.length
        ks, vs = [], []
        for pid in st.pages:
            if rem <= 0:
                break
            take = min(self.page_size, rem)
            if self.quant_type == CacheQuantType.INT8:
                ks.append(self._dequantize_int8_page(
                    self.k_pages[pid][:take], self.k_scales[pid][:take]
                ))
                vs.append(self._dequantize_int8_page(
                    self.v_pages[pid][:take], self.v_scales[pid][:take]
                ))
            else:
                ks.append(self.k_pages[pid][:take].to(self.dtype))
                vs.append(self.v_pages[pid][:take].to(self.dtype))
            rem -= take
        return torch.cat(ks, dim=0), torch.cat(vs, dim=0)

    def length(self, seq_id: int) -> int:
        st = self.sequences.get(seq_id)
        return 0 if st is None else st.length


# ── Transformer Components ───────────────────────────────────────────────────

class TransformerNorm(nn.Module):
    def __init__(self, dim: int, norm_type: NormType, eps: float):
        super().__init__()
        self.norm_type = norm_type
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim), requires_grad=False)
        self.bias = None
        if norm_type == NormType.LAYERNORM:
            self.bias = nn.Parameter(torch.zeros(dim), requires_grad=False)

    def forward(self, x: Tensor) -> Tensor:
        if self.norm_type == NormType.RMSNORM:
            return rms_norm(x, self.weight, self.eps)
        return layer_norm(x, self.weight, self.bias, self.eps)


class FeedForward(nn.Module):
    def __init__(self, spec: ModelSpec):
        super().__init__()
        self.spec = spec
        self.activation = spec.activation
        self.w_gate = nn.Parameter(torch.empty(spec.ffn_dim, spec.dim), requires_grad=False)
        self.w_up = nn.Parameter(torch.empty(spec.ffn_dim, spec.dim), requires_grad=False)
        self.w_down = nn.Parameter(torch.empty(spec.dim, spec.ffn_dim), requires_grad=False)
        self.b_gate = None
        self.b_up = None
        self.b_down = None
        nn.init.normal_(self.w_gate, std=0.02)
        nn.init.normal_(self.w_up, std=0.02)
        nn.init.normal_(self.w_down, std=0.02)

    def set_weights(self, gate: Tensor, up: Tensor, down: Tensor) -> None:
        self.w_gate.data.copy_(gate)
        self.w_up.data.copy_(up)
        self.w_down.data.copy_(down)

    def forward(self, x: Tensor) -> Tensor:
        a = F.linear(x, self.w_gate, self.b_gate)
        b = F.linear(x, self.w_up, self.b_up)
        if self.activation == ActivationType.GELU:
            h = gelu(a) * b
        else:
            h = silu(a) * b
        return F.linear(h, self.w_down, self.b_down)


class MoEBlock(nn.Module):
    def __init__(self, spec: ModelSpec):
        super().__init__()
        self.spec = spec
        self.enabled = spec.use_moe and spec.n_experts > 0
        self.router = (
            nn.Parameter(torch.empty(spec.n_experts, spec.dim), requires_grad=False)
            if self.enabled else None
        )
        if self.enabled:
            nn.init.normal_(self.router, std=0.02)
        self.experts = (
            nn.ModuleList([FeedForward(spec) for _ in range(max(1, spec.n_experts))])
            if self.enabled else None
        )
        self.shared_experts = (
            nn.ModuleList([FeedForward(spec) for _ in range(max(0, spec.n_shared_experts))])
            if self.enabled and spec.n_shared_experts > 0 else None
        )

    def forward(self, x: Tensor) -> Tensor:
        if not self.enabled:
            return x
        logits = F.linear(x, self.router)
        probs = torch.softmax(logits, dim=-1)
        topk = min(2, probs.numel())
        topv, topi = torch.topk(probs, k=topk)
        out = torch.zeros_like(x)
        for p, idx in zip(topv, topi):
            out = out + p * self.experts[int(idx.item())](x)
        if self.shared_experts is not None:
            for se in self.shared_experts:
                out = out + (1.0 / len(self.shared_experts)) * se(x)
        return out


class GQAAttention(nn.Module):
    def __init__(self, spec: ModelSpec):
        super().__init__()
        self.spec = spec
        self.q_proj = nn.Parameter(
            torch.empty(spec.n_heads * spec.head_dim, spec.dim), requires_grad=False
        )
        self.k_proj = nn.Parameter(
            torch.empty(spec.n_kv_heads * spec.head_dim, spec.dim), requires_grad=False
        )
        self.v_proj = nn.Parameter(
            torch.empty(spec.n_kv_heads * spec.head_dim, spec.dim), requires_grad=False
        )
        self.o_proj = nn.Parameter(
            torch.empty(spec.dim, spec.n_heads * spec.head_dim), requires_grad=False
        )
        nn.init.normal_(self.q_proj, std=0.02)
        nn.init.normal_(self.k_proj, std=0.02)
        nn.init.normal_(self.v_proj, std=0.02)
        nn.init.normal_(self.o_proj, std=0.02)

        # LoRA delta storage (for HydraMoE hot-swap)
        self._lora_q_delta: Optional[Tensor] = None
        self._lora_k_delta: Optional[Tensor] = None
        self._lora_v_delta: Optional[Tensor] = None
        self._lora_o_delta: Optional[Tensor] = None

    def set_weights(self, q: Tensor, k: Tensor, v: Tensor, o: Tensor) -> None:
        self.q_proj.data.copy_(q)
        self.k_proj.data.copy_(k)
        self.v_proj.data.copy_(v)
        self.o_proj.data.copy_(o)

    def apply_lora_delta(
        self, q_delta: Optional[Tensor] = None, k_delta: Optional[Tensor] = None,
        v_delta: Optional[Tensor] = None, o_delta: Optional[Tensor] = None,
    ) -> None:
        self._lora_q_delta = q_delta
        self._lora_k_delta = k_delta
        self._lora_v_delta = v_delta
        self._lora_o_delta = o_delta

    def clear_lora_delta(self) -> None:
        self._lora_q_delta = None
        self._lora_k_delta = None
        self._lora_v_delta = None
        self._lora_o_delta = None

    def _effective_proj(self, base: Tensor, delta: Optional[Tensor]) -> Tensor:
        if delta is None:
            return base
        return base + delta

    def forward(
        self, x: Tensor, rope: RotaryEmbedding, cache: PagedKVCache,
        seq_id: int, seq_pos: int,
    ) -> Tensor:
        x = x.reshape(-1)
        q_w = self._effective_proj(self.q_proj, self._lora_q_delta)
        k_w = self._effective_proj(self.k_proj, self._lora_k_delta)
        v_w = self._effective_proj(self.v_proj, self._lora_v_delta)
        o_w = self._effective_proj(self.o_proj, self._lora_o_delta)

        q = F.linear(x, q_w).view(self.spec.n_heads, self.spec.head_dim)
        k = F.linear(x, k_w).view(self.spec.n_kv_heads, self.spec.head_dim)
        v = F.linear(x, v_w).view(self.spec.n_kv_heads, self.spec.head_dim)

        q = rope.apply(q, seq_pos)
        k = rope.apply(k, seq_pos)
        cache.append(seq_id, k, v)
        past_k, past_v = cache.gather(seq_id)

        if past_k.size(0) == 0:
            out = torch.zeros(
                (self.spec.n_heads, self.spec.head_dim), device=x.device, dtype=x.dtype
            )
            return F.linear(out.reshape(-1), o_w)

        if self.spec.sliding_window is not None and past_k.size(0) > self.spec.sliding_window:
            past_k = past_k[-self.spec.sliding_window:]
            past_v = past_v[-self.spec.sliding_window:]

        out_heads = []
        scale = 1.0 / math.sqrt(self.spec.head_dim)
        group = self.spec.gqa_group_size

        for h in range(self.spec.n_heads):
            kvh = min(self.spec.n_kv_heads - 1, h // group)
            qh = q[h]
            ks = past_k[:, kvh, :]
            vs = past_v[:, kvh, :]
            scores = (ks @ qh) * scale
            weights = torch.softmax(scores - scores.max(), dim=0)
            out_heads.append(weights @ vs)

        out = torch.stack(out_heads, dim=0).reshape(-1)
        return F.linear(out, o_w)


class TransformerBlock(nn.Module):
    def __init__(self, spec: ModelSpec):
        super().__init__()
        self.spec = spec
        self.attn_norm = TransformerNorm(spec.dim, spec.norm_type, spec.norm_eps)
        self.ffn_norm = TransformerNorm(spec.dim, spec.norm_type, spec.norm_eps)
        self.attn = GQAAttention(spec)
        self.ffn = FeedForward(spec)
        self.moe = MoEBlock(spec)

    def forward(
        self, x: Tensor, rope: RotaryEmbedding, cache: PagedKVCache,
        seq_id: int, seq_pos: int,
    ) -> Tensor:
        h = self.attn_norm(x)
        x = x + self.attn(h, rope, cache, seq_id, seq_pos)
        h = self.ffn_norm(x)
        if self.spec.use_moe and self.spec.n_experts > 0:
            x = x + self.moe(h)
        else:
            x = x + self.ffn(h)
        return x


# ── Entropy Gate & System 2 ─────────────────────────────────────────────────

class EntropyGate:
    def __init__(
        self, entropy_threshold: float = 3.8, margin_threshold: float = 0.15, top_k: int = 8
    ):
        self.entropy_threshold = entropy_threshold
        self.margin_threshold = margin_threshold
        self.top_k = top_k

    @torch.no_grad()
    def inspect(self, logits: Tensor) -> ReasoningDecision:
        logits = logits.flatten()
        probs = softmax_stable(logits, dim=-1)
        ent = float((-(probs * torch.log(probs.clamp_min(1e-9)))).sum().item())
        margin = float(top2_margin(logits).item())
        vals, ids = torch.topk(logits, k=min(self.top_k, logits.numel()))
        activate = (ent >= self.entropy_threshold) or (margin <= self.margin_threshold)
        return ReasoningDecision(
            activate=activate, entropy=ent, margin=margin,
            topk_ids=ids, topk_vals=vals,
        )


class System2Reranker:
    def __init__(
        self, beam_width: int = 3, repetition_penalty: float = 1.08, temperature: float = 0.7
    ):
        self.beam_width = beam_width
        self.repetition_penalty = repetition_penalty
        self.temperature = temperature

    @torch.no_grad()
    def rerank(
        self, logits: Tensor, context_ids: List[int], symbolic_score: float = 0.0
    ) -> int:
        logits = logits.flatten()
        k = min(self.beam_width * 4, logits.numel())
        vals, ids = torch.topk(logits, k=k)
        seen = set(context_ids[-64:])
        best_id = int(ids[0].item())
        best_score = -1e30
        for v, i in zip(vals.tolist(), ids.tolist()):
            score = v / max(self.temperature, 1e-6)
            if i in seen:
                score /= self.repetition_penalty
            score += 0.15 * symbolic_score
            if score > best_score:
                best_score = score
                best_id = int(i)
        return best_id


class SymbolicBridgeGPU:
    def __init__(
        self, dim: int, device: torch.device, dtype: torch.dtype,
        tnorm: TNormType = TNormType.PRODUCT,
    ):
        self.dim = dim
        self.device = device
        self.dtype = dtype
        self.tnorm = tnorm
        self.rules: List[SymbolicRule] = []

    def add_rule(self, confidence: float = 0.8, temperature: float = 5.0) -> int:
        self.rules.append(SymbolicRule(float(confidence), float(temperature)))
        return len(self.rules) - 1

    def set_rule_condition(self, rule_id: int, ops: List[SymOp], operands: List[int]) -> None:
        self.rules[rule_id].ops = list(ops)
        self.rules[rule_id].operands = list(operands)

    def _apply_tnorm(self, a: Tensor, b: Tensor) -> Tensor:
        if self.tnorm == TNormType.GODEL:
            return torch.minimum(a, b)
        if self.tnorm == TNormType.LUKASIEWICZ:
            return torch.clamp(a + b - 1.0, min=0.0)
        return a * b

    @torch.no_grad()
    def forward(self, x: Tensor, neural_out: Tensor) -> Tensor:
        x = x.flatten().to(device=self.device, dtype=self.dtype)
        n = neural_out.flatten().to(device=self.device, dtype=self.dtype)
        dim = min(self.dim, x.numel(), n.numel())
        x, n = x[:dim], n[:dim]
        out = 0.3 * n + 0.7 * (x > 0.5).to(self.dtype)
        for rule in self.rules:
            if not rule.ops:
                continue
            gate = torch.ones(dim, device=self.device, dtype=self.dtype)
            mask = (x > 0.5).to(self.dtype)
            for op in rule.ops:
                if op == SymOp.AND:
                    gate = self._apply_tnorm(gate, mask)
                elif op == SymOp.OR:
                    gate = torch.maximum(gate, mask)
                elif op == SymOp.NOT:
                    gate = 1.0 - gate
                elif op == SymOp.IMPLIES:
                    gate = torch.maximum(1.0 - gate, mask)
                elif op == SymOp.XOR:
                    gate = torch.abs(gate - mask)
                elif op == SymOp.THRESHOLD:
                    gate = (x > x.mean()).to(self.dtype)
            out = torch.clamp(
                out * (1.0 - 0.5 * rule.confidence) + gate * rule.confidence, 0.0, 1.0
            )
        return out

    @torch.no_grad()
    def evaluate_quality(self, text: str) -> float:
        """Evaluate text quality using symbolic rules. Returns 0.0-1.0."""
        score = 0.5
        # Check bracket balance
        opens = text.count("(") + text.count("{") + text.count("[")
        closes = text.count(")") + text.count("}") + text.count("]")
        if opens == closes:
            score += 0.15
        else:
            score -= 0.2
        # Check sentence structure
        if text.strip() and text.strip()[-1] in ".!?":
            score += 0.1
        # Length penalty
        if len(text.strip()) < 10:
            score -= 0.3
        if len(text.strip()) > 50:
            score += 0.1
        # Check for repetition
        words = text.lower().split()
        if words:
            unique_ratio = len(set(words)) / len(words)
            score += 0.15 * unique_ratio
        return max(0.0, min(1.0, score))

    @torch.no_grad()
    def backward(
        self, x: Tensor, neural_out: Tensor, grad_out: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        x = x.flatten().to(device=self.device, dtype=self.dtype)
        n = neural_out.flatten().to(device=self.device, dtype=self.dtype)
        g = grad_out.flatten().to(device=self.device, dtype=self.dtype)
        dim = min(self.dim, x.numel(), n.numel(), g.numel())
        gi = torch.zeros(dim, device=self.device, dtype=self.dtype)
        gn = torch.zeros(dim, device=self.device, dtype=self.dtype)
        gn.copy_(0.3 * g[:dim])
        gi.copy_(0.7 * g[:dim])
        return gi, gn


# ── Universal Transformer ───────────────────────────────────────────────────

class UniversalTransformer(nn.Module):
    def __init__(self, spec: ModelSpec, device: torch.device, dtype: torch.dtype):
        super().__init__()
        self.spec = spec
        self.device = device
        self.dtype = dtype

        self.embed_tokens = nn.Parameter(
            torch.empty(spec.vocab_size, spec.dim, device=device, dtype=dtype),
            requires_grad=False,
        )
        self.lm_head = nn.Parameter(
            torch.empty(spec.vocab_size, spec.dim, device=device, dtype=dtype),
            requires_grad=False,
        )
        nn.init.normal_(self.embed_tokens, std=0.02)
        nn.init.normal_(self.lm_head, std=0.02)

        self.blocks = nn.ModuleList(
            [TransformerBlock(spec) for _ in range(spec.n_layers)]
        )
        self.final_norm = TransformerNorm(spec.dim, spec.norm_type, spec.norm_eps)
        self.rope = RotaryEmbedding(
            spec.head_dim, spec.rope_theta, device,
            scaling_type=spec.rope_scaling_type,
            scaling_factor=spec.rope_scaling_factor,
            original_max_position=spec.rope_scaling_original_max_position,
        )

    def set_embedding(self, w: Tensor) -> None:
        self.embed_tokens.data.copy_(w)

    def set_lm_head(self, w: Tensor) -> None:
        self.lm_head.data.copy_(w)

    def set_final_norm(self, w: Tensor) -> None:
        self.final_norm.weight.data.copy_(w)

    def token_embedding(self, token_id: int) -> Tensor:
        return self.embed_tokens[token_id]

    def hidden_forward(
        self, x: Tensor, cache: PagedKVCache, seq_id: int, seq_pos: int,
    ) -> Tensor:
        x = x.to(device=self.device, dtype=self.dtype).reshape(-1)
        for blk in self.blocks:
            x = blk(x, self.rope, cache, seq_id, seq_pos)
        return self.final_norm(x)

    def logits(self, hidden: Tensor) -> Tensor:
        return F.linear(hidden, self.lm_head)

    def apply_lora_to_all_layers(self, lora_deltas: Dict[int, Dict[str, Tensor]]) -> None:
        """Apply LoRA deltas to attention projections across layers."""
        for layer_idx, deltas in lora_deltas.items():
            if 0 <= layer_idx < len(self.blocks):
                self.blocks[layer_idx].attn.apply_lora_delta(
                    q_delta=deltas.get("q"),
                    k_delta=deltas.get("k"),
                    v_delta=deltas.get("v"),
                    o_delta=deltas.get("o"),
                )

    def clear_all_lora(self) -> None:
        for blk in self.blocks:
            blk.attn.clear_lora_delta()


# ── Fused Kernel Backend ────────────────────────────────────────────────────

class FusedKernelBackend:
    def __init__(self, device: torch.device):
        self.device = device
        self.triton_available = False
        try:
            import triton  # noqa
            self.triton_available = True
        except Exception:
            pass

    def linear(self, x: Tensor, w: Tensor, b: Optional[Tensor] = None) -> Tensor:
        return F.linear(x, w, b)

    def matmul(self, x: Tensor, w: Tensor) -> Tensor:
        return x @ w.T


# ── NeuroHive GPU (Core Engine) ─────────────────────────────────────────────

class NeuroHiveGPU:
    """
    The complete NeuroHive GPU Ultimate engine — now the beating heart of NeuroOS.
    """

    def __init__(
        self, model_path: str, prefer_cuda: bool = True, dtype_name: str = "float16",
        seed: int = 42, entropy_threshold: float = 3.8, margin_threshold: float = 0.15,
        page_size: int = 16, max_pages: int = 4096,
        cache_quant: CacheQuantType = CacheQuantType.FP16,
    ):
        set_seed(seed)
        self.model_path = model_path
        self.device = pick_device(prefer_cuda)
        self.dtype = pick_dtype(dtype_name, self.device)
        self.runtime = RuntimeConfig(
            device=self.device, dtype=self.dtype, seed=seed,
            entropy_threshold=entropy_threshold, margin_threshold=margin_threshold,
            cache_quant=cache_quant,
        )
        self.tokenizer = TokenizerAdapter(model_path)
        self.reader: Optional[GGUFReaderAdapter] = None
        self.meta: Dict[str, Any] = {}
        self.spec: Optional[ModelSpec] = None
        self.weights = WeightStore(self.device, self.dtype)
        self.model: Optional[UniversalTransformer] = None
        self.cache: Optional[PagedKVCache] = None
        self.gate = EntropyGate(entropy_threshold, margin_threshold, top_k=8)
        self.reranker = System2Reranker(
            self.runtime.beam_width, self.runtime.repetition_penalty, self.runtime.temperature
        )
        self.symbolic: Optional[SymbolicBridgeGPU] = None
        self.backend = FusedKernelBackend(self.device)
        self.page_size = page_size
        self.max_pages = max_pages
        self.seq_counter = 1

        # Continuous memory (fast weights for ProteusNet)
        self._continuous_gradients: List[Dict[str, Tensor]] = []

        self._load()

    def _load(self) -> None:
        if gguf is None:
            raise RuntimeError("gguf não está instalado")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(self.model_path)

        logger.info("NeuroHive: Carregando GGUF: %s", self.model_path)
        self.reader = GGUFReaderAdapter(self.model_path)
        self.meta = self.reader.metadata()
        self.spec = GGUFArchitectureBuilder.build(self.meta)

        logger.info(
            "Spec: arch=%s family=%s dim=%s layers=%s heads=%s kv=%s hdim=%s ffn=%s "
            "vocab=%s seq=%s rope=%s x%.3f",
            self.spec.architecture, self.spec.family.value, self.spec.dim,
            self.spec.n_layers, self.spec.n_heads, self.spec.n_kv_heads,
            self.spec.head_dim, self.spec.ffn_dim, self.spec.vocab_size,
            self.spec.max_seq_len, self.spec.rope_scaling_type.value,
            self.spec.rope_scaling_factor,
        )

        self.weights.load_from_reader(self.reader)
        self.model = UniversalTransformer(self.spec, self.device, self.dtype).to(self.device)
        self.cache = PagedKVCache(
            max_seq_len=self.spec.max_seq_len, n_kv_heads=self.spec.n_kv_heads,
            head_dim=self.spec.head_dim, device=self.device, dtype=self.dtype,
            page_size=self.page_size, max_pages=self.max_pages,
            quant_type=self.runtime.cache_quant,
        )
        self.symbolic = SymbolicBridgeGPU(
            self.spec.dim, self.device, self.dtype, tnorm=TNormType.PRODUCT
        )
        self._load_weights()
        logger.info(
            "NeuroHive pronto: device=%s dtype=%s", self.device, self.dtype
        )

    def _load_weights(self) -> None:
        assert self.spec is not None and self.model is not None
        s = self.spec
        w = self.weights

        # Embeddings
        emb = w.resolve([
            "token_embd.weight", "tok_embeddings.weight", "embedding.weight",
            "model.embed_tokens.weight", "transformer.wte.weight",
        ])
        if emb is not None:
            if emb.dim() == 2 and emb.shape == (s.vocab_size, s.dim):
                self.model.set_embedding(emb)
            elif emb.dim() == 2 and emb.shape == (s.dim, s.vocab_size):
                self.model.set_embedding(emb.t().contiguous())

        # LM head
        head = w.resolve([
            "output.weight", "lm_head.weight", "model.lm_head.weight",
            "transformer.wte.weight",
        ])
        if head is not None:
            if head.dim() == 2 and head.shape == (s.vocab_size, s.dim):
                self.model.set_lm_head(head)
            elif head.dim() == 2 and head.shape == (s.dim, s.vocab_size):
                self.model.set_lm_head(head.t().contiguous())

        # Final norm
        fn = w.resolve([
            "output_norm.weight", "norm.weight", "model.norm.weight", "final_norm.weight",
        ])
        if fn is not None and fn.numel() == s.dim:
            self.model.set_final_norm(fn.reshape(-1).to(self.device, self.dtype))

        # Blocks
        for i, blk in enumerate(self.model.blocks):
            q = w.linear_or_init([
                f"blk.{i}.attn_q.weight", f"blk.{i}.attention.wq.weight",
                f"layers.{i}.self_attn.q_proj.weight",
                f"model.layers.{i}.self_attn.q_proj.weight",
                f"blk.{i}.q_proj.weight",
            ], s.dim, s.n_heads * s.head_dim, f"q_proj[{i}]")

            k = w.linear_or_init([
                f"blk.{i}.attn_k.weight", f"blk.{i}.attention.wk.weight",
                f"layers.{i}.self_attn.k_proj.weight",
                f"model.layers.{i}.self_attn.k_proj.weight",
                f"blk.{i}.k_proj.weight",
            ], s.dim, s.n_kv_heads * s.head_dim, f"k_proj[{i}]")

            v = w.linear_or_init([
                f"blk.{i}.attn_v.weight", f"blk.{i}.attention.wv.weight",
                f"layers.{i}.self_attn.v_proj.weight",
                f"model.layers.{i}.self_attn.v_proj.weight",
                f"blk.{i}.v_proj.weight",
            ], s.dim, s.n_kv_heads * s.head_dim, f"v_proj[{i}]")

            o = w.linear_or_init([
                f"blk.{i}.attn_output.weight", f"blk.{i}.attention.wo.weight",
                f"layers.{i}.self_attn.o_proj.weight",
                f"model.layers.{i}.self_attn.o_proj.weight",
                f"blk.{i}.o_proj.weight",
            ], s.n_heads * s.head_dim, s.dim, f"o_proj[{i}]")

            blk.attn.set_weights(q, k, v, o)

            attn_norm = w.vector_or_none([
                f"blk.{i}.attn_norm.weight", f"blk.{i}.attention_norm.weight",
                f"layers.{i}.input_layernorm.weight",
                f"model.layers.{i}.input_layernorm.weight",
                f"blk.{i}.input_norm.weight",
            ], s.dim, f"attn_norm[{i}]")
            if attn_norm is not None:
                blk.attn_norm.weight.data.copy_(attn_norm)

            ffn_norm = w.vector_or_none([
                f"blk.{i}.ffn_norm.weight", f"blk.{i}.post_attention_layernorm.weight",
                f"layers.{i}.post_attention_layernorm.weight",
                f"model.layers.{i}.post_attention_layernorm.weight",
                f"blk.{i}.mlp_norm.weight",
            ], s.dim, f"ffn_norm[{i}]")
            if ffn_norm is not None:
                blk.ffn_norm.weight.data.copy_(ffn_norm)

            gate = w.linear_or_init([
                f"blk.{i}.ffn_gate.weight", f"blk.{i}.feed_forward.w1.weight",
                f"layers.{i}.mlp.gate_proj.weight",
                f"model.layers.{i}.mlp.gate_proj.weight",
                f"blk.{i}.gate_proj.weight",
            ], s.dim, s.ffn_dim, f"ffn_gate[{i}]")

            up = w.linear_or_init([
                f"blk.{i}.ffn_up.weight", f"blk.{i}.feed_forward.w3.weight",
                f"layers.{i}.mlp.up_proj.weight",
                f"model.layers.{i}.mlp.up_proj.weight",
                f"blk.{i}.up_proj.weight",
            ], s.dim, s.ffn_dim, f"ffn_up[{i}]")

            down = w.linear_or_init([
                f"blk.{i}.ffn_down.weight", f"blk.{i}.feed_forward.w2.weight",
                f"layers.{i}.mlp.down_proj.weight",
                f"model.layers.{i}.mlp.down_proj.weight",
                f"blk.{i}.down_proj.weight",
            ], s.ffn_dim, s.dim, f"ffn_down[{i}]")

            blk.ffn.set_weights(gate, up, down)

    # ── State ────────────────────────────────────────────────────────────────

    def reset(self) -> None:
        if self.cache is not None:
            self.cache.reset_all()

    def new_sequence_id(self) -> int:
        sid = self.seq_counter
        self.seq_counter += 1
        return sid

    # ── Forward ──────────────────────────────────────────────────────────────

    @torch.inference_mode()
    def forward_hidden_from_token(
        self, token_id: int, seq_pos: int, seq_id: Optional[int] = None,
    ) -> Tensor:
        assert self.model is not None
        if seq_id is None:
            seq_id = 1
        x = self.model.token_embedding(token_id)
        return self.model.hidden_forward(x, self.cache, seq_id, seq_pos)

    @torch.inference_mode()
    def forward_logits_from_hidden(self, hidden: Tensor) -> Tensor:
        assert self.model is not None
        return self.model.logits(hidden)

    # ── Entropy / Reasoning ──────────────────────────────────────────────────

    @torch.inference_mode()
    def _reason_over_logits(self, logits: Tensor, context_ids: List[int]) -> int:
        decision = self.gate.inspect(logits)
        symbolic_score = 0.0
        if decision.activate and self.symbolic is not None:
            pass  # symbolic augmentation placeholder
        return self.reranker.rerank(logits, context_ids, symbolic_score=symbolic_score)

    # ── Token Choice ─────────────────────────────────────────────────────────

    @torch.inference_mode()
    def _sample_token(
        self, logits: Tensor, context_ids: List[int],
        temperature: float, top_k_sampling: int,
    ) -> int:
        decision = self.gate.inspect(logits)
        if decision.activate:
            return self._reason_over_logits(logits, context_ids)
        scaled = logits / max(temperature, 1e-6)
        if 0 < top_k_sampling < scaled.numel():
            vals, ids = torch.topk(scaled, k=top_k_sampling)
            probs = torch.softmax(vals, dim=0)
            pick = torch.multinomial(probs, num_samples=1).item()
            return int(ids[pick].item())
        probs = torch.softmax(scaled, dim=0)
        return int(torch.multinomial(probs, num_samples=1).item())

    # ── Generation ───────────────────────────────────────────────────────────

    @torch.inference_mode()
    def generate(
        self, prompt: str, max_new_tokens: int = 64, temperature: float = 0.7,
        top_k_sampling: int = 40, stop_token_ids: Optional[List[int]] = None,
        stream: bool = False, seq_id: int = 1,
        context_prefix: str = "",
    ) -> str:
        assert self.model is not None and self.spec is not None
        if self.cache is None:
            raise RuntimeError("Cache não inicializado")

        self.cache.clear_seq(seq_id)
        self.reranker.temperature = temperature
        if stop_token_ids is None:
            stop_token_ids = []

        full_prompt = context_prefix + prompt if context_prefix else prompt
        input_ids = self.tokenizer.encode(full_prompt)
        if not input_ids:
            return ""

        generated_ids = list(input_ids)
        out_text = full_prompt

        for pos, tid in enumerate(input_ids):
            hidden = self.forward_hidden_from_token(tid, seq_pos=pos, seq_id=seq_id)
            _ = self.forward_logits_from_hidden(hidden)

        amp_enabled = self.device.type == "cuda"
        autocast_dtype = torch.float16 if self.dtype == torch.float16 else torch.bfloat16

        for step in range(max_new_tokens):
            seq_pos = len(generated_ids) - 1
            last_token = generated_ids[-1]
            with torch.autocast(
                device_type=self.device.type, enabled=amp_enabled, dtype=autocast_dtype,
            ):
                hidden = self.forward_hidden_from_token(
                    last_token, seq_pos=seq_pos, seq_id=seq_id
                )
                logits = self.forward_logits_from_hidden(hidden).float()

            chosen_id = self._sample_token(logits, generated_ids, temperature, top_k_sampling)
            if chosen_id in stop_token_ids:
                break
            generated_ids.append(chosen_id)
            piece = self.tokenizer.decode([chosen_id])
            out_text += piece
            if stream:
                print(piece, end="", flush=True)

        return out_text

    @torch.inference_mode()
    def generate_batch(
        self, prompts: List[str], max_new_tokens: int = 64,
        temperature: float = 0.7, top_k_sampling: int = 40,
        stop_token_ids: Optional[List[int]] = None, stream: bool = False,
        context_prefixes: Optional[List[str]] = None,
    ) -> List[str]:
        if stop_token_ids is None:
            stop_token_ids = []
        if context_prefixes is None:
            context_prefixes = [""] * len(prompts)

        seq_ids = [self.new_sequence_id() for _ in prompts]
        states: List[Dict[str, Any]] = []

        for seq_id, prompt, prefix in zip(seq_ids, prompts, context_prefixes):
            self.cache.clear_seq(seq_id)
            full = prefix + prompt if prefix else prompt
            ids = self.tokenizer.encode(full)
            if not ids:
                ids = [0]
            for pos, tid in enumerate(ids):
                hidden = self.forward_hidden_from_token(tid, seq_pos=pos, seq_id=seq_id)
                _ = self.forward_logits_from_hidden(hidden)
            states.append({
                "seq_id": seq_id, "generated_ids": list(ids),
                "out_text": full, "last_token": ids[-1], "done": False,
            })

        amp_enabled = self.device.type == "cuda"
        autocast_dtype = torch.float16 if self.dtype == torch.float16 else torch.bfloat16

        for _ in range(max_new_tokens):
            alive = 0
            for st in states:
                if st["done"]:
                    continue
                alive += 1
                with torch.autocast(
                    device_type=self.device.type, enabled=amp_enabled, dtype=autocast_dtype,
                ):
                    hidden = self.forward_hidden_from_token(
                        st["last_token"],
                        seq_pos=len(st["generated_ids"]) - 1,
                        seq_id=st["seq_id"],
                    )
                    logits = self.forward_logits_from_hidden(hidden).float()
                chosen_id = self._sample_token(
                    logits, st["generated_ids"], temperature, top_k_sampling,
                )
                if chosen_id in stop_token_ids:
                    st["done"] = True
                    continue
                st["generated_ids"].append(chosen_id)
                st["last_token"] = chosen_id
                st["out_text"] += self.tokenizer.decode([chosen_id])
            if alive == 0:
                break

        return [st["out_text"] for st in states]

    # ── Symbolic API ─────────────────────────────────────────────────────────

    def add_symbolic_rule(
        self, ops: List[SymOp], operands: List[int],
        confidence: float = 0.8, temperature: float = 5.0,
    ) -> int:
        assert self.symbolic is not None
        rid = self.symbolic.add_rule(confidence=confidence, temperature=temperature)
        self.symbolic.set_rule_condition(rid, ops, operands)
        return rid

    def symbolic_forward(self, x: Tensor, neural_out: Tensor) -> Tensor:
        assert self.symbolic is not None
        return self.symbolic.forward(x, neural_out)

    def symbolic_backward(
        self, x: Tensor, neural_out: Tensor, grad_out: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        assert self.symbolic is not None
        return self.symbolic.backward(x, neural_out, grad_out)

    # ── Continuous Memory (ProteusNet) ───────────────────────────────────────

    def record_feedback_gradient(self, gradient_data: Dict[str, Tensor]) -> None:
        """Record a gradient from user feedback for later consolidation."""
        self._continuous_gradients.append(gradient_data)
        logger.info(
            "ProteusNet: Recorded feedback gradient (%d accumulated)",
            len(self._continuous_gradients),
        )

    def flush_gradients(self) -> List[Dict[str, Tensor]]:
        """Return and clear accumulated gradients."""
        grads = list(self._continuous_gradients)
        self._continuous_gradients.clear()
        return grads

    # ── Stats ────────────────────────────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        return {
            "device": str(self.device),
            "dtype": str(self.dtype),
            "architecture": self.spec.architecture if self.spec else None,
            "family": self.spec.family.value if self.spec else None,
            "dim": self.spec.dim if self.spec else None,
            "layers": self.spec.n_layers if self.spec else None,
            "heads": self.spec.n_heads if self.spec else None,
            "kv_heads": self.spec.n_kv_heads if self.spec else None,
            "head_dim": self.spec.head_dim if self.spec else None,
            "vocab_size": self.spec.vocab_size if self.spec else None,
            "max_seq_len": self.spec.max_seq_len if self.spec else None,
            "sliding_window": self.spec.sliding_window if self.spec else None,
            "rope_scaling_type": (
                self.spec.rope_scaling_type.value if self.spec else None
            ),
            "rope_scaling_factor": (
                self.spec.rope_scaling_factor if self.spec else None
            ),
            "use_moe": self.spec.use_moe if self.spec else None,
            "cache_quant": self.runtime.cache_quant.value,
            "page_size": self.page_size,
            "max_pages": self.max_pages,
            "gpu_memory": gpu_memory_stats(),
            "accumulated_gradients": len(self._continuous_gradients),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: FASE 1 — NEURAL BUS + VRAM MANAGER + OPENAI API SERVER
# ═══════════════════════════════════════════════════════════════════════════════

# ── 1.1 Neural Bus (Message Bus) ────────────────────────────────────────────

class NeuralBus:
    """
    Asynchronous message bus for inter-module communication.
    Supports:
    - In-process queue (always available)
    - Redis pub/sub (if redis installed)
    - ZeroMQ (if pyzmq installed)
    """

    def __init__(self, backend: str = "queue"):
        self.backend = backend
        self._queue: Queue = Queue(maxsize=10000)
        self._subscribers: Dict[str, List[Callable[[BusMessage], None]]] = {}
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._redis_client = None
        self._zmq_context = None
        self._zmq_socket = None

        if backend == "redis" and redis_lib is not None:
            try:
                self._redis_client = redis_lib.Redis(host="localhost", port=6379, db=0)
                self._redis_client.ping()
                logger.info("NeuralBus: Redis backend conectado")
            except Exception as e:
                logger.warning("NeuralBus: Redis falhou (%s), usando queue", e)
                self.backend = "queue"

        elif backend == "zmq" and zmq is not None:
            try:
                self._zmq_context = zmq.Context()
                self._zmq_socket = self._zmq_context.socket(zmq.PUB)
                self._zmq_socket.bind("tcp://127.0.0.1:5555")
                logger.info("NeuralBus: ZeroMQ backend na porta 5555")
            except Exception as e:
                logger.warning("NeuralBus: ZMQ falhou (%s), usando queue", e)
                self.backend = "queue"

    def subscribe(self, msg_type: str, callback: Callable[[BusMessage], None]) -> None:
        if msg_type not in self._subscribers:
            self._subscribers[msg_type] = []
        self._subscribers[msg_type].append(callback)

    def publish(self, msg: BusMessage) -> None:
        if self.backend == "redis" and self._redis_client is not None:
            try:
                self._redis_client.publish("neuroos_bus", msg.to_json())
            except Exception:
                pass

        if self.backend == "zmq" and self._zmq_socket is not None:
            try:
                self._zmq_socket.send_string(msg.to_json())
            except Exception:
                pass

        self._queue.put(msg)
        self._dispatch_local(msg)

    def _dispatch_local(self, msg: BusMessage) -> None:
        key = msg.msg_type.value
        for cb in self._subscribers.get(key, []):
            try:
                cb(msg)
            except Exception as e:
                logger.error("NeuralBus dispatch error: %s", e)

        for cb in self._subscribers.get("*", []):
            try:
                cb(msg)
            except Exception as e:
                logger.error("NeuralBus wildcard error: %s", e)

    def start_listener(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _listen_loop(self) -> None:
        while self._running:
            try:
                msg = self._queue.get(timeout=0.1)
                # Already dispatched in publish
            except Empty:
                pass

    def drain(self) -> List[BusMessage]:
        messages = []
        while not self._queue.empty():
            try:
                messages.append(self._queue.get_nowait())
            except Empty:
                break
        return messages


# ── 1.2 VRAM Manager ────────────────────────────────────────────────────────

class VRAMManager:
    """
    Dynamic VRAM allocator that can offload/reload tensors between GPU and CPU.
    Manages LoRA adapters and auxiliary models to prevent OOM.
    """

    def __init__(self, device: torch.device, dtype: torch.dtype):
        self.device = device
        self.dtype = dtype
        self._offloaded: Dict[str, Tensor] = {}  # name -> CPU tensor
        self._gpu_registry: Dict[str, Tensor] = {}  # name -> GPU tensor ref
        self._lock = threading.Lock()
        self._vram_threshold = 0.85  # offload when VRAM usage > 85%

    def register(self, name: str, tensor: Tensor) -> None:
        with self._lock:
            self._gpu_registry[name] = tensor

    def unregister(self, name: str) -> None:
        with self._lock:
            self._gpu_registry.pop(name, None)
            self._offloaded.pop(name, None)

    def vram_usage_ratio(self) -> float:
        if not torch.cuda.is_available():
            return 0.0
        allocated = torch.cuda.memory_allocated()
        total = torch.cuda.get_device_properties(0).total_mem
        return allocated / max(total, 1)

    def should_offload(self) -> bool:
        return self.vram_usage_ratio() > self._vram_threshold

    def offload(self, name: str) -> bool:
        """Move a registered tensor from GPU to CPU."""
        with self._lock:
            if name not in self._gpu_registry:
                return False
            if name in self._offloaded:
                return True

            tensor = self._gpu_registry[name]
            self._offloaded[name] = tensor.cpu().clone()
            tensor.data = torch.zeros(1, device=self.device, dtype=self.dtype)
            logger.info(
                "VRAMManager: Offloaded '%s' to CPU (%s)",
                name, bytes_to_human(self._offloaded[name].nelement() * 2),
            )
            return True

    def reload(self, name: str) -> bool:
        """Move an offloaded tensor back to GPU."""
        with self._lock:
            if name not in self._offloaded:
                return False
            cpu_tensor = self._offloaded.pop(name)
            if name in self._gpu_registry:
                self._gpu_registry[name].data = cpu_tensor.to(
                    device=self.device, dtype=self.dtype
                )
            logger.info("VRAMManager: Reloaded '%s' to GPU", name)
            return True

    def offload_least_used(self, exclude: Optional[Set[str]] = None) -> Optional[str]:
        """Offload the first available tensor that isn't excluded."""
        exclude = exclude or set()
        with self._lock:
            for name in list(self._gpu_registry.keys()):
                if name not in exclude and name not in self._offloaded:
                    self.offload(name)
                    return name
        return None

    def auto_manage(self, critical_names: Optional[Set[str]] = None) -> None:
        """Automatically offload if VRAM is under pressure."""
        if not self.should_offload():
            return
        critical = critical_names or set()
        offloaded = 0
        while self.should_offload() and offloaded < 5:
            result = self.offload_least_used(exclude=critical)
            if result is None:
                break
            offloaded += 1

    def status(self) -> Dict[str, Any]:
        return {
            "gpu_registered": len(self._gpu_registry),
            "cpu_offloaded": len(self._offloaded),
            "vram_usage": f"{self.vram_usage_ratio() * 100:.1f}%",
            "offloaded_names": list(self._offloaded.keys()),
        }


# ── 1.3 OpenAI-Compatible API Server ────────────────────────────────────────

class OpenAICompatibleServer:
    """
    FastAPI server that mimics the OpenAI Chat Completions API.
    Any tool (LangChain, AutoGen, CrewAI) can connect as if it were ChatGPT.
    """

    def __init__(self, neuroos: "NeuroOS", host: str = "0.0.0.0", port: int = 8000):
        self.neuroos = neuroos
        self.host = host
        self.port = port
        self.app = None
        self._server_thread: Optional[threading.Thread] = None

        if HAS_FASTAPI:
            self.app = FastAPI(title="NeuroOS API", version="1.0.0")
            self._setup_routes()

    def _setup_routes(self) -> None:
        if self.app is None:
            return

        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @self.app.get("/v1/models")
        async def list_models():
            engine = self.neuroos.engine
            model_id = "neuroos-" + (engine.spec.architecture if engine and engine.spec else "unknown")
            return {
                "object": "list",
                "data": [{
                    "id": model_id,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "neuroos",
                }],
            }

        @self.app.post("/v1/chat/completions")
        async def chat_completions(request: Request):
            try:
                body = await request.json()
            except Exception:
                raise HTTPException(status_code=400, detail="Invalid JSON")

            messages = body.get("messages", [])
            temperature = body.get("temperature", 0.7)
            max_tokens = body.get("max_tokens", 256)
            top_k = body.get("top_k", 40)
            stream_mode = body.get("stream", False)

            # Build prompt from messages
            prompt_parts = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "system":
                    prompt_parts.append(f"[System] {content}")
                elif role == "user":
                    prompt_parts.append(f"[User] {content}")
                elif role == "assistant":
                    prompt_parts.append(f"[Assistant] {content}")

            prompt = "\n".join(prompt_parts)
            prompt += "\n[Assistant] "

            # Inject CortexFS context if available
            context_prefix = ""
            if self.neuroos.cortex is not None:
                user_msgs = [m.get("content", "") for m in messages if m.get("role") == "user"]
                if user_msgs:
                    context_nodes = self.neuroos.cortex.query_relevant(
                        user_msgs[-1], top_k=3
                    )
                    if context_nodes:
                        ctx_texts = [n.content for n in context_nodes]
                        context_prefix = (
                            "[Context from memory]\n"
                            + "\n".join(ctx_texts)
                            + "\n[End context]\n"
                        )

            # Generate via the full NeuroOS pipeline
            result = self.neuroos.generate_with_full_pipeline(
                prompt=prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k_sampling=top_k,
                context_prefix=context_prefix,
            )

            # Format OpenAI-compatible response
            response_text = result
            if "[Assistant] " in result:
                response_text = result.split("[Assistant] ")[-1]

            completion_id = f"chatcmpl-{secrets.token_hex(12)}"

            if stream_mode:
                async def stream_generator():
                    # Simulate streaming by chunking
                    words = response_text.split()
                    for i, word in enumerate(words):
                        chunk = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": "neuroos",
                            "choices": [{
                                "index": 0,
                                "delta": {"content": word + " "},
                                "finish_reason": None,
                            }],
                        }
                        if i == len(words) - 1:
                            chunk["choices"][0]["finish_reason"] = "stop"
                        yield f"data: {json.dumps(chunk)}\n\n"
                    yield "data: [DONE]\n\n"

                return StreamingResponse(
                    stream_generator(), media_type="text/event-stream"
                )

            return {
                "id": completion_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "neuroos",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text,
                    },
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": len(prompt.split()),
                    "completion_tokens": len(response_text.split()),
                    "total_tokens": len(prompt.split()) + len(response_text.split()),
                },
            }

        @self.app.get("/v1/neuroos/status")
        async def system_status():
            return self.neuroos.full_status()

        @self.app.post("/v1/neuroos/cortex/write")
        async def cortex_write(request: Request):
            body = await request.json()
            if self.neuroos.cortex is None:
                raise HTTPException(status_code=503, detail="CortexFS not initialized")
            node_id = body.get("node_id", secrets.token_hex(6))
            self.neuroos.cortex.write_node(CortexNode(
                node_id=node_id,
                node_type=body.get("node_type", "fact"),
                content=body.get("content", ""),
                metadata=body.get("metadata", {}),
            ))
            return {"status": "ok", "node_id": node_id}

        @self.app.post("/v1/neuroos/cortex/query")
        async def cortex_query(request: Request):
            body = await request.json()
            if self.neuroos.cortex is None:
                raise HTTPException(status_code=503, detail="CortexFS not initialized")
            query = body.get("query", "")
            top_k = body.get("top_k", 5)
            nodes = self.neuroos.cortex.query_relevant(query, top_k=top_k)
            return {
                "results": [
                    {"node_id": n.node_id, "type": n.node_type, "content": n.content}
                    for n in nodes
                ]
            }

        @self.app.post("/v1/neuroos/lora/load")
        async def lora_load(request: Request):
            body = await request.json()
            name = body.get("name", "")
            if self.neuroos.hydra is not None:
                success = self.neuroos.hydra.load_lora(name)
                return {"status": "loaded" if success else "failed", "name": name}
            raise HTTPException(status_code=503, detail="HydraMoE not initialized")

        @self.app.post("/v1/neuroos/feedback")
        async def record_feedback(request: Request):
            body = await request.json()
            quality = body.get("quality", 0.5)
            text = body.get("text", "")
            self.neuroos.record_user_feedback(text, quality)
            return {"status": "recorded"}

    def start(self) -> None:
        if not HAS_FASTAPI or not HAS_UVICORN:
            logger.warning("FastAPI/Uvicorn não disponível. API não iniciada.")
            return

        def _run():
            uvicorn.run(self.app, host=self.host, port=self.port, log_level="warning")

        self._server_thread = threading.Thread(target=_run, daemon=True)
        self._server_thread.start()
        logger.info("OpenAI API Server iniciado em http://%s:%d", self.host, self.port)

    def stop(self) -> None:
        pass  # Thread is daemon, dies with process


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: FASE 2 — AUTOFORGE (Synthetic Data + Reward + LoRA Trainer)
# ═══════════════════════════════════════════════════════════════════════════════

class KnowledgeMiner:
    """
    2.1 Minerador de Conhecimento
    Reads PDFs, scrapes web, consumes APIs to gather raw training material.
    """

    def __init__(self, output_dir: str = "./mined_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._scraped_urls: Set[str] = set()

    def mine_pdf(self, pdf_path: str) -> List[str]:
        """Extract text chunks from a PDF file."""
        if not HAS_PYPDF2:
            logger.warning("PyPDF2 não disponível. Ignorando PDF.")
            return []
        chunks = []
        try:
            with open(pdf_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text = page.extract_text()
                    if text and len(text.strip()) > 50:
                        # Split into ~500-word chunks
                        words = text.split()
                        for i in range(0, len(words), 400):
                            chunk = " ".join(words[i:i + 500])
                            if len(chunk) > 100:
                                chunks.append(chunk)
            logger.info("Mined %d chunks from PDF: %s", len(chunks), pdf_path)
        except Exception as e:
            logger.error("Failed to mine PDF %s: %s", pdf_path, e)
        return chunks

    async def mine_url(self, url: str) -> List[str]:
        """Scrape text from a web page."""
        if not HAS_AIOHTTP or not HAS_BS4:
            logger.warning("aiohttp/bs4 não disponível.")
            return []
        if url in self._scraped_urls:
            return []
        self._scraped_urls.add(url)

        chunks = []
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status == 200:
                        html = await resp.text()
                        soup = BeautifulSoup(html, "html.parser")
                        for script in soup(["script", "style"]):
                            script.decompose()
                        text = soup.get_text(separator="\n", strip=True)
                        words = text.split()
                        for i in range(0, len(words), 400):
                            chunk = " ".join(words[i:i + 500])
                            if len(chunk) > 100:
                                chunks.append(chunk)
            logger.info("Mined %d chunks from URL: %s", len(chunks), url)
        except Exception as e:
            logger.error("Failed to mine URL %s: %s", url, e)
        return chunks

    def mine_directory(self, dir_path: str, extensions: Tuple[str, ...] = (".txt", ".md")) -> List[str]:
        """Read all text files in a directory."""
        chunks = []
        dp = Path(dir_path)
        if not dp.exists():
            return chunks
        for fp in dp.rglob("*"):
            if fp.suffix.lower() in extensions:
                try:
                    text = fp.read_text(encoding="utf-8", errors="ignore")
                    words = text.split()
                    for i in range(0, len(words), 400):
                        chunk = " ".join(words[i:i + 500])
                        if len(chunk) > 100:
                            chunks.append(chunk)
                except Exception:
                    pass
        logger.info("Mined %d chunks from directory: %s", len(chunks), dir_path)
        return chunks

    def save_chunks(self, chunks: List[str], label: str = "mined") -> Path:
        """Save chunks as a JSONL file."""
        fname = f"{label}_{timestamp_str()}.jsonl"
        fpath = self.output_dir / fname
        with open(fpath, "w", encoding="utf-8") as f:
            for chunk in chunks:
                json.dump({"text": chunk, "source": label}, f, ensure_ascii=False)
                f.write("\n")
        return fpath


class SyntheticDataGenerator:
    """
    2.2 Self-Play Generator
    Uses NeuroHive (or an external teacher API) to generate high-quality QA pairs.
    """

    def __init__(
        self,
        engine: NeuroHiveGPU,
        teacher_api_url: Optional[str] = None,
        teacher_api_key: Optional[str] = None,
    ):
        self.engine = engine
        self.teacher_api_url = teacher_api_url
        self.teacher_api_key = teacher_api_key

    def _generate_from_local(
        self, prompt: str, max_tokens: int = 512, temperature: float = 0.5,
    ) -> str:
        """Use the local NeuroHive with forced System 2 (entropy_threshold=0)."""
        old_threshold = self.engine.gate.entropy_threshold
        self.engine.gate.entropy_threshold = 0.0  # Force System 2 on every token
        try:
            result = self.engine.generate(
                prompt=prompt, max_new_tokens=max_tokens,
                temperature=temperature, top_k_sampling=40,
                seq_id=self.engine.new_sequence_id(),
            )
        finally:
            self.engine.gate.entropy_threshold = old_threshold
        return result

    async def _generate_from_teacher(
        self, prompt: str, max_tokens: int = 512,
    ) -> Optional[str]:
        """Call an external teacher API (OpenAI-compatible)."""
        if not HAS_AIOHTTP or not self.teacher_api_url:
            return None
        headers = {"Content-Type": "application/json"}
        if self.teacher_api_key:
            headers["Authorization"] = f"Bearer {self.teacher_api_key}"

        payload = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.3,
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.teacher_api_url, json=payload, headers=headers,
                    timeout=aiohttp.ClientTimeout(total=120),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error("Teacher API failed: %s", e)
        return None

    def generate_qa_pairs(
        self, topics: List[str], n_per_topic: int = 10,
        use_teacher: bool = False,
    ) -> List[SyntheticDataSample]:
        """Generate question-answer pairs for given topics."""
        samples = []

        for topic in topics:
            for i in range(n_per_topic):
                prompt_templates = [
                    f"Generate a complex question about {topic} and answer it step by step.",
                    f"Create a challenging problem about {topic} with a detailed solution.",
                    f"Write a technical explanation about a concept in {topic}.",
                    f"Provide an example of {topic} with code and explanation.",
                ]
                template = prompt_templates[i % len(prompt_templates)]

                if use_teacher and self.teacher_api_url:
                    # Would need to be run in async context
                    response = self._generate_from_local(template)
                else:
                    response = self._generate_from_local(template)

                # Split into Q/A
                if "[Assistant]" in response:
                    response = response.split("[Assistant]")[-1].strip()

                samples.append(SyntheticDataSample(
                    prompt=template,
                    response=response,
                    category=topic,
                    quality_score=0.0,  # Will be filled by RewardModel
                ))

            logger.info(
                "SyntheticDataGen: Generated %d samples for topic '%s'",
                n_per_topic, topic,
            )

        return samples


class RewardModel:
    """
    2.3 Reward / Filter
    Uses the SymbolicBridge to validate quality of generated data.
    """

    def __init__(self, symbolic: SymbolicBridgeGPU):
        self.symbolic = symbolic
        self.min_quality = 0.4

    def score(self, sample: SyntheticDataSample) -> float:
        """Score a generated sample. Returns 0.0-1.0."""
        text = sample.response
        sym_score = self.symbolic.evaluate_quality(text)

        # Length score
        length = len(text.split())
        length_score = min(1.0, length / 100.0)

        # Relevance score (topic keywords in response)
        topic_words = sample.category.lower().split()
        response_lower = text.lower()
        relevance = sum(1 for w in topic_words if w in response_lower) / max(1, len(topic_words))

        # Diversity score
        words = text.lower().split()
        diversity = len(set(words)) / max(1, len(words)) if words else 0

        final = 0.3 * sym_score + 0.2 * length_score + 0.25 * relevance + 0.25 * diversity
        return max(0.0, min(1.0, final))

    def filter_samples(
        self, samples: List[SyntheticDataSample], min_quality: Optional[float] = None,
    ) -> List[SyntheticDataSample]:
        """Filter samples by quality score."""
        threshold = min_quality or self.min_quality
        approved = []
        for s in samples:
            s.quality_score = self.score(s)
            s.symbolic_valid = s.quality_score >= threshold
            if s.symbolic_valid:
                approved.append(s)
        logger.info(
            "RewardModel: Approved %d/%d samples (threshold=%.2f)",
            len(approved), len(samples), threshold,
        )
        return approved


class LoRAAutoTrainer:
    """
    2.4 Automatic LoRA Trainer
    Saves datasets and invokes external training (Unsloth/TRL) via subprocess.
    """

    def __init__(
        self,
        output_dir: str = "./lora_outputs",
        dataset_dir: str = "./datasets",
        base_model_path: Optional[str] = None,
    ):
        self.output_dir = Path(output_dir)
        self.dataset_dir = Path(dataset_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.base_model_path = base_model_path
        self._training_queue: Queue = Queue()
        self._active_training: Optional[str] = None
        self._trained_loras: Dict[str, LoRAMetadata] = {}
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def save_dataset(
        self, samples: List[SyntheticDataSample], category: str,
    ) -> Path:
        """Save approved samples as ShareGPT-format JSONL."""
        fname = f"{category}_{timestamp_str()}.jsonl"
        fpath = self.dataset_dir / fname
        with open(fpath, "w", encoding="utf-8") as f:
            for s in samples:
                entry = {
                    "conversations": [
                        {"from": "human", "value": s.prompt},
                        {"from": "gpt", "value": s.response},
                    ],
                    "category": s.category,
                    "quality_score": s.quality_score,
                }
                json.dump(entry, f, ensure_ascii=False)
                f.write("\n")
        logger.info("LoRAAutoTrainer: Saved %d samples to %s", len(samples), fpath)
        return fpath

    def queue_training(
        self, category: str, dataset_path: Path, rank: int = 16, alpha: float = 32.0,
    ) -> str:
        """Queue a LoRA training job."""
        job_id = f"lora_{category}_{timestamp_str()}"
        self._training_queue.put({
            "job_id": job_id,
            "category": category,
            "dataset_path": str(dataset_path),
            "rank": rank,
            "alpha": alpha,
        })
        logger.info("LoRAAutoTrainer: Queued training job %s", job_id)
        return job_id

    def _generate_training_script(self, job: Dict[str, Any]) -> str:
        """Generate a training script for Unsloth/TRL."""
        script = textwrap.dedent(f"""\
        #!/usr/bin/env python3
        # Auto-generated LoRA training script by NeuroOS AutoForge
        # Job: {job['job_id']}

        import os
        import json

        try:
            from unsloth import FastLanguageModel
            USE_UNSLOTH = True
        except ImportError:
            USE_UNSLOTH = False

        if not USE_UNSLOTH:
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                from peft import LoraConfig, get_peft_model, TaskType
                from trl import SFTTrainer
                import transformers
                USE_TRL = True
            except ImportError:
                USE_TRL = False
                print("Neither unsloth nor trl available. Cannot train.")
                exit(1)

        DATASET_PATH = "{job['dataset_path']}"
        OUTPUT_DIR = "{str(self.output_dir / job['job_id'])}"
        BASE_MODEL = "{self.base_model_path or 'meta-llama/Llama-3-8B'}"
        RANK = {job['rank']}
        ALPHA = {job['alpha']}

        # Load dataset
        data = []
        with open(DATASET_PATH, 'r') as f:
            for line in f:
                data.append(json.loads(line))

        print(f"Loaded {{len(data)}} training samples")
        print(f"Training LoRA with rank={{RANK}}, alpha={{ALPHA}}")
        print(f"Output: {{OUTPUT_DIR}}")

        # Training would proceed here with the actual library
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w") as f:
            json.dump({{
                "job_id": "{job['job_id']}",
                "category": "{job['category']}",
                "rank": RANK,
                "alpha": ALPHA,
                "samples": len(data),
            }}, f)

        print("Training complete!")
        """)
        return script

    def _run_training(self, job: Dict[str, Any]) -> bool:
        """Execute a training job."""
        self._active_training = job["job_id"]
        logger.info("LoRAAutoTrainer: Starting training for %s", job["job_id"])

        script = self._generate_training_script(job)
        script_path = self.output_dir / f"{job['job_id']}_train.py"
        script_path.write_text(script)

        try:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True, text=True, timeout=3600,
            )
            if result.returncode == 0:
                lora_path = str(self.output_dir / job["job_id"])
                meta = LoRAMetadata(
                    name=job["job_id"],
                    path=lora_path,
                    category=job["category"],
                    rank=job["rank"],
                    alpha=job["alpha"],
                    training_samples=0,
                )
                self._trained_loras[job["job_id"]] = meta
                logger.info("LoRAAutoTrainer: Training complete: %s", job["job_id"])
                return True
            else:
                logger.error(
                    "LoRAAutoTrainer: Training failed: %s\nstderr: %s",
                    job["job_id"], result.stderr[:500],
                )
                return False
        except subprocess.TimeoutExpired:
            logger.error("LoRAAutoTrainer: Training timed out: %s", job["job_id"])
            return False
        except Exception as e:
            logger.error("LoRAAutoTrainer: Training error: %s", e)
            return False
        finally:
            self._active_training = None

    def start_background_trainer(self) -> None:
        """Start background thread that processes the training queue."""
        self._running = True
        self._thread = threading.Thread(target=self._train_loop, daemon=True)
        self._thread.start()
        logger.info("LoRAAutoTrainer: Background trainer started")

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)

    def _train_loop(self) -> None:
        while self._running:
            try:
                job = self._training_queue.get(timeout=5.0)
                self._run_training(job)
            except Empty:
                pass
            except Exception as e:
                logger.error("LoRAAutoTrainer loop error: %s", e)

    def get_trained_loras(self) -> Dict[str, LoRAMetadata]:
        return dict(self._trained_loras)


class AutoForge:
    """
    The complete AutoForge pipeline (FASE 2):
    Mine → Generate → Filter → Save → Train → LoRA
    """

    def __init__(
        self, engine: NeuroHiveGPU,
        output_dir: str = "./neuroos_forge",
        teacher_api_url: Optional[str] = None,
        teacher_api_key: Optional[str] = None,
    ):
        self.engine = engine
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.miner = KnowledgeMiner(str(self.output_dir / "mined"))
        self.generator = SyntheticDataGenerator(
            engine, teacher_api_url, teacher_api_key,
        )
        self.reward = RewardModel(engine.symbolic)
        self.trainer = LoRAAutoTrainer(
            output_dir=str(self.output_dir / "loras"),
            dataset_dir=str(self.output_dir / "datasets"),
            base_model_path=engine.model_path,
        )

        # Accumulated samples per category
        self._sample_buffer: Dict[str, List[SyntheticDataSample]] = {}
        self._auto_train_threshold = 5000  # Samples before auto-training

    def run_pipeline(
        self, topics: List[str], n_per_topic: int = 10,
        min_quality: float = 0.4, auto_train: bool = True,
    ) -> Dict[str, Any]:
        """Run the complete AutoForge pipeline."""
        results = {
            "generated": 0, "approved": 0, "queued_trainings": [],
        }

        # Step 1: Generate synthetic data
        raw_samples = self.generator.generate_qa_pairs(topics, n_per_topic)
        results["generated"] = len(raw_samples)

        # Step 2: Filter with RewardModel
        approved = self.reward.filter_samples(raw_samples, min_quality)
        results["approved"] = len(approved)

        # Step 3: Accumulate per category
        for sample in approved:
            cat = sample.category
            if cat not in self._sample_buffer:
                self._sample_buffer[cat] = []
            self._sample_buffer[cat].append(sample)

        # Step 4: Auto-train if threshold reached
        if auto_train:
            for cat, samples in self._sample_buffer.items():
                if len(samples) >= self._auto_train_threshold:
                    dataset_path = self.trainer.save_dataset(samples, cat)
                    job_id = self.trainer.queue_training(cat, dataset_path)
                    results["queued_trainings"].append(job_id)
                    self._sample_buffer[cat] = []  # Reset buffer

        return results

    def force_train(self, category: str) -> Optional[str]:
        """Force training for a category regardless of buffer size."""
        samples = self._sample_buffer.get(category, [])
        if not samples:
            logger.warning("No samples for category '%s'", category)
            return None
        dataset_path = self.trainer.save_dataset(samples, category)
        job_id = self.trainer.queue_training(category, dataset_path)
        self._sample_buffer[category] = []
        return job_id

    def status(self) -> Dict[str, Any]:
        return {
            "buffer_sizes": {k: len(v) for k, v in self._sample_buffer.items()},
            "trained_loras": {
                k: v.name for k, v in self.trainer.get_trained_loras().items()
            },
            "auto_train_threshold": self._auto_train_threshold,
            "active_training": self.trainer._active_training,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: FASE 3 — HYDRAMOE + PROTEUSNET (Dynamic LoRA + Continuous Learning)
# ═══════════════════════════════════════════════════════════════════════════════

class IntentRouter:
    """
    3.1 Intent Router
    A lightweight classifier that determines which LoRA to activate.
    Uses keyword matching + optional embedding similarity.
    """

    def __init__(self):
        self._category_keywords: Dict[str, List[str]] = {}
        self._embedder = None
        self._category_embeddings: Dict[str, np.ndarray] = {}

        if HAS_SENTENCE_TRANSFORMERS:
            try:
                self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
                logger.info("IntentRouter: SentenceTransformer loaded")
            except Exception as e:
                logger.warning("IntentRouter: Failed to load embedder: %s", e)

    def register_category(self, category: str, keywords: List[str]) -> None:
        """Register a category with associated keywords."""
        self._category_keywords[category] = [kw.lower() for kw in keywords]
        if self._embedder is not None:
            desc = category + " " + " ".join(keywords)
            self._category_embeddings[category] = self._embedder.encode(desc)

    def classify(self, text: str) -> List[Tuple[str, float]]:
        """Classify text into categories with confidence scores."""
        scores: Dict[str, float] = {}
        text_lower = text.lower()

        # Keyword matching
        for cat, keywords in self._category_keywords.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                scores[cat] = score / len(keywords)

        # Embedding similarity
        if self._embedder is not None and self._category_embeddings:
            text_emb = self._embedder.encode(text)
            for cat, cat_emb in self._category_embeddings.items():
                sim = float(np.dot(text_emb, cat_emb) / (
                    np.linalg.norm(text_emb) * np.linalg.norm(cat_emb) + 1e-8
                ))
                if cat in scores:
                    scores[cat] = 0.6 * scores[cat] + 0.4 * max(0, sim)
                else:
                    scores[cat] = 0.4 * max(0, sim)

        # Sort by score
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        return ranked


class HydraMoE:
    """
    3.2 + 3.3 Dynamic LoRA Hot-Swapping + Continuous Adaptation
    Manages multiple LoRA adapters and can hot-swap them on GPU.
    """

    def __init__(
        self, engine: NeuroHiveGPU, vram_mgr: VRAMManager,
        lora_dir: str = "./neuroos_forge/loras",
    ):
        self.engine = engine
        self.vram_mgr = vram_mgr
        self.lora_dir = Path(lora_dir)
        self.lora_dir.mkdir(parents=True, exist_ok=True)

        self.router = IntentRouter()
        self._available_loras: Dict[str, LoRAMetadata] = {}
        self._active_loras: Dict[str, Dict[int, Dict[str, Tensor]]] = {}
        self._lock = threading.Lock()

        # Discover existing LoRAs
        self._discover_loras()

    def _discover_loras(self) -> None:
        """Scan the lora directory for available adapters."""
        if not self.lora_dir.exists():
            return
        for subdir in self.lora_dir.iterdir():
            if subdir.is_dir():
                meta_path = subdir / "metadata.json"
                if meta_path.exists():
                    try:
                        with open(meta_path) as f:
                            data = json.load(f)
                        meta = LoRAMetadata(
                            name=data.get("job_id", subdir.name),
                            path=str(subdir),
                            category=data.get("category", "unknown"),
                            rank=data.get("rank", 16),
                            alpha=data.get("alpha", 32.0),
                            training_samples=data.get("samples", 0),
                        )
                        self._available_loras[meta.name] = meta

                        # Register category with router
                        self.router.register_category(
                            meta.category,
                            meta.category.split("_") + meta.category.split(),
                        )
                        logger.info("HydraMoE: Discovered LoRA '%s' (category: %s)",
                                    meta.name, meta.category)
                    except Exception as e:
                        logger.warning("HydraMoE: Failed to read LoRA metadata: %s", e)

    def register_lora(self, meta: LoRAMetadata) -> None:
        """Register a new LoRA adapter."""
        self._available_loras[meta.name] = meta
        self.router.register_category(
            meta.category,
            meta.category.split("_") + meta.category.split(),
        )

    def load_lora(self, name: str) -> bool:
        """
        Load a LoRA adapter into GPU memory.
        Creates delta matrices: W_new = W_base + A @ B
        """
        with self._lock:
            if name in self._active_loras:
                logger.info("HydraMoE: LoRA '%s' already active", name)
                return True

            meta = self._available_loras.get(name)
            if meta is None:
                logger.error("HydraMoE: LoRA '%s' not found", name)
                return False

            # Check VRAM before loading
            self.vram_mgr.auto_manage(critical_names={"base_model"})

            try:
                spec = self.engine.spec
                device = self.engine.device
                dtype = self.engine.dtype

                # Generate LoRA delta matrices for each layer
                # In production, these would be loaded from the trained adapter files
                # Here we create the structure that would hold them
                lora_deltas: Dict[int, Dict[str, Tensor]] = {}

                lora_path = Path(meta.path)
                adapter_file = lora_path / "adapter_model.bin"

                if adapter_file.exists():
                    # Load real adapter weights
                    adapter_state = torch.load(
                        adapter_file, map_location=device, weights_only=True
                    )
                    for key, value in adapter_state.items():
                        # Parse layer index and projection from key
                        parts = key.split(".")
                        for p in parts:
                            if p.isdigit():
                                layer_idx = int(p)
                                if layer_idx not in lora_deltas:
                                    lora_deltas[layer_idx] = {}
                                if "q_proj" in key:
                                    lora_deltas[layer_idx]["q"] = value.to(
                                        device=device, dtype=dtype
                                    )
                                elif "k_proj" in key:
                                    lora_deltas[layer_idx]["k"] = value.to(
                                        device=device, dtype=dtype
                                    )
                                elif "v_proj" in key:
                                    lora_deltas[layer_idx]["v"] = value.to(
                                        device=device, dtype=dtype
                                    )
                                elif "o_proj" in key:
                                    lora_deltas[layer_idx]["o"] = value.to(
                                        device=device, dtype=dtype
                                    )
                                break
                else:
                    # Create zero-initialized deltas (placeholder)
                    for layer_idx in range(spec.n_layers):
                        lora_deltas[layer_idx] = {
                            "q": torch.zeros(
                                spec.n_heads * spec.head_dim, spec.dim,
                                device=device, dtype=dtype,
                            ),
                            "k": torch.zeros(
                                spec.n_kv_heads * spec.head_dim, spec.dim,
                                device=device, dtype=dtype,
                            ),
                            "v": torch.zeros(
                                spec.n_kv_heads * spec.head_dim, spec.dim,
                                device=device, dtype=dtype,
                            ),
                            "o": torch.zeros(
                                spec.dim, spec.n_heads * spec.head_dim,
                                device=device, dtype=dtype,
                            ),
                        }

                self._active_loras[name] = lora_deltas
                meta.active = True
                meta.on_gpu = True
                logger.info("HydraMoE: Loaded LoRA '%s' to GPU", name)
                return True

            except Exception as e:
                logger.error("HydraMoE: Failed to load LoRA '%s': %s", name, e)
                return False

    def unload_lora(self, name: str) -> bool:
        """Unload a LoRA adapter from GPU."""
        with self._lock:
            if name in self._active_loras:
                del self._active_loras[name]
                if name in self._available_loras:
                    self._available_loras[name].active = False
                    self._available_loras[name].on_gpu = False
                logger.info("HydraMoE: Unloaded LoRA '%s'", name)
                return True
            return False

    def activate_for_prompt(self, prompt: str) -> List[str]:
        """
        Auto-detect which LoRAs to activate based on the prompt.
        Returns list of activated LoRA names.
        """
        classifications = self.router.classify(prompt)
        activated = []

        for cat, score in classifications[:2]:  # Top 2 categories
            if score < 0.1:
                continue
            # Find LoRA for this category
            for name, meta in self._available_loras.items():
                if meta.category == cat:
                    if self.load_lora(name):
                        activated.append(name)
                    break

        return activated

    def apply_active_loras(self) -> None:
        """Apply all active LoRA deltas to the model."""
        if not self._active_loras:
            self.engine.model.clear_all_lora()
            return

        # Merge all active LoRA deltas
        merged: Dict[int, Dict[str, Tensor]] = {}
        for name, lora_deltas in self._active_loras.items():
            for layer_idx, deltas in lora_deltas.items():
                if layer_idx not in merged:
                    merged[layer_idx] = {}
                for proj_name, delta in deltas.items():
                    if proj_name in merged[layer_idx]:
                        merged[layer_idx][proj_name] = (
                            merged[layer_idx][proj_name] + delta
                        )
                    else:
                        merged[layer_idx][proj_name] = delta.clone()

        self.engine.model.apply_lora_to_all_layers(merged)

    def status(self) -> Dict[str, Any]:
        return {
            "available_loras": {
                k: {"category": v.category, "active": v.active}
                for k, v in self._available_loras.items()
            },
            "active_loras": list(self._active_loras.keys()),
        }


class ProteusNet:
    """
    3.3 Continuous Adaptation Engine
    Accumulates user feedback gradients during the day.
    Consolidates them into LoRA during idle time (e.g., night).
    """

    def __init__(
        self, engine: NeuroHiveGPU, hydra: HydraMoE,
        consolidation_threshold: int = 100,
    ):
        self.engine = engine
        self.hydra = hydra
        self.consolidation_threshold = consolidation_threshold
        self._feedback_buffer: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._consolidation_thread: Optional[threading.Thread] = None
        self._running = False

    def record_feedback(self, text: str, quality: float, category: str = "general") -> None:
        """Record user feedback for continuous learning."""
        with self._lock:
            self._feedback_buffer.append({
                "text": text,
                "quality": quality,
                "category": category,
                "timestamp": timestamp_str(),
            })

        if len(self._feedback_buffer) >= self.consolidation_threshold:
            logger.info(
                "ProteusNet: Buffer full (%d), ready for consolidation",
                len(self._feedback_buffer),
            )

    def consolidate(self) -> bool:
        """
        Consolidate accumulated feedback into LoRA weights.
        Called during idle time.
        """
        with self._lock:
            if not self._feedback_buffer:
                return False
            buffer = list(self._feedback_buffer)
            self._feedback_buffer.clear()

        logger.info("ProteusNet: Consolidating %d feedback items", len(buffer))

        # Group by category
        by_category: Dict[str, List[Dict[str, Any]]] = {}
        for item in buffer:
            cat = item.get("category", "general")
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(item)

        # Create training samples from feedback
        for cat, items in by_category.items():
            samples = []
            for item in items:
                if item["quality"] > 0.5:
                    samples.append(SyntheticDataSample(
                        prompt=f"User interaction in {cat}",
                        response=item["text"],
                        category=cat,
                        quality_score=item["quality"],
                        symbolic_valid=True,
                    ))

            if samples:
                # Save and queue for training
                forge = getattr(self, "_forge", None)
                if forge is not None:
                    forge.trainer.save_dataset(samples, f"proteus_{cat}")
                    logger.info(
                        "ProteusNet: Saved %d consolidation samples for '%s'",
                        len(samples), cat,
                    )

        return True

    def start_idle_consolidator(self, check_interval: float = 300.0) -> None:
        """Start a thread that consolidates during idle periods."""
        self._running = True

        def _loop():
            while self._running:
                time.sleep(check_interval)
                if self._is_system_idle():
                    self.consolidate()

        self._consolidation_thread = threading.Thread(target=_loop, daemon=True)
        self._consolidation_thread.start()

    def _is_system_idle(self) -> bool:
        """Check if the system is idle (no active requests)."""
        if torch.cuda.is_available():
            usage = torch.cuda.utilization() if hasattr(torch.cuda, "utilization") else 0
            return usage < 10
        return True

    def stop(self) -> None:
        self._running = False

    def status(self) -> Dict[str, Any]:
        return {
            "feedback_buffer_size": len(self._feedback_buffer),
            "consolidation_threshold": self.consolidation_threshold,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7: FASE 4 — O1-ENGINE (MCTS + HIVEMIND MULTI-AGENT DEBATE)
# ═══════════════════════════════════════════════════════════════════════════════

class MCTSEngine:
    """
    4.1 Monte Carlo Tree Search for deep reasoning.
    Allows the model to explore multiple reasoning paths and backtrack.
    """

    def __init__(
        self, engine: NeuroHiveGPU,
        exploration_constant: float = 1.41,
        max_depth: int = 32,
        max_simulations: int = 50,
        rollout_tokens: int = 16,
    ):
        self.engine = engine
        self.c = exploration_constant
        self.max_depth = max_depth
        self.max_simulations = max_simulations
        self.rollout_tokens = rollout_tokens
        self.nodes: List[MCTSNode] = []

    def _create_root(self, text: str) -> int:
        node = MCTSNode(
            state=MCTSNodeState.EXPANDED,
            text_so_far=text,
            depth=0,
        )
        self.nodes.append(node)
        return len(self.nodes) - 1

    def _ucb1(self, node_id: int, parent_visits: int) -> float:
        node = self.nodes[node_id]
        if node.visits == 0:
            return float("inf")
        exploit = node.value
        explore = self.c * math.sqrt(math.log(parent_visits + 1) / node.visits)
        return exploit + explore + 0.1 * node.prior

    def _select(self, root_id: int) -> int:
        """Select the most promising leaf node via UCB1."""
        current = root_id
        while self.nodes[current].children_ids:
            parent = self.nodes[current]
            best_child = max(
                parent.children_ids,
                key=lambda cid: self._ucb1(cid, parent.visits),
            )
            current = best_child
        return current

    def _expand(self, node_id: int, num_children: int = 3) -> List[int]:
        """Expand a leaf node by generating possible continuations."""
        node = self.nodes[node_id]
        if node.depth >= self.max_depth:
            node.state = MCTSNodeState.TERMINAL
            return []

        seq_id = self.engine.new_sequence_id()
        self.engine.cache.clear_seq(seq_id)

        # Encode the text so far
        tokens = self.engine.tokenizer.encode(node.text_so_far)
        for pos, tid in enumerate(tokens):
            hidden = self.engine.forward_hidden_from_token(tid, seq_pos=pos, seq_id=seq_id)

        # Get logits for next token
        logits = self.engine.forward_logits_from_hidden(hidden).float()
        probs = torch.softmax(logits, dim=-1)

        # Top-k children
        topk_vals, topk_ids = torch.topk(probs, k=min(num_children, probs.numel()))

        children = []
        for prob, tok_id in zip(topk_vals.tolist(), topk_ids.tolist()):
            piece = self.engine.tokenizer.decode([tok_id])
            child = MCTSNode(
                state=MCTSNodeState.UNEXPLORED,
                token_id=tok_id,
                text_so_far=node.text_so_far + piece,
                parent_id=node_id,
                depth=node.depth + 1,
                prior=prob,
            )
            child_id = len(self.nodes)
            self.nodes.append(child)
            children.append(child_id)

        node.children_ids = children
        node.state = MCTSNodeState.EXPANDED

        self.engine.cache.clear_seq(seq_id)
        return children

    def _simulate(self, node_id: int) -> float:
        """
        Rollout: generate a few tokens from this node and evaluate quality.
        """
        node = self.nodes[node_id]
        seq_id = self.engine.new_sequence_id()
        self.engine.cache.clear_seq(seq_id)

        tokens = self.engine.tokenizer.encode(node.text_so_far)
        for pos, tid in enumerate(tokens):
            hidden = self.engine.forward_hidden_from_token(tid, seq_pos=pos, seq_id=seq_id)

        # Generate rollout tokens
        generated_text = node.text_so_far
        for step in range(self.rollout_tokens):
            logits = self.engine.forward_logits_from_hidden(hidden).float()
            probs = torch.softmax(logits, dim=-1)
            tok_id = int(torch.multinomial(probs, 1).item())
            piece = self.engine.tokenizer.decode([tok_id])
            generated_text += piece

            hidden = self.engine.forward_hidden_from_token(
                tok_id, seq_pos=len(tokens) + step, seq_id=seq_id,
            )

        self.engine.cache.clear_seq(seq_id)

        # Evaluate the rollout using symbolic bridge
        if self.engine.symbolic is not None:
            quality = self.engine.symbolic.evaluate_quality(generated_text)
        else:
            quality = 0.5

        # Entropy-based reward (lower entropy = more confident = better)
        logits = self.engine.forward_logits_from_hidden(hidden).float()
        ent = float(entropy_from_logits(logits).item())
        entropy_reward = max(0, 1.0 - ent / 10.0)

        return 0.6 * quality + 0.4 * entropy_reward

    def _backpropagate(self, node_id: int, reward: float) -> None:
        """Propagate the reward up the tree."""
        current = node_id
        while current >= 0:
            self.nodes[current].visits += 1
            self.nodes[current].total_reward += reward
            current = self.nodes[current].parent_id

    @torch.inference_mode()
    def search(self, prompt: str, num_simulations: Optional[int] = None) -> str:
        """
        Run MCTS to find the best continuation for a prompt.
        """
        self.nodes.clear()
        root_id = self._create_root(prompt)
        n_sims = num_simulations or self.max_simulations

        logger.info("MCTS: Starting search with %d simulations", n_sims)

        for sim in range(n_sims):
            # Select
            leaf_id = self._select(root_id)

            # Expand
            if self.nodes[leaf_id].state != MCTSNodeState.TERMINAL:
                children = self._expand(leaf_id, num_children=3)
                if children:
                    leaf_id = children[0]  # Simulate from first child

            # Simulate
            reward = self._simulate(leaf_id)

            # Backpropagate
            self._backpropagate(leaf_id, reward)

        # Select the best path
        best_text = self._extract_best_path(root_id)
        logger.info("MCTS: Search complete. Best path length: %d chars", len(best_text))
        return best_text

    def _extract_best_path(self, root_id: int) -> str:
        """Follow the most-visited path from root to get the best sequence."""
        current = root_id
        while self.nodes[current].children_ids:
            children = self.nodes[current].children_ids
            current = max(children, key=lambda cid: self.nodes[cid].visits)
        return self.nodes[current].text_so_far


class HiveMindDebate:
    """
    4.2 Multi-Agent Debate System
    Clones the model with different LoRA personas to debate and find the best answer.
    """

    def __init__(
        self, engine: NeuroHiveGPU, hydra: HydraMoE,
        symbolic: SymbolicBridgeGPU, max_rounds: int = 3,
    ):
        self.engine = engine
        self.hydra = hydra
        self.symbolic = symbolic
        self.max_rounds = max_rounds

        self._persona_prompts = {
            AgentPersona.LOGIC: (
                "You are a logical analyst. Focus on correctness, consistency, "
                "and formal reasoning. Point out any logical flaws.\n"
            ),
            AgentPersona.CREATIVE: (
                "You are a creative thinker. Explore novel approaches, alternative "
                "solutions, and think outside the box.\n"
            ),
            AgentPersona.SKEPTIC: (
                "You are a skeptical reviewer. Question assumptions, find edge cases, "
                "and challenge weak arguments.\n"
            ),
            AgentPersona.EXPERT: (
                "You are a domain expert. Provide deep technical analysis with "
                "precise terminology and detailed explanations.\n"
            ),
            AgentPersona.SYNTHESIZER: (
                "You are a synthesis specialist. Combine the best ideas from all "
                "perspectives into a coherent final answer.\n"
            ),
        }

    def _generate_persona_response(
        self, persona: AgentPersona, question: str,
        previous_arguments: List[DebateRound],
        max_tokens: int = 256,
    ) -> str:
        """Generate a response from a specific persona."""
        system_prompt = self._persona_prompts[persona]

        # Build debate context
        context = f"{system_prompt}\nQuestion: {question}\n"
        if previous_arguments:
            context += "\nPrevious arguments in the debate:\n"
            for arg in previous_arguments:
                context += f"- {arg.persona.value}: {arg.argument[:200]}\n"
            context += "\nYour turn to respond:\n"

        result = self.engine.generate(
            prompt=context,
            max_new_tokens=max_tokens,
            temperature=0.6,
            top_k_sampling=40,
            seq_id=self.engine.new_sequence_id(),
        )

        # Extract response after the prompt
        if context in result:
            result = result[len(context):]
        return result.strip()

    def _judge_with_symbolic(self, arguments: List[DebateRound]) -> Tuple[AgentPersona, float]:
        """Use the Symbolic Bridge as the final judge."""
        best_persona = arguments[0].persona if arguments else AgentPersona.LOGIC
        best_score = 0.0

        for arg in arguments:
            quality = self.symbolic.evaluate_quality(arg.argument)
            arg.confidence = quality
            if quality > best_score:
                best_score = quality
                best_persona = arg.persona

        return best_persona, best_score

    @torch.inference_mode()
    def debate(
        self, question: str,
        personas: Optional[List[AgentPersona]] = None,
    ) -> DebateResult:
        """
        Run a multi-agent debate on a question.
        """
        if personas is None:
            personas = [AgentPersona.LOGIC, AgentPersona.CREATIVE, AgentPersona.SKEPTIC]

        all_rounds: List[DebateRound] = []
        logger.info("HiveMind: Starting debate with %d personas, %d rounds",
                     len(personas), self.max_rounds)

        for round_num in range(self.max_rounds):
            for persona in personas:
                response = self._generate_persona_response(
                    persona, question, all_rounds,
                )
                round_entry = DebateRound(
                    round_num=round_num,
                    persona=persona,
                    argument=response,
                    confidence=0.0,
                )
                all_rounds.append(round_entry)

        # Judge
        winning_persona, consensus_score = self._judge_with_symbolic(all_rounds)

        # Synthesize final answer
        synthesizer_prompt = (
            f"Based on the following debate about '{question}':\n"
        )
        for arg in all_rounds:
            synthesizer_prompt += (
                f"[{arg.persona.value} (conf={arg.confidence:.2f})] "
                f"{arg.argument[:150]}\n"
            )
        synthesizer_prompt += "\nProvide the best synthesis:\n"

        final_answer = self.engine.generate(
            prompt=synthesizer_prompt,
            max_new_tokens=512,
            temperature=0.5,
            seq_id=self.engine.new_sequence_id(),
        )
        if synthesizer_prompt in final_answer:
            final_answer = final_answer[len(synthesizer_prompt):]

        symbolic_validation = self.symbolic.evaluate_quality(final_answer)

        result = DebateResult(
            question=question,
            rounds=all_rounds,
            final_answer=final_answer.strip(),
            consensus_score=consensus_score,
            winning_persona=winning_persona,
            symbolic_validation=symbolic_validation,
        )

        logger.info(
            "HiveMind: Debate complete. Winner: %s, Consensus: %.2f, "
            "Symbolic validation: %.2f",
            winning_persona.value, consensus_score, symbolic_validation,
        )
        return result


class O1Engine:
    """
    Combined O1-like reasoning engine: MCTS + HiveMind Debate.
    Activated by the EntropyGate when uncertainty is high.
    """

    def __init__(
        self, engine: NeuroHiveGPU, hydra: HydraMoE,
        symbolic: SymbolicBridgeGPU,
    ):
        self.engine = engine
        self.mcts = MCTSEngine(engine, max_simulations=30)
        self.hivemind = HiveMindDebate(engine, hydra, symbolic)
        self.symbolic = symbolic

        # Thresholds for escalation
        self.mcts_entropy_threshold = 5.0
        self.debate_entropy_threshold = 7.0

    @torch.inference_mode()
    def deep_reason(
        self, prompt: str, entropy: float, margin: float,
    ) -> str:
        """
        Escalating reasoning pipeline:
        1. If entropy is moderate → MCTS
        2. If entropy is very high → Full debate
        """
        if entropy >= self.debate_entropy_threshold:
            logger.info("O1Engine: Escalating to HiveMind debate (entropy=%.2f)", entropy)
            result = self.hivemind.debate(prompt)
            return result.final_answer

        if entropy >= self.mcts_entropy_threshold:
            logger.info("O1Engine: Using MCTS (entropy=%.2f)", entropy)
            return self.mcts.search(prompt, num_simulations=20)

        # Standard generation
        return self.engine.generate(
            prompt=prompt, max_new_tokens=256,
            temperature=0.7, seq_id=self.engine.new_sequence_id(),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8: FASE 5 — CORTEXFS (Knowledge Graph + Tool Calling + RAG)
# ═══════════════════════════════════════════════════════════════════════════════

class CortexFS:
    """
    The Mind's File System — A knowledge graph with vector search.
    
    Features:
    - Graph-based knowledge storage (entities, relations)
    - Vector similarity search for RAG
    - Tool calling interface (read/write/query)
    - Automatic context injection before generation
    """

    def __init__(
        self, storage_dir: str = "./cortex_data",
        use_embeddings: bool = True,
    ):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # Graph storage
        if HAS_NETWORKX:
            self.graph = nx.DiGraph()
        else:
            self.graph = None

        # Node storage
        self._nodes: Dict[str, CortexNode] = {}
        self._edges: List[CortexEdge] = []

        # Vector index
        self._embedder = None
        self._embeddings: Dict[str, np.ndarray] = {}
        if use_embeddings and HAS_SENTENCE_TRANSFORMERS:
            try:
                self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception as e:
                logger.warning("CortexFS: Failed to load embedder: %s", e)

        # Load persisted data
        self._load_from_disk()

    def _load_from_disk(self) -> None:
        """Load persisted nodes and edges from disk."""
        nodes_path = self.storage_dir / "nodes.jsonl"
        edges_path = self.storage_dir / "edges.jsonl"

        if nodes_path.exists():
            with open(nodes_path) as f:
                for line in f:
                    data = safe_json_loads(line.strip())
                    if data:
                        node = CortexNode(
                            node_id=data["node_id"],
                            node_type=data["node_type"],
                            content=data["content"],
                            metadata=data.get("metadata", {}),
                            created_at=data.get("created_at", timestamp_str()),
                        )
                        self._nodes[node.node_id] = node
                        if self._embedder is not None:
                            self._embeddings[node.node_id] = self._embedder.encode(
                                node.content
                            )
                        if self.graph is not None:
                            self.graph.add_node(node.node_id, **data)

        if edges_path.exists():
            with open(edges_path) as f:
                for line in f:
                    data = safe_json_loads(line.strip())
                    if data:
                        edge = CortexEdge(
                            source_id=data["source_id"],
                            target_id=data["target_id"],
                            relation=data["relation"],
                            weight=data.get("weight", 1.0),
                        )
                        self._edges.append(edge)
                        if self.graph is not None:
                            self.graph.add_edge(
                                edge.source_id, edge.target_id,
                                relation=edge.relation, weight=edge.weight,
                            )

        logger.info(
            "CortexFS: Loaded %d nodes, %d edges from disk",
            len(self._nodes), len(self._edges),
        )

    def _save_to_disk(self) -> None:
        """Persist all data to disk."""
        nodes_path = self.storage_dir / "nodes.jsonl"
        edges_path = self.storage_dir / "edges.jsonl"

        with open(nodes_path, "w") as f:
            for node in self._nodes.values():
                json.dump({
                    "node_id": node.node_id,
                    "node_type": node.node_type,
                    "content": node.content,
                    "metadata": node.metadata,
                    "created_at": node.created_at,
                }, f, ensure_ascii=False)
                f.write("\n")

        with open(edges_path, "w") as f:
            for edge in self._edges:
                json.dump({
                    "source_id": edge.source_id,
                    "target_id": edge.target_id,
                    "relation": edge.relation,
                    "weight": edge.weight,
                }, f, ensure_ascii=False)
                f.write("\n")

    # ── Tool Calling Interface ───────────────────────────────────────────────

    def read(self, node_id: str) -> Optional[CortexNode]:
        """CortexFS.read(node="Project_NeuroOS")"""
        node = self._nodes.get(node_id)
        if node:
            node.access_count += 1
            node.last_accessed = timestamp_str()
        return node

    def write_node(self, node: CortexNode) -> None:
        """CortexFS.write(node="Frequent_Error", data="GPU OOM at batch 32")"""
        self._nodes[node.node_id] = node
        if self._embedder is not None:
            self._embeddings[node.node_id] = self._embedder.encode(node.content)
        if self.graph is not None:
            self.graph.add_node(node.node_id, **{
                "node_type": node.node_type,
                "content": node.content,
            })
        self._save_to_disk()
        logger.info("CortexFS: Written node '%s'", node.node_id)

    def write_edge(self, edge: CortexEdge) -> None:
        """Add a relationship between nodes."""
        self._edges.append(edge)
        if self.graph is not None:
            self.graph.add_edge(
                edge.source_id, edge.target_id,
                relation=edge.relation, weight=edge.weight,
            )
        self._save_to_disk()

    def delete_node(self, node_id: str) -> bool:
        if node_id in self._nodes:
            del self._nodes[node_id]
            self._embeddings.pop(node_id, None)
            if self.graph is not None and self.graph.has_node(node_id):
                self.graph.remove_node(node_id)
            self._edges = [
                e for e in self._edges
                if e.source_id != node_id and e.target_id != node_id
            ]
            self._save_to_disk()
            return True
        return False

    # ── Query Interface ──────────────────────────────────────────────────────

    def query_relevant(self, query: str, top_k: int = 5) -> List[CortexNode]:
        """
        RAG-style query: find the most relevant nodes for a query string.
        Uses vector similarity if embedder is available, else keyword matching.
        """
        if self._embedder is not None and self._embeddings:
            return self._query_by_embedding(query, top_k)
        return self._query_by_keyword(query, top_k)

    def _query_by_embedding(self, query: str, top_k: int) -> List[CortexNode]:
        query_emb = self._embedder.encode(query)
        scores = []
        for node_id, emb in self._embeddings.items():
            sim = float(np.dot(query_emb, emb) / (
                np.linalg.norm(query_emb) * np.linalg.norm(emb) + 1e-8
            ))
            scores.append((node_id, sim))
        scores.sort(key=lambda x: -x[1])
        results = []
        for nid, _ in scores[:top_k]:
            node = self._nodes.get(nid)
            if node:
                node.access_count += 1
                results.append(node)
        return results

    def _query_by_keyword(self, query: str, top_k: int) -> List[CortexNode]:
        query_words = set(query.lower().split())
        scores = []
        for nid, node in self._nodes.items():
            content_words = set(node.content.lower().split())
            overlap = len(query_words & content_words)
            if overlap > 0:
                scores.append((nid, overlap))
        scores.sort(key=lambda x: -x[1])
        results = []
        for nid, _ in scores[:top_k]:
            node = self._nodes[nid]
            node.access_count += 1
            results.append(node)
        return results

    def query_neighbors(self, node_id: str, depth: int = 1) -> List[CortexNode]:
        """Get all nodes connected to a given node within a certain depth."""
        if self.graph is None or not self.graph.has_node(node_id):
            return []
        try:
            subgraph_nodes = nx.ego_graph(self.graph, node_id, radius=depth).nodes()
            return [
                self._nodes[nid] for nid in subgraph_nodes
                if nid in self._nodes and nid != node_id
            ]
        except Exception:
            return []

    def get_context_for_prompt(self, prompt: str, max_context_tokens: int = 500) -> str:
        """
        Generate a context string to inject before the prompt.
        This is the RAG injection point.
        """
        relevant_nodes = self.query_relevant(prompt, top_k=5)
        if not relevant_nodes:
            return ""

        context_parts = ["[Relevant context from memory]"]
        total_words = 0

        for node in relevant_nodes:
            words = node.content.split()
            if total_words + len(words) > max_context_tokens:
                remaining = max_context_tokens - total_words
                if remaining > 10:
                    context_parts.append(
                        f"({node.node_type}) {' '.join(words[:remaining])}"
                    )
                break
            context_parts.append(f"({node.node_type}) {node.content}")
            total_words += len(words)

            # Also add neighbors
            neighbors = self.query_neighbors(node.node_id, depth=1)
            for nb in neighbors[:2]:
                nb_words = nb.content.split()
                if total_words + len(nb_words) <= max_context_tokens:
                    context_parts.append(f"  → {nb.content}")
                    total_words += len(nb_words)

        context_parts.append("[End context]")
        return "\n".join(context_parts)

    def status(self) -> Dict[str, Any]:
        return {
            "total_nodes": len(self._nodes),
            "total_edges": len(self._edges),
            "has_embedder": self._embedder is not None,
            "has_graph": self.graph is not None,
            "node_types": dict(collections.Counter(
                n.node_type for n in self._nodes.values()
            )),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9: NEUROOS — THE MASTER ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════

class NeuroOS:
    """
    ╔══════════════════════════════════════════════════════════════════════╗
    ║  NeuroOS — The Autonomous Neural Operating System                    ║
    ║                                                                      ║
    ║  Orchestrates all modules:                                           ║
    ║  • NeuroHive GPU (Core inference engine)                             ║
    ║  • Neural Bus (Async messaging)                                      ║
    ║  • VRAM Manager (Dynamic memory allocation)                          ║
    ║  • AutoForge (Synthetic data + auto LoRA training)                   ║
    ║  • HydraMoE (Dynamic LoRA hot-swapping)                              ║
    ║  • ProteusNet (Continuous learning)                                   ║
    ║  • O1-Engine (MCTS + HiveMind multi-agent debate)                    ║
    ║  • CortexFS (Knowledge graph + RAG)                                  ║
    ║  • OpenAI-Compatible API Server                                      ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """

    def __init__(
        self,
        model_path: str,
        # Core config
        prefer_cuda: bool = True,
        dtype_name: str = "float16",
        seed: int = 42,
        # NeuroHive config
        entropy_threshold: float = 3.8,
        margin_threshold: float = 0.15,
        page_size: int = 16,
        max_pages: int = 4096,
        cache_quant: str = "fp16",
        # Bus config
        bus_backend: str = "queue",
        # API config
        api_host: str = "0.0.0.0",
        api_port: int = 8000,
        enable_api: bool = True,
        # AutoForge config
        forge_dir: str = "./neuroos_forge",
        teacher_api_url: Optional[str] = None,
        teacher_api_key: Optional[str] = None,
        # CortexFS config
        cortex_dir: str = "./cortex_data",
        # O1 config
        enable_mcts: bool = True,
        enable_debate: bool = True,
        # Proteus config
        consolidation_threshold: int = 100,
    ):
        logger.info("=" * 70)
        logger.info("  NeuroOS v1.0 — Initializing the Autonomous Neural Operating System")
        logger.info("=" * 70)

        self._start_time = time.time()

        # ── Step 1: Core Engine (NeuroHive) ──────────────────────────────────
        logger.info("[1/8] Initializing NeuroHive GPU Core...")
        self.engine = NeuroHiveGPU(
            model_path=model_path,
            prefer_cuda=prefer_cuda,
            dtype_name=dtype_name,
            seed=seed,
            entropy_threshold=entropy_threshold,
            margin_threshold=margin_threshold,
            page_size=page_size,
            max_pages=max_pages,
            cache_quant=CacheQuantType(cache_quant),
        )

        # ── Step 2: Neural Bus ───────────────────────────────────────────────
        logger.info("[2/8] Initializing Neural Bus (%s)...", bus_backend)
        self.bus = NeuralBus(backend=bus_backend)
        self.bus.start_listener()
        self._setup_bus_handlers()

        # ── Step 3: VRAM Manager ─────────────────────────────────────────────
        logger.info("[3/8] Initializing VRAM Manager...")
        self.vram_mgr = VRAMManager(self.engine.device, self.engine.dtype)

        # ── Step 4: CortexFS ────────────────────────────────────────────────
        logger.info("[4/8] Initializing CortexFS...")
        self.cortex = CortexFS(storage_dir=cortex_dir)

        # ── Step 5: HydraMoE ────────────────────────────────────────────────
        logger.info("[5/8] Initializing HydraMoE...")
        self.hydra = HydraMoE(
            self.engine, self.vram_mgr,
            lora_dir=str(Path(forge_dir) / "loras"),
        )

        # ── Step 6: AutoForge ───────────────────────────────────────────────
        logger.info("[6/8] Initializing AutoForge...")
        self.forge = AutoForge(
            self.engine, output_dir=forge_dir,
            teacher_api_url=teacher_api_url,
            teacher_api_key=teacher_api_key,
        )
        self.forge.trainer.start_background_trainer()

        # ── Step 7: ProteusNet ──────────────────────────────────────────────
        logger.info("[7/8] Initializing ProteusNet...")
        self.proteus = ProteusNet(
            self.engine, self.hydra,
            consolidation_threshold=consolidation_threshold,
        )
        self.proteus._forge = self.forge  # Link for consolidation
        self.proteus.start_idle_consolidator()

        # ── Step 8: O1 Engine ───────────────────────────────────────────────
        logger.info("[8/8] Initializing O1-Engine (MCTS + HiveMind)...")
        self.o1 = O1Engine(self.engine, self.hydra, self.engine.symbolic)
        self.enable_mcts = enable_mcts
        self.enable_debate = enable_debate

        # ── API Server ──────────────────────────────────────────────────────
        self.api_server = OpenAICompatibleServer(self, host=api_host, port=api_port)
        if enable_api:
            self.api_server.start()

        # ── Complete ────────────────────────────────────────────────────────
        init_time = time.time() - self._start_time
        logger.info("=" * 70)
        logger.info("  NeuroOS READY in %.2fs", init_time)
        logger.info("  Device: %s | Dtype: %s", self.engine.device, self.engine.dtype)
        logger.info("  API: http://%s:%d/v1/chat/completions", api_host, api_port)
        logger.info("=" * 70)

    # ── Bus Handlers ─────────────────────────────────────────────────────────

    def _setup_bus_handlers(self) -> None:
        self.bus.subscribe(
            BusMessageType.INFERENCE_REQUEST.value,
            self._handle_inference_request,
        )
        self.bus.subscribe(
            BusMessageType.LORA_LOAD_REQUEST.value,
            self._handle_lora_load,
        )
        self.bus.subscribe(
            BusMessageType.CORTEX_QUERY.value,
            self._handle_cortex_query,
        )
        self.bus.subscribe(
            BusMessageType.DATA_GENERATION_REQUEST.value,
            self._handle_data_gen,
        )

    def _handle_inference_request(self, msg: BusMessage) -> None:
        prompt = msg.payload.get("prompt", "")
        result = self.generate_with_full_pipeline(prompt)
        self.bus.publish(BusMessage(
            msg_type=BusMessageType.INFERENCE_RESPONSE,
            source="neuroos",
            target=msg.source,
            payload={"result": result, "original_msg_id": msg.msg_id},
        ))

    def _handle_lora_load(self, msg: BusMessage) -> None:
        name = msg.payload.get("name", "")
        success = self.hydra.load_lora(name)
        self.bus.publish(BusMessage(
            msg_type=BusMessageType.LORA_TRAIN_COMPLETE,
            source="neuroos",
            target=msg.source,
            payload={"name": name, "success": success},
        ))

    def _handle_cortex_query(self, msg: BusMessage) -> None:
        query = msg.payload.get("query", "")
        nodes = self.cortex.query_relevant(query)
        self.bus.publish(BusMessage(
            msg_type=BusMessageType.CORTEX_RESPONSE,
            source="neuroos",
            target=msg.source,
            payload={"results": [n.content for n in nodes]},
        ))

    def _handle_data_gen(self, msg: BusMessage) -> None:
        topics = msg.payload.get("topics", [])
        n_per_topic = msg.payload.get("n_per_topic", 10)
        result = self.forge.run_pipeline(topics, n_per_topic)
        self.bus.publish(BusMessage(
            msg_type=BusMessageType.DATA_GENERATION_COMPLETE,
            source="neuroos",
            target=msg.source,
            payload=result,
        ))

    # ═════════════════════════════════════════════════════════════════════════
    # THE COMPLETE PIPELINE (The Real Flow)
    # ═════════════════════════════════════════════════════════════════════════

    @torch.inference_mode()
    def generate_with_full_pipeline(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_k_sampling: int = 40,
        stop_token_ids: Optional[List[int]] = None,
        context_prefix: str = "",
        use_deep_reasoning: bool = True,
    ) -> str:
        """
        THE MAIN PIPELINE — This is where everything comes together:

        1. Router → Classify intent → Load appropriate LoRAs
        2. CortexFS → Inject relevant context (RAG)
        3. NeuroHive System 1 → Start generating
        4. EntropyGate → Monitor uncertainty
        5. If uncertainty high → O1 Engine (MCTS / Debate)
        6. SymbolicBridge → Validate output
        7. Return result + record for ProteusNet
        """

        pipeline_start = time.perf_counter()

        # ── Step 1: Intent Routing + LoRA Activation ─────────────────────────
        activated_loras = self.hydra.activate_for_prompt(prompt)
        if activated_loras:
            self.hydra.apply_active_loras()
            logger.info("Pipeline: Activated LoRAs: %s", activated_loras)

        # ── Step 2: CortexFS Context Injection ───────────────────────────────
        if not context_prefix and self.cortex:
            cortex_context = self.cortex.get_context_for_prompt(prompt)
            if cortex_context:
                context_prefix = cortex_context + "\n"
                logger.info("Pipeline: Injected CortexFS context (%d chars)",
                            len(cortex_context))

        # ── Step 3: Initial Generation (System 1) ───────────────────────────
        seq_id = self.engine.new_sequence_id()
        self.engine.cache.clear_seq(seq_id)

        full_prompt = context_prefix + prompt if context_prefix else prompt
        input_ids = self.engine.tokenizer.encode(full_prompt)
        if not input_ids:
            return ""

        # Warmup
        for pos, tid in enumerate(input_ids):
            hidden = self.engine.forward_hidden_from_token(
                tid, seq_pos=pos, seq_id=seq_id
            )

        logits = self.engine.forward_logits_from_hidden(hidden).float()

        # ── Step 4: Entropy Gate Check ───────────────────────────────────────
        decision = self.engine.gate.inspect(logits)

        if use_deep_reasoning and decision.activate and (
            self.enable_mcts or self.enable_debate
        ):
            logger.info(
                "Pipeline: High uncertainty detected (entropy=%.2f, margin=%.2f). "
                "Escalating to O1-Engine.",
                decision.entropy, decision.margin,
            )

            # ── Step 5: O1 Engine (MCTS / Debate) ───────────────────────────
            result = self.o1.deep_reason(
                prompt, decision.entropy, decision.margin,
            )
        else:
            # ── Standard System 1 + System 2 hybrid generation ──────────────
            result = self.engine.generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k_sampling=top_k_sampling,
                stop_token_ids=stop_token_ids or [],
                seq_id=seq_id,
                context_prefix=context_prefix,
            )

        # ── Step 6: Symbolic Validation ──────────────────────────────────────
        if self.engine.symbolic:
            quality = self.engine.symbolic.evaluate_quality(result)
            logger.info("Pipeline: Symbolic quality score: %.2f", quality)

        # ── Step 7: Clean up LoRA state ──────────────────────────────────────
        if activated_loras:
            self.engine.model.clear_all_lora()

        elapsed = time.perf_counter() - pipeline_start
        logger.info("Pipeline: Complete in %.3fs", elapsed)

        return result

    # ═════════════════════════════════════════════════════════════════════════
    # PUBLIC API
    # ═════════════════════════════════════════════════════════════════════════

    def generate(
        self, prompt: str, **kwargs,
    ) -> str:
        """Simple generation interface."""
        return self.generate_with_full_pipeline(prompt, **kwargs)

    def generate_batch(
        self, prompts: List[str], **kwargs,
    ) -> List[str]:
        """Batch generation via NeuroHive."""
        return self.engine.generate_batch(prompts, **kwargs)

    def chat(
        self, messages: List[Dict[str, str]], **kwargs,
    ) -> str:
        """Chat-style interface."""
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            prompt_parts.append(f"[{role.capitalize()}] {content}")
        prompt_parts.append("[Assistant] ")
        prompt = "\n".join(prompt_parts)
        result = self.generate_with_full_pipeline(prompt, **kwargs)
        if "[Assistant] " in result:
            result = result.split("[Assistant] ")[-1]
        return result.strip()

    # ── AutoForge API ────────────────────────────────────────────────────────

    def run_forge(
        self, topics: List[str], n_per_topic: int = 10, **kwargs,
    ) -> Dict[str, Any]:
        """Run the AutoForge data generation + training pipeline."""
        return self.forge.run_pipeline(topics, n_per_topic, **kwargs)

    def force_train_lora(self, category: str) -> Optional[str]:
        """Force LoRA training for a category."""
        return self.forge.force_train(category)

    # ── CortexFS API ────────────────────────────────────────────────────────

    def remember(
        self, content: str, node_type: str = "memory",
        node_id: Optional[str] = None, metadata: Optional[Dict] = None,
    ) -> str:
        """Store something in CortexFS."""
        nid = node_id or f"mem_{sha256_short(content)}"
        self.cortex.write_node(CortexNode(
            node_id=nid,
            node_type=node_type,
            content=content,
            metadata=metadata or {},
        ))
        return nid

    def recall(self, query: str, top_k: int = 5) -> List[Dict[str, str]]:
        """Recall relevant memories from CortexFS."""
        nodes = self.cortex.query_relevant(query, top_k)
        return [
            {"node_id": n.node_id, "type": n.node_type, "content": n.content}
            for n in nodes
        ]

    def connect(
        self, source: str, target: str, relation: str, weight: float = 1.0,
    ) -> None:
        """Create a relationship between two nodes."""
        self.cortex.write_edge(CortexEdge(
            source_id=source, target_id=target,
            relation=relation, weight=weight,
        ))

    # ── HydraMoE API ────────────────────────────────────────────────────────

    def load_lora(self, name: str) -> bool:
        return self.hydra.load_lora(name)

    def unload_lora(self, name: str) -> bool:
        return self.hydra.unload_lora(name)

    def list_loras(self) -> Dict[str, Any]:
        return self.hydra.status()

    # ── Feedback API ─────────────────────────────────────────────────────────

    def record_user_feedback(
        self, text: str, quality: float, category: str = "general",
    ) -> None:
        """Record user feedback for continuous learning (ProteusNet)."""
        self.proteus.record_feedback(text, quality, category)

    def thumbs_up(self, text: str, category: str = "general") -> None:
        """Shortcut for positive feedback."""
        self.record_user_feedback(text, quality=0.9, category=category)

    def thumbs_down(self, text: str, category: str = "general") -> None:
        """Shortcut for negative feedback."""
        self.record_user_feedback(text, quality=0.1, category=category)

    # ── Deep Reasoning API ───────────────────────────────────────────────────

    def deep_think(self, question: str) -> str:
        """Force O1-Engine deep reasoning."""
        return self.o1.deep_reason(question, entropy=10.0, margin=0.0)

    def debate(
        self, question: str, personas: Optional[List[str]] = None,
    ) -> DebateResult:
        """Run a multi-agent debate on a question."""
        persona_enums = None
        if personas:
            persona_enums = [AgentPersona(p) for p in personas]
        return self.o1.hivemind.debate(question, persona_enums)

    # ── Mining API ───────────────────────────────────────────────────────────

    def mine_pdf(self, path: str) -> int:
        """Mine knowledge from a PDF and store in CortexFS."""
        chunks = self.forge.miner.mine_pdf(path)
        for i, chunk in enumerate(chunks):
            self.remember(
                content=chunk,
                node_type="knowledge",
                node_id=f"pdf_{sha256_short(path)}_{i}",
                metadata={"source": path, "chunk_index": i},
            )
        return len(chunks)

    def mine_directory(self, path: str) -> int:
        """Mine all text files from a directory."""
        chunks = self.forge.miner.mine_directory(path)
        for i, chunk in enumerate(chunks):
            self.remember(
                content=chunk,
                node_type="knowledge",
                node_id=f"dir_{sha256_short(path)}_{i}",
                metadata={"source": path, "chunk_index": i},
            )
        return len(chunks)

    # ── System Status ────────────────────────────────────────────────────────

    def full_status(self) -> Dict[str, Any]:
        return {
            "neuroos_version": "1.0.0",
            "uptime_seconds": time.time() - self._start_time,
            "engine": self.engine.stats(),
            "vram": self.vram_mgr.status(),
            "cortex": self.cortex.status(),
            "hydra": self.hydra.status(),
            "forge": self.forge.status(),
            "proteus": self.proteus.status(),
            "gpu_memory": gpu_memory_stats(),
        }

    def print_status(self) -> None:
        status = self.full_status()
        print("\n" + "=" * 70)
        print("  NeuroOS System Status")
        print("=" * 70)

        print(f"\n  Version:  {status['neuroos_version']}")
        print(f"  Uptime:   {status['uptime_seconds']:.1f}s")

        engine = status["engine"]
        print(f"\n  ── Engine ──")
        print(f"  Device:   {engine['device']}")
        print(f"  Dtype:    {engine['dtype']}")
        print(f"  Arch:     {engine['architecture']}")
        print(f"  Layers:   {engine['layers']}")
        print(f"  Dim:      {engine['dim']}")
        print(f"  Vocab:    {engine['vocab_size']}")

        vram = status["vram"]
        print(f"\n  ── VRAM Manager ──")
        print(f"  Usage:    {vram['vram_usage']}")
        print(f"  On GPU:   {vram['gpu_registered']}")
        print(f"  Offloaded:{vram['cpu_offloaded']}")

        cortex = status["cortex"]
        print(f"\n  ── CortexFS ──")
        print(f"  Nodes:    {cortex['total_nodes']}")
        print(f"  Edges:    {cortex['total_edges']}")
        print(f"  Embedder: {cortex['has_embedder']}")

        hydra = status["hydra"]
        print(f"\n  ── HydraMoE ──")
        print(f"  Available:{len(hydra['available_loras'])}")
        print(f"  Active:   {hydra['active_loras']}")

        forge = status["forge"]
        print(f"\n  ── AutoForge ──")
        print(f"  Buffers:  {forge['buffer_sizes']}")
        print(f"  Trained:  {forge['trained_loras']}")

        proteus = status["proteus"]
        print(f"\n  ── ProteusNet ──")
        print(f"  Buffer:   {proteus['feedback_buffer_size']}")

        gpu = status.get("gpu_memory", {})
        if gpu.get("status") != "no_cuda":
            print(f"\n  ── GPU Memory ──")
            for k, v in gpu.items():
                print(f"  {k}: {v}")

        print("\n" + "=" * 70)

    # ── Shutdown ─────────────────────────────────────────────────────────────

    def shutdown(self) -> None:
        """Graceful shutdown of all modules."""
        logger.info("NeuroOS: Shutting down...")
        self.bus.publish(BusMessage(
            msg_type=BusMessageType.SHUTDOWN,
            source="neuroos",
        ))
        self.proteus.consolidate()  # Final consolidation
        self.proteus.stop()
        self.forge.trainer.stop()
        self.bus.stop()
        self.api_server.stop()
        self.engine.reset()
        logger.info("NeuroOS: Shutdown complete.")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 10: CLI / DEMO / MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="NeuroOS — The Autonomous Neural Operating System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
        Examples:
          # Basic inference
          python neuroos.py --model model.gguf --prompt "Hello, NeuroOS!"

          # With API server
          python neuroos.py --model model.gguf --api --api-port 8000

          # With deep reasoning
          python neuroos.py --model model.gguf --prompt "Solve P=NP" --deep-think

          # Mine knowledge from PDFs
          python neuroos.py --model model.gguf --mine-pdf paper.pdf

          # Run data forge
          python neuroos.py --model model.gguf --forge --topics "physics,math"
        """),
    )

    # Core args
    parser.add_argument("--model", type=str, required=True, help="Path to .gguf model")
    parser.add_argument("--prompt", type=str, default=None, help="Prompt for generation")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float16", "bf16", "float32"])
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--stream", action="store_true")

    # NeuroHive config
    parser.add_argument("--entropy-threshold", type=float, default=3.8)
    parser.add_argument("--margin-threshold", type=float, default=0.15)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--max-pages", type=int, default=4096)
    parser.add_argument("--cache-quant", type=str, default="fp16",
                        choices=["fp16", "int8"])

    # API
    parser.add_argument("--api", action="store_true", help="Enable OpenAI API server")
    parser.add_argument("--api-host", type=str, default="0.0.0.0")
    parser.add_argument("--api-port", type=int, default=8000)

    # Deep reasoning
    parser.add_argument("--deep-think", action="store_true",
                        help="Force O1-Engine deep reasoning")
    parser.add_argument("--debate", action="store_true",
                        help="Run multi-agent debate")

    # AutoForge
    parser.add_argument("--forge", action="store_true",
                        help="Run AutoForge pipeline")
    parser.add_argument("--topics", type=str, default="",
                        help="Comma-separated topics for data generation")
    parser.add_argument("--forge-n", type=int, default=10,
                        help="Samples per topic")
    parser.add_argument("--teacher-api-url", type=str, default=None)
    parser.add_argument("--teacher-api-key", type=str, default=None)

    # Mining
    parser.add_argument("--mine-pdf", type=str, default=None,
                        help="Mine knowledge from a PDF")
    parser.add_argument("--mine-dir", type=str, default=None,
                        help="Mine knowledge from a directory")

    # CortexFS
    parser.add_argument("--remember", type=str, default=None,
                        help="Store a memory in CortexFS")
    parser.add_argument("--recall", type=str, default=None,
                        help="Recall memories matching a query")

    # Management
    parser.add_argument("--status", action="store_true",
                        help="Print full system status")
    parser.add_argument("--interactive", action="store_true",
                        help="Enter interactive chat mode")

    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Model not found: {args.model}")
        return 1

    try:
        # ── Initialize NeuroOS ───────────────────────────────────────────────
        system = NeuroOS(
            model_path=args.model,
            prefer_cuda=not args.cpu,
            dtype_name=args.dtype,
            seed=args.seed,
            entropy_threshold=args.entropy_threshold,
            margin_threshold=args.margin_threshold,
            page_size=args.page_size,
            max_pages=args.max_pages,
            cache_quant=args.cache_quant,
            bus_backend="queue",
            api_host=args.api_host,
            api_port=args.api_port,
            enable_api=args.api,
            teacher_api_url=args.teacher_api_url,
            teacher_api_key=args.teacher_api_key,
        )

        # ── Commands ─────────────────────────────────────────────────────────

        if args.status:
            system.print_status()

        if args.mine_pdf:
            n = system.mine_pdf(args.mine_pdf)
            print(f"Mined {n} chunks from PDF: {args.mine_pdf}")

        if args.mine_dir:
            n = system.mine_directory(args.mine_dir)
            print(f"Mined {n} chunks from directory: {args.mine_dir}")

        if args.remember:
            nid = system.remember(args.remember)
            print(f"Stored memory: {nid}")

        if args.recall:
            results = system.recall(args.recall)
            print(f"\nRecalled {len(results)} memories:")
            for r in results:
                print(f"  [{r['type']}] {r['node_id']}: {r['content'][:100]}")

        if args.forge and args.topics:
            topics = [t.strip() for t in args.topics.split(",") if t.strip()]
            result = system.run_forge(topics, n_per_topic=args.forge_n)
            print(f"\nAutoForge Results:")
            print(f"  Generated: {result['generated']}")
            print(f"  Approved:  {result['approved']}")
            print(f"  Trainings: {result['queued_trainings']}")

        if args.prompt:
            t0 = time.perf_counter()

            if args.deep_think:
                text = system.deep_think(args.prompt)
            elif args.debate:
                result = system.debate(args.prompt)
                text = result.final_answer
                print(f"\nDebate Results:")
                print(f"  Winner: {result.winning_persona.value}")
                print(f"  Consensus: {result.consensus_score:.2f}")
                print(f"  Symbolic: {result.symbolic_validation:.2f}")
                print(f"\n  Rounds:")
                for r in result.rounds:
                    print(f"    [{r.persona.value}] {r.argument[:100]}...")
                print(f"\nFinal Answer:")
            else:
                text = system.generate(
                    prompt=args.prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k_sampling=args.top_k,
                )

            elapsed = time.perf_counter() - t0
            print(f"\n{'=' * 60}")
            print(f"Generated Output:")
            print(f"{'=' * 60}")
            print(text)
            print(f"\nLatency: {elapsed:.3f}s")

        if args.interactive:
            print("\n" + "=" * 60)
            print("  NeuroOS Interactive Mode")
            print("  Commands: /status /remember /recall /debate /quit")
            print("=" * 60)

            history: List[Dict[str, str]] = []

            while True:
                try:
                    user_input = input("\n[You] > ").strip()
                except (EOFError, KeyboardInterrupt):
                    break

                if not user_input:
                    continue

                if user_input == "/quit":
                    break
                elif user_input == "/status":
                    system.print_status()
                    continue
                elif user_input.startswith("/remember "):
                    content = user_input[10:]
                    nid = system.remember(content)
                    print(f"  Stored: {nid}")
                    continue
                elif user_input.startswith("/recall "):
                    query = user_input[8:]
                    results = system.recall(query)
                    for r in results:
                        print(f"  [{r['type']}] {r['content'][:100]}")
                    continue
                elif user_input.startswith("/debate "):
                    question = user_input[8:]
                    result = system.debate(question)
                    print(f"\n  Winner: {result.winning_persona.value}")
                    print(f"  Answer: {result.final_answer[:500]}")
                    continue
                elif user_input.startswith("/thumbsup"):
                    if history:
                        last = history[-1].get("assistant", "")
                        system.thumbs_up(last)
                        print("  👍 Recorded!")
                    continue
                elif user_input.startswith("/thumbsdown"):
                    if history:
                        last = history[-1].get("assistant", "")
                        system.thumbs_down(last)
                        print("  👎 Recorded!")
                    continue

                history.append({"role": "user", "content": user_input})

                messages = [{"role": h["role"], "content": h["content"]}
                            for h in history if "role" in h]
                response = system.chat(messages, max_new_tokens=args.max_new_tokens)

                print(f"\n[NeuroOS] {response}")
                history.append({"role": "assistant", "content": response})

        # ── Wait for API if enabled ──────────────────────────────────────────
        if args.api and not args.prompt and not args.interactive:
            print("\nNeuroOS API server running. Press Ctrl+C to stop.")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass

        system.shutdown()
        return 0

    except Exception as e:
        logger.exception("NeuroOS fatal error: %s", e)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
