"""Holographic Reduced Representations (HRR) with phase encoding.

Ported from Hermes Agent's holographic plugin (MIT license).
Provides the vector-symbolic core for Butterfly Dream's semantic encoding.

HRRs encode compositional structure into fixed-width distributed vectors
using phase vectors (angles in [0, 2π)). Operations:
  bind   — circular convolution (phase addition)     — associate two concepts
  unbind — circular correlation (phase subtraction)   — retrieve a bound value
  bundle — superposition (circular mean)              — merge multiple concepts

References:
  Plate (1995) — Holographic Reduced Representations
  Gayler (2004) — Vector Symbolic Architectures
"""

import hashlib
import logging
import math
import struct

import jieba

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

logger = logging.getLogger(__name__)

_TWO_PI = 2.0 * math.pi


def _require_numpy() -> None:
    if not _HAS_NUMPY:
        raise RuntimeError("numpy is required for holographic operations")


def encode_atom(word: str, dim: int = 1024) -> "np.ndarray":
    """Deterministic phase vector via SHA-256 counter blocks.

    Uses hashlib (not numpy RNG) for cross-platform reproducibility.
    Returns np.float64 array of shape (dim,) with values in [0, 2π).
    """
    _require_numpy()
    values_per_block = 16  # Each SHA-256 digest = 32 bytes = 16 uint16 values
    blocks_needed = math.ceil(dim / values_per_block)
    uint16_values: list[int] = []
    for i in range(blocks_needed):
        digest = hashlib.sha256(f"{word}:{i}".encode()).digest()
        uint16_values.extend(struct.unpack("<16H", digest))
    phases = np.array(uint16_values[:dim], dtype=np.float64) * (_TWO_PI / 65536.0)
    return phases


def _check_dim(a: "np.ndarray", b: "np.ndarray") -> None:
    if a.shape != b.shape:
        raise ValueError(f"HRR dimension mismatch: {a.shape} vs {b.shape}")


def bind(a: "np.ndarray", b: "np.ndarray") -> "np.ndarray":
    """Circular convolution = element-wise phase addition.
    Associates two concepts into a single composite vector.
    """
    _require_numpy()
    _check_dim(a, b)
    return (a + b) % _TWO_PI


def unbind(memory: "np.ndarray", key: "np.ndarray") -> "np.ndarray":
    """Circular correlation = element-wise phase subtraction.
    Retrieves the value associated with a key from a memory vector.
    """
    _require_numpy()
    _check_dim(memory, key)
    return (memory - key) % _TWO_PI


def bundle(*vectors: "np.ndarray") -> "np.ndarray":
    """Superposition via circular mean of complex exponentials.
    Merges multiple vectors into one similar to each input.
    Can hold O(sqrt(dim)) items before similarity degrades.
    """
    _require_numpy()
    complex_sum = np.sum([np.exp(1j * v) for v in vectors], axis=0)
    return np.angle(complex_sum) % _TWO_PI


def similarity(a: "np.ndarray", b: "np.ndarray") -> float:
    """Phase cosine similarity. Range [-1, 1].
    1.0 = identical, near 0.0 = unrelated, -1.0 = anti-correlated.
    """
    _require_numpy()
    _check_dim(a, b)
    return float(np.mean(np.cos(a - b)))


def encode_text(text: str, dim: int = 1024) -> "np.ndarray":
    """Bag-of-words encoding: bundle atom vectors for each token.

    Tokenizes by lowercasing, splitting on whitespace (for English),
    and using jieba for CJK word segmentation. CJK text without spaces
    would otherwise become a single HRR token, losing semantic signal.
    Empty text returns encode_atom("__hrr_empty__", dim).
    """
    _require_numpy()
    tokens = _tokenize_hrr(text)
    tokens = [t for t in tokens if t]
    if not tokens:
        return encode_atom("__hrr_empty__", dim)
    atom_vectors = [encode_atom(token, dim) for token in tokens]
    return bundle(*atom_vectors)


def _tokenize_hrr(text: str) -> list[str]:
    """Split text into tokens for HRR encoding.

    Uses jieba for CJK segments and regex for English/Latin words.
    This mirrors retrieval.tokenize() but returns a list (order not important
    for HRR bundling, but preserving individual occurrences matters).
    """
    import re
    tokens = []
    # English/Latin words (e.g. "VS", "Code", "Python", "C++")
    for token in re.findall(r'[a-zA-Z][a-zA-Z0-9_\-+#]{1,}', text):
        tokens.append(token.lower())
    # CJK segments — jieba word segmentation
    cjk_parts = re.findall(r'[\u4e00-\u9fff]+', text)
    for part in cjk_parts:
        tokens.extend(jieba.cut(part))
    return tokens


def encode_fact(content: str, entities: list[str], dim: int = 1024) -> "np.ndarray":
    """Structured encoding: content bound to ROLE_CONTENT, entities bound to ROLE_ENTITY.

    Enables algebraic extraction:
        unbind(fact, bind(entity, ROLE_ENTITY)) ≈ content_vector
    """
    _require_numpy()
    role_content = encode_atom("__hrr_role_content__", dim)
    role_entity = encode_atom("__hrr_role_entity__", dim)
    components: list[np.ndarray] = [
        bind(encode_text(content, dim), role_content)
    ]
    for entity in entities:
        components.append(bind(encode_atom(entity.lower(), dim), role_entity))
    return bundle(*components)


def phases_to_bytes(phases: "np.ndarray") -> bytes:
    """Serialize phase vector to bytes. ~8 KB at dim=1024."""
    _require_numpy()
    return phases.tobytes()


def bytes_to_phases(data: bytes) -> "np.ndarray":
    """Deserialize bytes back to phase vector.
    Uses .copy() to return a mutable array from the read-only buffer.
    """
    _require_numpy()
    return np.frombuffer(data, dtype=np.float64).copy()


def snr_estimate(dim: int, n_items: int) -> float:
    """Signal-to-noise ratio for holographic storage.
    SNR = sqrt(dim / n_items). Falls below 2.0 when n_items > dim / 4.
    """
    _require_numpy()
    if n_items <= 0:
        return float("inf")
    snr = math.sqrt(dim / n_items)
    if snr < 2.0:
        logger.warning(
            "HRR storage near capacity: SNR=%.2f (dim=%d, n_items=%d). "
            "Retrieval accuracy may degrade.",
            snr, dim, n_items,
        )
    return snr
