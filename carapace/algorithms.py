"""Lightweight similarity algorithms used by Carapace.

Implemented algorithms:
- MinHash signatures + LSH bands
- SimHash (64-bit)
- Winnowing fingerprints
"""

from __future__ import annotations

import hashlib
from collections import deque
from collections.abc import Iterable


def _stable_hash64(text: str, seed: int = 0) -> int:
    material = f"{seed}:{text}".encode()
    digest = hashlib.blake2b(material, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def build_shingles(tokens: list[str], k: int = 3) -> set[str]:
    if not tokens:
        return set()
    if len(tokens) < k:
        return {" ".join(tokens)}
    return {" ".join(tokens[i : i + k]) for i in range(0, len(tokens) - k + 1)}


def minhash_signature(tokens: list[str], num_perm: int = 64, shingle_k: int = 3) -> list[int]:
    shingles = build_shingles(tokens, k=shingle_k)
    if not shingles:
        return [0] * num_perm

    signature: list[int] = []
    for seed in range(num_perm):
        signature.append(min(_stable_hash64(shingle, seed=seed) for shingle in shingles))
    return signature


def minhash_similarity(sig_a: list[int], sig_b: list[int]) -> float:
    if not sig_a or not sig_b or len(sig_a) != len(sig_b):
        return 0.0
    matches = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
    return matches / len(sig_a)


def minhash_lsh_bands(signature: list[int], bands: int = 8) -> list[tuple[int, tuple[int, ...]]]:
    if not signature or bands <= 0:
        return []

    rows = max(1, len(signature) // bands)
    out: list[tuple[int, tuple[int, ...]]] = []
    for idx in range(bands):
        start = idx * rows
        end = start + rows
        if start >= len(signature):
            break
        band = tuple(signature[start:end])
        if band:
            out.append((idx, band))
    return out


def simhash64(tokens: Iterable[str], bits: int = 64) -> int:
    if bits <= 0:
        raise ValueError("bits must be positive")

    weights = [0] * bits
    seen = False
    for token in tokens:
        seen = True
        h = _stable_hash64(token)
        for i in range(bits):
            bit_is_set = (h >> i) & 1
            weights[i] += 1 if bit_is_set else -1

    if not seen:
        return 0

    result = 0
    for i, weight in enumerate(weights):
        if weight >= 0:
            result |= 1 << i
    return result


def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def simhash_similarity(a: int, b: int, bits: int = 64) -> float:
    if bits <= 0:
        return 0.0
    dist = hamming_distance(a, b)
    return max(0.0, 1.0 - (dist / bits))


def simhash_chunks(value: int, bits: int = 64, chunk_bits: int = 16) -> list[tuple[int, int]]:
    if chunk_bits <= 0:
        return []
    chunk_count = max(1, bits // chunk_bits)
    mask = (1 << chunk_bits) - 1
    chunks: list[tuple[int, int]] = []
    for idx in range(chunk_count):
        part = (value >> (idx * chunk_bits)) & mask
        chunks.append((idx, part))
    return chunks


def winnowing_fingerprints(tokens: list[str], k: int = 5, window: int = 4) -> set[int]:
    """Compute winnowing fingerprints from token k-grams.

    Uses stable hashes and keeps the minimum hash in each sliding window.
    """
    if not tokens:
        return set()
    if k <= 0:
        k = 1
    if window <= 0:
        window = 1

    if len(tokens) < k:
        grams = [" ".join(tokens)]
    else:
        grams = [" ".join(tokens[i : i + k]) for i in range(0, len(tokens) - k + 1)]

    hashes = [_stable_hash64(gram) for gram in grams]
    if len(hashes) <= window:
        return {min(hashes)} if hashes else set()

    selected: set[int] = set()
    current: deque[tuple[int, int]] = deque()  # (index, hash)

    for idx, value in enumerate(hashes):
        while current and current[-1][1] >= value:
            current.pop()
        current.append((idx, value))

        min_valid_index = idx - window + 1
        while current and current[0][0] < min_valid_index:
            current.popleft()

        if idx >= window - 1 and current:
            selected.add(current[0][1])

    return selected
