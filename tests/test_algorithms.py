from carapace.algorithms import (
    build_shingles,
    hamming_distance,
    minhash_lsh_bands,
    minhash_signature,
    minhash_similarity,
    simhash64,
    simhash_chunks,
    simhash_similarity,
    winnowing_fingerprints,
)


def test_minhash_similarity_high_for_identical_sequences() -> None:
    tokens = ["fix", "cache", "key", "bug"]
    sig_a = minhash_signature(tokens, num_perm=32, shingle_k=2)
    sig_b = minhash_signature(tokens, num_perm=32, shingle_k=2)
    assert minhash_similarity(sig_a, sig_b) == 1.0


def test_minhash_lsh_band_generation() -> None:
    sig = list(range(16))
    bands = minhash_lsh_bands(sig, bands=4)
    assert len(bands) == 4
    assert bands[0][0] == 0


def test_simhash_and_hamming_distance() -> None:
    a = simhash64(["fix", "cache", "key"])
    b = simhash64(["fix", "cache", "key"])
    c = simhash64(["docs", "typo", "readme"])

    assert hamming_distance(a, b) == 0
    assert simhash_similarity(a, b) == 1.0
    assert simhash_similarity(a, c) < 1.0


def test_simhash_chunks_partition_value() -> None:
    value = simhash64(["one", "two", "three"])
    chunks = simhash_chunks(value, bits=64, chunk_bits=16)
    assert len(chunks) == 4


def test_winnowing_returns_stable_fingerprints() -> None:
    tokens = ["fix", "cache", "key", "collision", "bug", "fix"]
    fp_a = winnowing_fingerprints(tokens, k=3, window=2)
    fp_b = winnowing_fingerprints(tokens, k=3, window=2)
    assert fp_a == fp_b
    assert len(fp_a) > 0


def test_build_shingles_with_short_input() -> None:
    shingles = build_shingles(["single"], k=3)
    assert shingles == {"single"}
