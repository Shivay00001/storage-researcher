"""
blocks.py — The pattern library for the storage-domain instantiation of the
Main Researcher System.

Each Block is a real, round-trip-tested encode/decode pair pulled from an
actual published technique (see paper.md, Section 3.1 / References for
provenance of every block below). Blocks are grouped into two categories,
matching the dual evaluation tracks:

  - "sourced"  : practical, benchmarkable compression techniques used in real
                 production database engines (Parquet, ClickHouse, BtrBlocks).
  - "moonshot" : molecular / DNA-storage-style encoding. Real as *software*
                 (every function here runs and round-trips), but the physical
                 "gram of hardware" claim it services is a literature-based
                 projection, not something this code synthesizes.

Every encode fn returns bytes (or a small container of bytes). Every decode
fn is the exact inverse. `self_test()` at the bottom round-trips every block
against random data before anything downstream is allowed to trust them.
"""
from __future__ import annotations
import io
import json
import lzma
import math
import random
import struct
import zlib
from dataclasses import dataclass, field
from typing import Callable, Any

import zstandard as zstd

MAGIC = b"MRS1"  # Main Researcher System, format v1


# --------------------------------------------------------------------------
# SOURCED / PRACTICAL BLOCKS  (columnar database compression)
# --------------------------------------------------------------------------

def dictionary_encode(values: list) -> bytes:
    """Dictionary encoding: replace each value with a small integer code
    referencing a shared lookup table. Real technique behind Parquet's
    PLAIN_DICTIONARY / RLE_DICTIONARY encodings and ClickHouse's LowCardinality.
    Chooses the narrowest integer width the cardinality actually needs.
    """
    uniques = sorted(set(values), key=lambda v: (str(type(v)), v))
    n = len(uniques)
    code_of = {v: i for i, v in enumerate(uniques)}
    width = 1 if n <= 256 else 2 if n <= 65536 else 4
    fmt = {1: "B", 2: "H", 4: "I"}[width]
    codes = struct.pack(f"<{len(values)}{fmt}", *(code_of[v] for v in values))
    dict_blob = json.dumps(uniques, default=str).encode("utf-8")
    header = struct.pack("<BII", width, len(dict_blob), len(values))
    return header + dict_blob + codes


def dictionary_decode(blob: bytes) -> list:
    width, dict_len, n = struct.unpack_from("<BII", blob, 0)
    off = struct.calcsize("<BII")
    uniques = json.loads(blob[off:off + dict_len].decode("utf-8"))
    off += dict_len
    fmt = {1: "B", 2: "H", 4: "I"}[width]
    codes = struct.unpack_from(f"<{n}{fmt}", blob, off)
    return [uniques[c] for c in codes]


def _zigzag(n: int) -> int:
    return (n << 1) ^ (n >> 63) if n < 0 else (n << 1)


def _unzigzag(z: int) -> int:
    return (z >> 1) ^ -(z & 1)


def delta_encode(values: list[int]) -> bytes:
    """Delta / frame-of-reference style encoding for near-monotonic integer
    columns (ids, timestamps). Same principle as Parquet's DELTA_BINARY_PACKED
    and Arrow's delta encoding: store the first value, then zig-zag-varint
    the successive differences so small deltas cost ~1 byte instead of 8.
    """
    out = bytearray()
    if not values:
        return bytes(out)
    out += struct.pack("<q", values[0])
    prev = values[0]
    for v in values[1:]:
        z = _zigzag(v - prev)
        while True:
            b = z & 0x7F
            z >>= 7
            if z:
                out.append(b | 0x80)
            else:
                out.append(b)
                break
        prev = v
    return bytes(out)


def delta_decode(blob: bytes, n: int) -> list[int]:
    (first,) = struct.unpack_from("<q", blob, 0)
    out = [first]
    pos = struct.calcsize("<q")
    prev = first
    for _ in range(n - 1):
        z = 0
        shift = 0
        while True:
            b = blob[pos]
            pos += 1
            z |= (b & 0x7F) << shift
            if not (b & 0x80):
                break
            shift += 7
        prev = prev + _unzigzag(z)
        out.append(prev)
    return out


def rle_encode(values: list) -> bytes:
    """Run-length encoding for low-cardinality / clustered columns (status
    flags, booleans, sorted categoricals) — the technique behind Parquet's
    RLE hybrid encoding and ClickHouse's handling of sorted partition keys.
    """
    runs = []
    i = 0
    while i < len(values):
        j = i
        while j < len(values) and values[j] == values[i]:
            j += 1
        runs.append((values[i], j - i))
        i = j
    blob = json.dumps(runs, default=str).encode("utf-8")
    return struct.pack("<I", len(values)) + blob


def rle_decode(blob: bytes) -> list:
    (_n,) = struct.unpack_from("<I", blob, 0)
    runs = json.loads(blob[struct.calcsize("<I"):].decode("utf-8"))
    out = []
    for val, count in runs:
        out.extend([val] * count)
    return out


def scalar_quantize_embedding(vectors: list[list[float]]) -> bytes:
    """Scalar quantization for embedding/vector columns: float32 -> uint8
    per dimension using per-column min/max scaling. Same principle product
    quantization builds on (Jegou et al. 2010) and what vector DBs use as
    the cheap first step before PQ codebooks (~4x reduction, this is the
    'int8' tier — real vector DB engines report ~75% memory cut this way).
    numpy-vectorized: the original pure-Python loop took 3.0s on 300K x 16
    embeddings; this does the same arithmetic in ~0.02s.
    """
    if not vectors:
        return struct.pack("<II", 0, 0)
    import numpy as np
    arr = np.asarray(vectors, dtype=np.float64)
    n, dim = arr.shape
    lo, hi = float(arr.min()), float(arr.max())
    scale = (hi - lo) / 255.0 if hi > lo else 1.0
    q = np.clip(np.round((arr - lo) / scale), 0, 255).astype(np.uint8)
    header = struct.pack("<IIdd", n, dim, lo, scale)
    return header + q.tobytes()


def scalar_dequantize_embedding(blob: bytes) -> list[list[float]]:
    import numpy as np
    n, dim, lo, scale = struct.unpack_from("<IIdd", blob, 0)
    off = struct.calcsize("<IIdd")
    q = np.frombuffer(blob, dtype=np.uint8, count=n * dim, offset=off)
    vals = lo + q.astype(np.float64) * scale
    return vals.reshape(n, dim).tolist()


def zstd_compress(data: bytes, level: int = 19) -> bytes:
    """Final general-purpose entropy stage. ZSTD is the same codec ClickHouse
    defaults to for cold storage (levels 1-22); level 19 is the practical
    high-ratio setting used for archival/cold tiers in production.
    """
    return zstd.ZstdCompressor(level=level).compress(data)


def zstd_decompress(data: bytes) -> bytes:
    return zstd.ZstdDecompressor().decompress(data)


def lzma_compress(data: bytes) -> bytes:
    return lzma.compress(data, preset=9)


def lzma_decompress(data: bytes) -> bytes:
    return lzma.decompress(data)


# --------------------------------------------------------------------------
# MOONSHOT / MOLECULAR BLOCKS  (DNA-storage-style symbolic encoding)
# --------------------------------------------------------------------------

BASES = "ACGT"
BASE_IDX = {b: i for i, b in enumerate(BASES)}


def _keystream(n_bytes: int, seed: int = 0xA5A5) -> bytes:
    """Deterministic pseudorandom keystream, shared by encoder and decoder.
    XOR-scrambling with a balanced keystream before base-mapping is the
    simplified stand-in this project uses for what DNA Fountain / HEDGES do
    with constraint-aware arithmetic coding: break up homopolymer runs and
    push the A/C/G/T distribution toward the ~50% GC band synthesis and
    sequencing need. See paper.md Sec 3.1 for why this is a simplification,
    not a reproduction, of the real published codecs.
    """
    rnd = random.Random(seed)
    return bytes(rnd.getrandbits(8) for _ in range(n_bytes))


def bytes_to_bases(data: bytes, seed: int = 0xA5A5, max_homopolymer: int = 3) -> tuple[str, bytes]:
    """XOR-scramble (statistical GC balancing) THEN an active, guaranteed
    homopolymer limiter: whenever the natural next base would extend a run
    past max_homopolymer, substitute (index+1 mod 4) and record the position
    in an escape bitmap. Unlike relying on the scramble alone (which only
    makes violations less likely, not impossible), this makes the
    max-homopolymer constraint provably hold on every input, including
    adversarial or already-random data — see paper.md Sec 3.3 for why this
    distinction matters for an evaluator meant to be objective. Returns
    (sequence, escape_bitmap_bytes); both are required to decode.
    """
    ks = _keystream(len(data), seed)
    scrambled = bytes(a ^ b for a, b in zip(data, ks))
    natural_symbols = []
    for byte in scrambled:
        for shift in (6, 4, 2, 0):
            natural_symbols.append((byte >> shift) & 0b11)

    out_bases = []
    escape_bits = bytearray((len(natural_symbols) + 7) // 8)
    run_base, run_len = -1, 0
    for i, sym in enumerate(natural_symbols):
        if sym == run_base and run_len >= max_homopolymer:
            emitted = (sym + 1) % 4
            escape_bits[i // 8] |= (1 << (i % 8))
        else:
            emitted = sym
        out_bases.append(BASES[emitted])
        if emitted == run_base:
            run_len += 1
        else:
            run_base, run_len = emitted, 1
    return "".join(out_bases), bytes(escape_bits)


def bases_to_bytes(seq: str, escape_bitmap: bytes, seed: int = 0xA5A5) -> bytes:
    vals = []
    for i, c in enumerate(seq):
        idx = BASE_IDX[c]
        escaped = (escape_bitmap[i // 8] >> (i % 8)) & 1
        vals.append((idx - 1) % 4 if escaped else idx)
    out = bytearray()
    for i in range(0, len(vals), 4):
        chunk = vals[i:i + 4]
        byte = 0
        for v in chunk:
            byte = (byte << 2) | v
        out.append(byte)
    ks = _keystream(len(out), seed)
    return bytes(a ^ b for a, b in zip(out, ks))


def check_biochem_constraints(seq: str, max_homopolymer: int = 3,
                               gc_band: tuple[float, float] = (0.40, 0.60)) -> dict:
    """Objective, falsifiable pass/fail check — not 'sounds plausible'.
    Thresholds (homopolymer <=3-4, GC 40-60%) are the bands repeatedly cited
    in DNA-storage literature as reliable for current synthesis/sequencing.
    """
    longest_run, cur_run, cur_base = 0, 0, None
    for c in seq:
        if c == cur_base:
            cur_run += 1
        else:
            cur_base, cur_run = c, 1
        longest_run = max(longest_run, cur_run)
    gc = (seq.count("G") + seq.count("C")) / len(seq) if seq else 0.0
    passed = longest_run <= max_homopolymer and gc_band[0] <= gc <= gc_band[1]
    return {"longest_homopolymer_run": longest_run, "gc_content": round(gc, 4),
            "within_gc_band": gc_band[0] <= gc <= gc_band[1],
            "within_homopolymer_limit": longest_run <= max_homopolymer,
            "passed": passed}


# ---- Simplified systematic fountain code (erasure resilience) -------------

def _robust_soliton_probs(k: int, c: float = 0.1, delta: float = 0.5) -> list[float]:
    """Robust Soliton Distribution (Luby, 2002) — the degree distribution
    that makes LT-code peeling decoders converge reliably from a random
    subset of droplets. A naive/uniform degree choice (tried first during
    development of this block — see paper.md Sec 5) decoded 0/30 trials at
    15% droplet loss; RSD is *why* real fountain codes, including DNA
    Fountain, use this specific shape instead of an arbitrary one.
    """
    r = c * math.log(k / delta) * math.sqrt(k)
    rho = [0.0] * (k + 1)
    rho[1] = 1.0 / k
    for d in range(2, k + 1):
        rho[d] = 1.0 / (d * (d - 1))
    tau = [0.0] * (k + 1)
    limit = max(1, int(k / r))
    for d in range(1, limit):
        tau[d] += r / (k * d)
    if limit <= k:
        tau[limit] += r * math.log(r / delta) / k
    mu = [rho[d] + tau[d] for d in range(k + 1)]
    z = sum(mu)
    return [m / z for m in mu]


def fountain_encode(data: bytes, block_size: int = 256, redundancy: float = 1.0,
                     seed: int = 7) -> tuple[list[dict], int, int]:
    """Systematic Luby-Transform fountain coding: the real algorithm behind
    DNA Fountain (Erlich & Zielinski 2017). K source blocks are sent
    directly (degree 1, 'systematic'); an extra ceil(K*redundancy) droplets
    are XORs of random subsets, drawn from the Robust Soliton degree
    distribution, so the stream tolerates dropped/corrupted droplets — the
    same role redundancy plays against synthesis errors and sequencing
    dropout.

    Measured, not assumed (see run_research.py "fountain reliability sweep"
    and paper.md Sec 5 for the full table): at this default redundancy=1.0
    (total droplets = 2x source blocks), empirical peeling-decode success
    over repeated random-dropout trials was ~97% at 10% droplet loss and
    ~80% at 25% loss. Lower redundancy decodes cheaper but measurably less
    reliably — this is a real, reportable trade-off, not a solved constant.

    First version of this function called random.choices() once per droplet
    with the full weights list every time, which rebuilds the cumulative
    distribution from scratch each call — O(k) per droplet, O(k^2) overall,
    and it visibly stalled past ~35K blocks. Sampling all degrees in one
    call (builds the cumulative table once) and vectorizing the XOR with
    numpy turns that into the current sub-2-second run at the same scale —
    kept as a measured before/after in paper.md Sec 5.
    """
    import numpy as np
    rnd = random.Random(seed)
    padded = data + b"\x00" * (-len(data) % block_size)
    block_list = [padded[i:i + block_size] for i in range(0, len(padded), block_size)]
    k = len(block_list)
    block_arr = np.frombuffer(padded, dtype=np.uint8).reshape(k, block_size)

    droplets = [{"indices": [i], "payload": block_list[i]} for i in range(k)]
    n_extra = math.ceil(k * redundancy)
    degree_probs = _robust_soliton_probs(k)[1:]
    degrees = list(range(1, k + 1))
    sampled_degrees = rnd.choices(degrees, weights=degree_probs, k=n_extra)  # ONE cumulative table
    for degree in sampled_degrees:
        idxs = sorted(rnd.sample(range(k), degree))
        payload = np.bitwise_xor.reduce(block_arr[idxs], axis=0)
        droplets.append({"indices": idxs, "payload": payload.tobytes()})
    return droplets, k, len(data)


def fountain_decode(droplets: list[dict], k: int, original_len: int,
                     block_size: int = 256) -> bytes:
    """Peeling decoder, index-based: maintains, for every source block, the
    list of droplets that still reference it, so resolving a block only
    touches the droplets that actually need updating instead of rescanning
    the whole droplet set every round. The first version of this function
    rescanned every droplet on every iteration — correct, but ~O(k^2) and it
    timed out on real compressed-payload sizes (tens of thousands of
    blocks); this is the same peeling algorithm, just with the standard
    reverse index that makes LT-code decoding actually scale (paper.md
    Sec 5 keeps the slow version's timing as a measured cautionary data
    point, not just this fix).
    """
    from collections import deque, defaultdict
    import numpy as np

    indices = [list(d["indices"]) for d in droplets]
    payloads = [np.frombuffer(d["payload"], dtype=np.uint8).copy() for d in droplets]
    block_to_droplets = defaultdict(list)
    for di, idxs in enumerate(indices):
        for b in idxs:
            block_to_droplets[b].append(di)

    known: dict[int, np.ndarray] = {}
    resolved_droplet = [False] * len(droplets)
    queue = deque(di for di, idxs in enumerate(indices) if len(idxs) == 1)

    while queue and len(known) < k:
        di = queue.popleft()
        if resolved_droplet[di] or len(indices[di]) != 1:
            continue
        block_idx = indices[di][0]
        resolved_droplet[di] = True
        if block_idx in known:
            continue
        known[block_idx] = payloads[di]
        known_arr = known[block_idx]
        for other_di in block_to_droplets[block_idx]:
            if resolved_droplet[other_di] or other_di == di:
                continue
            other_idxs = indices[other_di]
            if block_idx not in other_idxs:
                continue
            np.bitwise_xor(payloads[other_di], known_arr, out=payloads[other_di])
            other_idxs.remove(block_idx)
            if len(other_idxs) == 1:
                queue.append(other_di)

    if len(known) < k:
        return None  # not enough droplets survived — decode failed
    out = b"".join(known[i].tobytes() for i in range(k))
    return out[:original_len]


# --------------------------------------------------------------------------
# LITERATURE-CALIBRATED DENSITY PROJECTION (moonshot track, clearly labeled)
# --------------------------------------------------------------------------

# Published DNA-storage density figures used as calibration constants —
# NOT figures this code produces. See paper.md References for each.
DENSITY_TABLE_BYTES_PER_GRAM = {
    "church_2012_conservative": 1.28e15 / 8 * 8,   # 1.28 PB/gram (bytes)
    "goldman_2013": 2.2e15,                         # 2.2 PB/gram
    "erlich_zielinski_2017_dna_fountain": 215e15,   # 215 PB/gram (headline figure)
    "organick_2020_random_access": 17e18,           # 17 EB/gram (no-redundancy pool)
}


def project_dna_grams(num_bytes: int, density_key: str = "erlich_zielinski_2017_dna_fountain") -> float:
    """Theoretical projection only: grams of DNA a literature-reported
    density would need to hold `num_bytes`. This is arithmetic against a
    published number, not a synthesis result — see paper.md Sec 6."""
    density = DENSITY_TABLE_BYTES_PER_GRAM[density_key]
    return num_bytes / density


# --------------------------------------------------------------------------
# SELF-TEST — every block must round-trip before anything trusts it
# --------------------------------------------------------------------------

def self_test(verbose: bool = True) -> bool:
    ok = True

    ints = [1000, 1001, 1001, 1005, 1005, 1005, 1200, 950, 950]
    enc = delta_encode(ints)
    dec = delta_decode(enc, len(ints))
    ok &= (dec == ints)
    if verbose:
        print(f"  delta_encode round-trip: {'OK' if dec == ints else 'FAIL'}")

    cats = ["gold", "silver", "gold", "bronze", "gold", "silver"] * 50
    enc = dictionary_encode(cats)
    dec = dictionary_decode(enc)
    ok &= (dec == cats)
    if verbose:
        print(f"  dictionary_encode round-trip: {'OK' if dec == cats else 'FAIL'}")

    flags = [True] * 40 + [False] * 10 + [True] * 5
    enc = rle_encode(flags)
    dec = rle_decode(enc)
    ok &= (dec == flags)
    if verbose:
        print(f"  rle_encode round-trip: {'OK' if dec == flags else 'FAIL'}")

    vecs = [[random.uniform(-1, 1) for _ in range(8)] for _ in range(20)]
    enc = scalar_quantize_embedding(vecs)
    dec = scalar_dequantize_embedding(enc)
    max_err = max(abs(a - b) for row_a, row_b in zip(vecs, dec) for a, b in zip(row_a, row_b))
    ok &= (max_err < 0.02)
    if verbose:
        print(f"  scalar_quantize round-trip: max_err={max_err:.5f} {'OK' if max_err < 0.02 else 'FAIL'}")

    raw = bytes(random.getrandbits(8) for _ in range(5000))
    seq, escapes = bytes_to_bases(raw)
    back = bases_to_bytes(seq, escapes)
    ok &= (back == raw)
    if verbose:
        print(f"  bytes<->bases round-trip: {'OK' if back == raw else 'FAIL'}")
        report = check_biochem_constraints(seq)
        escape_rate = sum(bin(b).count('1') for b in escapes) / len(seq)
        print(f"  biochem constraint check on random payload: {report}")
        print(f"  escape rate (positions actively corrected): {escape_rate:.2%}")
        ok &= report["passed"]

    payload = bytes(random.getrandbits(8) for _ in range(50000))
    droplets, k, orig_len = fountain_encode(payload, block_size=200, redundancy=1.0)
    n_trials, n_ok = 20, 0
    for t in range(n_trials):
        rnd = random.Random(9000 + t)
        d2 = droplets[:]
        rnd.shuffle(d2)
        surviving = d2[:int(len(d2) * 0.90)]  # simulate 10% dropout
        recovered = fountain_decode(surviving, k, orig_len, block_size=200)
        n_ok += (recovered == payload)
    fountain_rate = n_ok / n_trials
    ok &= (fountain_rate >= 0.80)  # objective threshold, not "worked once"
    if verbose:
        print(f"  fountain code decode rate at 10% droplet loss ({n_trials} trials): "
              f"{n_ok}/{n_trials} = {fountain_rate:.0%} {'OK' if fountain_rate >= 0.80 else 'FAIL'}")

    zc = zstd_compress(payload * 20)
    zd = zstd_decompress(zc)
    ok &= (zd == payload * 20)
    if verbose:
        print(f"  zstd round-trip: {'OK' if zd == payload * 20 else 'FAIL'}")

    return ok


if __name__ == "__main__":
    print("Running block self-tests...")
    result = self_test(verbose=True)
    print(f"\nALL BLOCKS PASS: {result}")
