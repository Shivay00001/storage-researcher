"""
engine.py — the Main Researcher System's four components (Pattern Extraction,
Combinatorial Engine, Dual Evaluation Tracks, Recursive Archive — see
paper.md Sec 3), instantiated for the storage/compression domain instead of
the agentic-AI domain the original main_researcher_engine.py targeted.

A Recipe is a per-column encoding strategy (which block each column uses)
plus a final codec choice and a moonshot on/off flag. Recipes are evaluated
against a REAL dataset (generate_dataset.py) with a REAL, executed,
round-trip-verified compression ratio — never a placeholder score.
"""
from __future__ import annotations
import json
import random
import struct
from dataclasses import dataclass, field

from blocks import (
    dictionary_encode, dictionary_decode,
    delta_encode, delta_decode,
    rle_encode, rle_decode,
    scalar_quantize_embedding, scalar_dequantize_embedding,
    zstd_compress, zstd_decompress,
    lzma_compress, lzma_decompress,
    bytes_to_bases, bases_to_bytes,
    check_biochem_constraints,
    fountain_encode, fountain_decode,
    project_dna_grams, DENSITY_TABLE_BYTES_PER_GRAM,
)

# --------------------------------------------------------------------------
# 3.1 PATTERN EXTRACTION — the building-block library for THIS domain
# --------------------------------------------------------------------------
# Every option here is a real, cited technique (see blocks.py docstrings /
# paper.md Sec 3.1), not a placeholder. "raw" = no column-aware structuring,
# the honest baseline every other option has to actually beat.
COLUMN_STRATEGIES = {
    "id": ["delta", "dictionary", "raw"],
    "ts": ["delta", "dictionary", "raw"],
    "category": ["dictionary", "rle", "raw"],
    "status": ["rle", "dictionary", "raw"],
    "description": ["dictionary", "raw"],
    "value": ["raw"],
    "embedding": ["scalar_quantize", "raw"],
}
CODEC_OPTIONS = ["zstd", "lzma"]


@dataclass
class Recipe:
    column_strategy: dict
    codec: str
    generation: int = 0
    parents: tuple = ()

    def label(self) -> str:
        cs = ",".join(f"{k}:{v}" for k, v in self.column_strategy.items())
        return f"[{cs}]+{self.codec}"


def seed_population() -> list:
    """A handful of hand-picked starting points, mirroring how the original
    engine seeded generation 0 — including a deliberately weak baseline so
    the archive has something real to improve on, not just strong recipes."""
    return [
        Recipe({c: "raw" for c in COLUMN_STRATEGIES}, "zstd"),
        Recipe({c: "raw" for c in COLUMN_STRATEGIES}, "lzma"),
        Recipe({"id": "delta", "ts": "delta", "category": "dictionary",
                "status": "dictionary", "description": "raw", "value": "raw",
                "embedding": "raw"}, "zstd"),
        Recipe({"id": "delta", "ts": "delta", "category": "dictionary",
                "status": "rle", "description": "dictionary", "value": "raw",
                "embedding": "scalar_quantize"}, "zstd"),
    ]


# --------------------------------------------------------------------------
# Encode / decode a full dataset under a given recipe (needed both to
# compress for real AND to verify the round-trip is lossless)
# --------------------------------------------------------------------------

def _encode_column(name: str, values: list, strategy: str) -> bytes:
    if name == "embedding" and strategy == "scalar_quantize":
        return scalar_quantize_embedding(values)
    if strategy == "delta":
        return delta_encode(values)
    if strategy == "dictionary":
        return dictionary_encode(values)
    if strategy == "rle":
        return rle_encode(values)
    return json.dumps(values, default=str).encode("utf-8")  # raw


def _decode_column(name: str, blob: bytes, strategy: str, n: int) -> list:
    if name == "embedding" and strategy == "scalar_quantize":
        return scalar_dequantize_embedding(blob)
    if strategy == "delta":
        return delta_decode(blob, n)
    if strategy == "dictionary":
        return dictionary_decode(blob)
    if strategy == "rle":
        return rle_decode(blob)
    return json.loads(blob.decode("utf-8"))  # raw


def apply_recipe(recipe: Recipe, dataset: dict, zstd_level: int = 9) -> bytes:
    """Column-encode everything per the recipe, concatenate with
    length-prefixes, then apply the final codec. Returns compressed bytes.
    zstd_level is deliberately a parameter, not hardcoded to the highest
    setting: Section 5 measures a ~24x slowdown between level 9 and level 19
    on this blob for a ~2% ratio gain, so search uses a fast level to
    evaluate many candidates and only the final winner gets recompressed at
    the high-ratio level for the number actually reported."""
    parts = []
    for col, values in dataset.items():
        enc = _encode_column(col, values, recipe.column_strategy[col])
        parts.append(struct.pack("<I", len(enc)) + enc)
    blob = b"".join(parts)
    if recipe.codec == "zstd":
        return zstd_compress(blob, level=zstd_level)
    return lzma_compress(blob)


def invert_recipe(recipe: Recipe, compressed: bytes, dataset_shape: dict) -> dict:
    """Full inverse of apply_recipe — used ONLY to verify losslessness, the
    same way a real production system would run a restore-and-diff check
    before trusting a backup, not just assume compression is lossless."""
    blob = zstd_decompress(compressed) if recipe.codec == "zstd" else lzma_decompress(compressed)
    out = {}
    pos = 0
    for col, n in dataset_shape.items():
        (length,) = struct.unpack_from("<I", blob, pos)
        pos += 4
        enc = blob[pos:pos + length]
        pos += length
        out[col] = _decode_column(col, enc, recipe.column_strategy[col], n)
    return out


# --------------------------------------------------------------------------
# 3.2 COMBINATORIAL / EVOLUTIONARY ENGINE
# --------------------------------------------------------------------------

def crossover(a: Recipe, b: Recipe, rnd: random.Random, generation: int) -> Recipe:
    cs = {col: (a.column_strategy[col] if rnd.random() < 0.5 else b.column_strategy[col])
          for col in COLUMN_STRATEGIES}
    codec = a.codec if rnd.random() < 0.5 else b.codec
    return Recipe(cs, codec, generation=generation, parents=(a.label(), b.label()))


def mutate(r: Recipe, rnd: random.Random, generation: int) -> Recipe:
    cs = dict(r.column_strategy)
    codec = r.codec
    if rnd.random() < 0.75:
        col = rnd.choice(list(COLUMN_STRATEGIES))
        cs[col] = rnd.choice(COLUMN_STRATEGIES[col])
    else:
        codec = rnd.choice(CODEC_OPTIONS)
    return Recipe(cs, codec, generation=generation, parents=(r.label(),))


# --------------------------------------------------------------------------
# 3.3 DUAL EVALUATION TRACKS
# --------------------------------------------------------------------------

def evaluate_sourced(recipe: Recipe, dataset: dict, original_size: int = None,
                      zstd_level: int = 9) -> dict:
    """Objective, automatic: real compression ratio on real data.

    Schema fields (id/ts/category/status/description/value) are held to an
    exact round-trip — DISQUALIFIED (score=0) if any of them come back
    altered, the same way a real system should never silently corrupt a
    record's id or status. The embedding column is graded differently ONLY
    when the recipe explicitly chose "scalar_quantize": that is a disclosed,
    intentional lossy channel (this is how production vector databases
    actually operate — see paper.md Sec 2), so it is measured and reported
    as a reconstruction error, not silently passed as bit-exact or silently
    disqualified for being approximate.

    original_size is accepted as a parameter rather than recomputed here:
    it doesn't depend on the recipe, and re-serializing a large dataset to
    JSON on every candidate evaluation was the single biggest cost in the
    search loop (measured at 6.6s on 300K rows — see paper.md Sec 5).
    """
    shape = {col: len(vals) for col, vals in dataset.items()}
    if original_size is None:
        original_size = len(json.dumps(dataset, default=str).encode("utf-8"))
    compressed = apply_recipe(recipe, dataset, zstd_level=zstd_level)
    restored = invert_recipe(recipe, compressed, shape)

    schema_cols = [c for c in dataset if c != "embedding"]
    exact_lossless = all(dataset[c] == restored[c] for c in schema_cols)

    embedding_error = None
    if recipe.column_strategy.get("embedding") == "scalar_quantize":
        orig, rest = dataset["embedding"], restored["embedding"]
        errs = [abs(x - y) for ra, rb in zip(orig, rest) for x, y in zip(ra, rb)]
        embedding_error = {"mean_abs_error": sum(errs) / len(errs), "max_abs_error": max(errs)}
    else:
        exact_lossless = exact_lossless and (dataset["embedding"] == restored["embedding"])

    ratio = (original_size / len(compressed)) if exact_lossless else 0.0
    return {"lossless": exact_lossless, "original_bytes": original_size,
            "compressed_bytes": len(compressed), "ratio": ratio,
            "embedding_error": embedding_error, "compressed_blob": compressed}


def evaluate_moonshot(compressed_blob: bytes, n_trials: int = 8,
                       loss_rate: float = 0.10) -> dict:
    """The molecular/moonshot layer as an always-on final stage on top of
    whatever compressed bytes the sourced track already produced — it has no
    dependency on which column strategy was used upstream, so it isn't
    something worth evolving a per-recipe flag for; every candidate's
    compressed output could be handed to this stage. Runs the ACTUAL
    molecular blocks and reports measured outcomes — no step here is
    asserted without being executed. See paper.md Sec 3.3 / Sec 5 for why
    this, not a plausibility judgment, is what 'moonshot track' has to mean
    in practice.
    """
    seq, escapes = bytes_to_bases(compressed_blob)
    roundtrip_ok = bases_to_bytes(seq, escapes) == compressed_blob
    constraints = check_biochem_constraints(seq)

    droplets, k, orig_len = fountain_encode(compressed_blob, block_size=200, redundancy=1.0)
    successes = 0
    rnd = random.Random(2026)
    for t in range(n_trials):
        d2 = droplets[:]
        rnd.shuffle(d2)
        surviving = d2[:int(len(d2) * (1 - loss_rate))]
        rec = fountain_decode(surviving, k, orig_len, block_size=200)
        successes += (rec == compressed_blob)
    fountain_reliability = successes / n_trials

    grams = {key: project_dna_grams(len(compressed_blob), key)
             for key in DENSITY_TABLE_BYTES_PER_GRAM}

    passed = roundtrip_ok and constraints["passed"] and fountain_reliability >= 0.75
    return {"roundtrip_ok": roundtrip_ok, "constraints": constraints,
            "fountain_reliability": fountain_reliability,
            "projected_grams": grams, "passed": passed}


# --------------------------------------------------------------------------
# 3.4 RECURSIVE ARCHIVE + MAIN EVOLUTIONARY LOOP
# --------------------------------------------------------------------------

def run_evolution(dataset: dict, generations: int = 4, population_size: int = 6,
                   seed: int = 11, verbose: bool = True,
                   search_zstd_level: int = 9, final_zstd_level: int = 19) -> dict:
    rnd = random.Random(seed)
    original_size = len(json.dumps(dataset, default=str).encode("utf-8"))
    archive: list[tuple[Recipe, dict]] = []  # (recipe, sourced_eval) sorted by ratio desc
    population = seed_population()

    history = []
    for gen in range(generations):
        scored = []
        for r in population:
            ev = evaluate_sourced(r, dataset, original_size=original_size,
                                   zstd_level=search_zstd_level)
            scored.append((r, ev))
        scored.sort(key=lambda t: t[1]["ratio"], reverse=True)
        archive.extend(scored)
        archive.sort(key=lambda t: t[1]["ratio"], reverse=True)
        archive[:] = archive[:8]  # keep top-8 across all generations so far

        best_r, best_ev = archive[0]
        history.append({"generation": gen, "best_ratio": best_ev["ratio"],
                         "best_recipe": best_r.label()})
        if verbose:
            print(f"  gen {gen}: best ratio so far = {best_ev['ratio']:.3f}x  "
                  f"({best_r.label()})")

        # breed next generation from the archive
        next_pop = []
        parents = [r for r, _ in archive[:4]]
        while len(next_pop) < population_size and len(parents) >= 2:
            a, b = rnd.sample(parents, 2)
            child = crossover(a, b, rnd, gen + 1)
            if rnd.random() < 0.5:
                child = mutate(child, rnd, gen + 1)
            next_pop.append(child)
        while len(next_pop) < population_size:
            next_pop.append(mutate(rnd.choice(parents), rnd, gen + 1))
        population = next_pop

    winner, _ = archive[0]
    # the number that actually matters gets the high-ratio, slow codec level,
    # run once, on the recipe search already converged on with the fast level
    winner_eval = evaluate_sourced(winner, dataset, original_size=original_size,
                                    zstd_level=final_zstd_level)
    moonshot_eval = evaluate_moonshot(winner_eval["compressed_blob"])

    return {"history": history, "archive": archive, "winner": winner,
            "winner_eval": winner_eval, "moonshot_eval": moonshot_eval}


if __name__ == "__main__":
    import generate_dataset
    print("Generating test dataset (5,000 rows)...")
    ds = generate_dataset.generate(5_000)
    print("Running evolutionary search...")
    result = run_evolution(ds, generations=4, population_size=6)
    print("\nWinning recipe:", result["winner"].label())
    print("Sourced eval:", {k: v for k, v in result["winner_eval"].items() if k != "compressed_blob"})
    print("\nMoonshot eval (applied to winner's compressed bytes):")
    me = result["moonshot_eval"]
    for k, v in me.items():
        print(f"  {k}: {v}")
