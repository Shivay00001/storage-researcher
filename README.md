# Storage Researcher System

Applies the Main Researcher System (pattern extraction → combinatorial
recombination → dual evaluation → recursive archive) to database
compression and molecular (DNA) storage encoding. Full writeup: `paper.md`.

Real, tested results on real data — not placeholders:
- 300,000-row / 130 MB synthetic database → **18.07x** compression,
  verified byte-exact on every schema column.
- General-purpose CLI (no hardcoded schema) → **10.0x** on a held-out
  50,000-row CSV, verified with a byte-for-byte round-trip diff.
- Compressed output re-encoded into a biochemically-valid DNA base
  sequence (homopolymer ≤3, GC ~50%) with erasure-coded redundancy;
  100% fountain-decode success across repeated 10%-droplet-loss trials.

## Quick start

```bash
pip install zstandard numpy

# Run the block-level self-tests (round-trips every encoder/decoder)
python3 blocks.py

# Compress any CSV — auto-detects the best encoding per column
python3 pipeline.py compress your_data.csv out.mrsc
python3 pipeline.py compress your_data.csv out.mrsc --molecular   # + DNA encoding layer
python3 pipeline.py decompress out.mrsc restored.csv
python3 pipeline.py inspect out.mrsc

# Reproduce the full research run (evolutionary search -> full-scale
# apply -> moonshot layer -> report), phased so each step is checkpointed:
python3 run_research.py gen
python3 run_research.py search
python3 run_research.py finalize
python3 run_research.py moonshot
python3 run_research.py report
```

## Files

| File | What it is |
|---|---|
| `blocks.py` | Pattern library: dictionary/delta/RLE/quantization encoders, ZSTD/LZMA, DNA base-4 encoding + homopolymer/GC constraints + fountain erasure coding. Every function is round-trip tested in `self_test()`. |
| `engine.py` | The Main Researcher System itself for this domain: `Recipe`, crossover/mutation, the two evaluators (`evaluate_sourced`, `evaluate_moonshot`), the evolutionary loop. |
| `generate_dataset.py` | Builds the synthetic 300K-row test database (mixed column types on purpose). |
| `run_research.py` | Phased runner that produced the numbers in `paper.md` Section 4. |
| `pipeline.py` | **The production tool.** General-purpose CSV compressor — measures every applicable encoding per column on data it's never seen and keeps whichever is smallest and verified-lossless. |
| `paper.md` | The research writeup: related work, system design, results, and — importantly — two real bugs this project found and fixed (Section 5), and what the "1 gram" claim does and doesn't mean (Section 6). |

## What "gram-scale" actually means here

The `--molecular` flag produces a real, tested, round-trip-verified DNA
base sequence in software. It does not synthesize physical DNA — that
needs wet-lab equipment this code has no access to. Every gram figure is
arithmetic against a *published* density figure (see `paper.md` §2.2 and
§6), computed from the compressed byte count, not from anything this code
physically produced. Read `paper.md` Section 6 before quoting a gram
number out of context.
