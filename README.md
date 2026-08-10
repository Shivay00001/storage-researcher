<div align="center">
  <img src="assets/hero.png" alt="Storage Researcher Hero" width="100%">

  # Storage Researcher System

  **Database compression system evolving per-column encoding recipes and a molecular (DNA) storage layer.**

  [![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
  [![License](https://img.shields.io/badge/License-Non--Commercial-blue.svg)](LICENSE)
  [![Release](https://img.shields.io/github/v/release/Shivay00001/storage-researcher)](https://github.com/Shivay00001/storage-researcher/releases)
</div>

---

Applies the **Main Researcher System** (pattern extraction → combinatorial recombination → dual evaluation → recursive archive) to structured database compression and molecular (DNA) storage encoding. This repository provides a general-purpose, production-ready CLI for compressing arbitrary CSVs, alongside the full evolutionary research code.

Read the full writeup in [`paper.md`](paper.md) for related work, system design, and results.

## 🚀 Key Results (Real, Tested Data)

- **18.07x Compression**: Tested on a 300,000-row (130 MB) synthetic database. Evolved optimal per-column encodings and verified byte-exact on every schema column.
- **10.0x General-Purpose CLI Compression**: Zero hardcoded schemas. Automatically measures and applies the best encoding per column on an arbitrary held-out 50,000-row CSV.
- **Biochemically-Valid DNA Encoding Layer**: Output is successfully re-encoded into a base sequence (homopolymer ≤3, GC ~50%) featuring erasure-coded redundancy. Demonstrated 100% fountain-decode success across repeated 10%-droplet-loss trials.

## 🛠️ Quick Start

Install required dependencies:

```bash
pip install zstandard numpy
```

### Self-Test & Production CLI

```bash
# Run the block-level self-tests (round-trips every encoder/decoder)
python3 blocks.py

# Compress any CSV — auto-detects the best encoding per column
python3 pipeline.py compress your_data.csv out.mrsc

# Compress with the DNA molecular encoding layer
python3 pipeline.py compress your_data.csv out.mrsc --molecular

# Decompress and verify
python3 pipeline.py decompress out.mrsc restored.csv
python3 pipeline.py inspect out.mrsc
```

### Reproduce the Research Run

Reproduce the full evolutionary search (evolutionary search -> full-scale apply -> moonshot layer -> report), phased so each step is checkpointed:

```bash
python3 run_research.py gen
python3 run_research.py search
python3 run_research.py finalize
python3 run_research.py moonshot
python3 run_research.py report
```

## 📂 Architecture & Files

| File | Description |
|---|---|
| `blocks.py` | **Pattern library:** Dictionary/delta/RLE/quantization encoders, ZSTD/LZMA, DNA base-4 encoding + homopolymer/GC constraints + fountain erasure coding. Every function is round-trip tested in `self_test()`. |
| `engine.py` | **The Main Researcher System:** `Recipe` generation, crossover/mutation, two evaluators (`evaluate_sourced`, `evaluate_moonshot`), and the evolutionary loop. |
| `pipeline.py` | **The production tool:** General-purpose CSV compressor. Measures every applicable encoding per column on unseen data, retaining the most efficient and verified-lossless option. |
| `run_research.py` | **Phased runner:** Executes the evolutionary loop to produce the numbers in `paper.md`. |
| `generate_dataset.py` | **Data generator:** Builds the synthetic 300K-row test database with deliberately mixed column types. |
| `paper.md` | **The research writeup:** Details related work, system design, results, real bugs found and fixed, and clarifies what the "1 gram" claim signifies. |

## 🧬 What "Gram-Scale" Actually Means

The `--molecular` flag produces a real, tested, round-trip-verified DNA base sequence in software. It **does not** synthesize physical DNA — that requires wet-lab equipment beyond the scope of this repository. Every gram figure mentioned is calculated arithmetically against *published* density figures (see `paper.md` §2.2 and §6), based on the compressed byte count.

> [!WARNING]
> Please read `paper.md` Section 6 before quoting any gram figure out of context.

---
<div align="center">
  <i>Built as a practical application of the Main Researcher System.</i>
</div>
