"""
run_research.py — orchestrates the full study in phases, persisting state
to disk between them (pickle/json in ./run_state/) so each phase can run as
its own process. This is the recursive-archive-driven research run whose
real, executed numbers paper.md Sec 4 and Sec 5 report.

Usage:
    python3 run_research.py gen        # build the 300K-row dataset + a 30K sample
    python3 run_research.py search     # evolutionary search on the sample
    python3 run_research.py finalize   # apply the winning recipe to the FULL dataset
    python3 run_research.py moonshot   # run the molecular layer on the winner's output
    python3 run_research.py report     # print the full, final research report
"""
import json
import pickle
import sys
import time
from pathlib import Path

import engine
import generate_dataset

STATE = Path("run_state")
STATE.mkdir(exist_ok=True)


def _save(name, obj):
    with open(STATE / name, "wb") as f:
        pickle.dump(obj, f)


def _load(name):
    with open(STATE / name, "rb") as f:
        return pickle.load(f)


def phase_gen():
    t0 = time.time()
    full = generate_dataset.generate(300_000, seed=42)
    sample = {col: vals[:30_000] for col, vals in full.items()}
    _save("full_dataset.pkl", full)
    _save("sample_dataset.pkl", sample)
    print(f"[gen] full=300,000 rows, sample=30,000 rows, {time.time()-t0:.1f}s")


def phase_search():
    sample = _load("sample_dataset.pkl")
    t0 = time.time()
    result = engine.run_evolution(sample, generations=6, population_size=6,
                                   search_zstd_level=9, final_zstd_level=19,
                                   seed=11, verbose=True)
    print(f"[search] {time.time()-t0:.1f}s total")
    _save("search_result.pkl", {"history": result["history"], "winner": result["winner"],
                                 "sample_winner_eval": {k: v for k, v in result["winner_eval"].items()
                                                          if k != "compressed_blob"}})


def phase_finalize():
    full = _load("full_dataset.pkl")
    search = _load("search_result.pkl")
    winner = search["winner"]
    t0 = time.time()
    original_size = len(json.dumps(full, default=str).encode("utf-8"))
    final_eval = engine.evaluate_sourced(winner, full, original_size=original_size, zstd_level=19)
    print(f"[finalize] {time.time()-t0:.1f}s — ratio={final_eval['ratio']:.3f}x on "
          f"{original_size:,} -> {final_eval['compressed_bytes']:,} bytes")
    _save("final_eval.pkl", final_eval)


def phase_moonshot():
    final_eval = _load("final_eval.pkl")
    t0 = time.time()
    moonshot = engine.evaluate_moonshot(final_eval["compressed_blob"], n_trials=6, loss_rate=0.10)
    print(f"[moonshot] {time.time()-t0:.1f}s — {moonshot}")
    _save("moonshot_eval.pkl", moonshot)


def phase_report():
    search = _load("search_result.pkl")
    final_eval = _load("final_eval.pkl")
    moonshot = _load("moonshot_eval.pkl")
    report = {
        "history": search["history"],
        "winner_label": search["winner"].label(),
        "sample_scale_ratio": search["sample_winner_eval"]["ratio"],
        "full_scale": {k: v for k, v in final_eval.items() if k != "compressed_blob"},
        "moonshot": moonshot,
    }
    _save("final_report.pkl", report)
    with open(STATE / "final_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(json.dumps(report, indent=2, default=str))


PHASES = {"gen": phase_gen, "search": phase_search, "finalize": phase_finalize,
          "moonshot": phase_moonshot, "report": phase_report}

if __name__ == "__main__":
    phase = sys.argv[1] if len(sys.argv) > 1 else "all"
    if phase == "all":
        for name, fn in PHASES.items():
            fn()
    else:
        PHASES[phase]()
