#!/usr/bin/env python3
"""
pipeline.py — the production artifact. Unlike engine.py (which evolves and
compares recipes across a whole dataset for the research run), this is a
general-purpose CLI: point it at any CSV export of a real database and it
measures every applicable encoding for EACH column, keeps whichever is
smallest (verified lossless before it's trusted), and writes a single
.mrsc container. No hardcoded schema — it works on datasets it has never
seen, which is the actual bar for "production," not just the synthetic
dataset the research run used.

Usage:
    python3 pipeline.py compress   input.csv  output.mrsc  [--molecular] [--level 19]
    python3 pipeline.py decompress output.mrsc restored.csv
    python3 pipeline.py inspect    output.mrsc

Container format (.mrsc): MAGIC + json header (per-column strategy chosen,
row count, column order, molecular metadata if used) + zstd/lzma payload.
"""
import argparse
import csv
import json
import struct
import sys
import time
from pathlib import Path

from blocks import (
    dictionary_encode, dictionary_decode, delta_encode, delta_decode,
    rle_encode, rle_decode, zstd_compress, zstd_decompress,
    lzma_compress, lzma_decompress, bytes_to_bases, bases_to_bytes,
    check_biochem_constraints, fountain_encode, fountain_decode,
    project_dna_grams, DENSITY_TABLE_BYTES_PER_GRAM, MAGIC,
)

HEADER_LEN_FMT = "<I"


def _looks_like_int(values: list) -> bool:
    try:
        return all(float(v) == int(float(v)) for v in values if v != "")
    except (ValueError, TypeError):
        return False


def _coerce_column(raw_strings: list) -> tuple[list, str]:
    """CSV gives every cell as a string; recover ints/floats where the whole
    column is numeric so delta/dictionary get a fair shot, same as a real
    ingestion layer would type a column instead of leaving it text."""
    if _looks_like_int(raw_strings):
        return [int(float(v)) for v in raw_strings], "int"
    try:
        return [float(v) for v in raw_strings], "float"
    except (ValueError, TypeError):
        return raw_strings, "str"


def auto_select_strategy(values: list, col_type: str) -> tuple[str, bytes]:
    """Measure every strategy that actually applies to this column's type
    and keep the smallest verified-lossless encoding — no heuristic guess,
    an actual byte-count race, mirroring engine.py's evaluator philosophy
    but applied per-column instead of per-whole-recipe."""
    candidates = {"raw": json.dumps(values, default=str).encode("utf-8")}
    if col_type == "int":
        candidates["delta"] = delta_encode(values)
    n_unique = len(set(values))
    if n_unique <= max(256, len(values) // 4):
        candidates["dictionary"] = dictionary_encode(values)
        candidates["rle"] = rle_encode(values)

    # verify losslessness of every candidate before it's eligible to win
    verified = {}
    for name, blob in candidates.items():
        try:
            restored = _decode_by_name(name, blob, values, col_type)
            if restored == values:
                verified[name] = blob
        except Exception:
            continue
    best = min(verified, key=lambda n: len(verified[n]))
    return best, verified[best]


def _decode_by_name(name: str, blob: bytes, values_for_len: list, col_type: str) -> list:
    if name == "delta":
        return delta_decode(blob, len(values_for_len))
    if name == "dictionary":
        return dictionary_decode(blob)
    if name == "rle":
        return rle_decode(blob)
    return json.loads(blob.decode("utf-8"))


def compress_csv(input_path: str, output_path: str, molecular: bool = False,
                  zstd_level: int = 19) -> dict:
    t0 = time.time()
    with open(input_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    n_rows = len(rows)
    columns = {name: [row[i] for row in rows] for i, name in enumerate(header)}

    col_meta = {}
    parts = []
    original_size = sum(len(json.dumps(v)) for v in columns.values())
    for name in header:
        raw_strings = columns[name]
        values, col_type = _coerce_column(raw_strings)
        strategy, blob = auto_select_strategy(values, col_type)
        col_meta[name] = {"strategy": strategy, "type": col_type, "n": len(values)}
        parts.append(struct.pack(HEADER_LEN_FMT, len(blob)) + blob)

    blob = b"".join(parts)
    payload = zstd_compress(blob, level=zstd_level)
    compressed_bytes_before_molecular = len(payload)

    molecular_meta = None
    if molecular:
        seq, escapes = bytes_to_bases(payload)
        constraints = check_biochem_constraints(seq)
        molecular_meta = {
            "sequence_length_bases": len(seq),
            "escape_bitmap_b64": escapes.hex(),
            "escape_bitmap_bytes": len(escapes),
            "constraints": constraints,
            "note": ("This ASCII base-sequence file is a *simulation artifact* for "
                     "inspection/round-trip testing only — it is larger on disk than "
                     "the compressed bytes it encodes (1 byte per base vs 2 bits of "
                     "real information per base), because a real deployment would "
                     "synthesize this sequence into actual DNA molecules, not write "
                     "more bytes to a hard drive. The meaningful number is "
                     "projected_grams below, computed from the pre-sequence "
                     "compressed byte count, not from this file's size."),
            "projected_grams": {k: project_dna_grams(len(payload), k)
                                 for k in DENSITY_TABLE_BYTES_PER_GRAM},
        }
        payload = seq.encode("ascii")  # store the base sequence itself when --molecular

    header_json = json.dumps({"columns": header, "col_meta": col_meta, "n_rows": n_rows,
                               "codec": "zstd", "zstd_level": zstd_level,
                               "molecular": molecular_meta is not None,
                               "molecular_meta": molecular_meta}).encode("utf-8")

    with open(output_path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack(HEADER_LEN_FMT, len(header_json)))
        f.write(header_json)
        f.write(payload)

    compressed_size = Path(output_path).stat().st_size
    stats = {"rows": n_rows, "columns": len(header), "original_bytes": original_size,
             "compressed_bytes_before_molecular_encoding": compressed_bytes_before_molecular,
             "ratio": original_size / compressed_bytes_before_molecular,
             "per_column_strategy": {k: v["strategy"] for k, v in col_meta.items()},
             "elapsed_seconds": round(time.time() - t0, 2)}
    if molecular_meta:
        stats["on_disk_simulation_file_bytes"] = compressed_size
        stats["molecular"] = {k: v for k, v in molecular_meta.items() if k != "escape_bitmap_b64"}
    return stats


def decompress_csv(input_path: str, output_path: str) -> dict:
    t0 = time.time()
    with open(input_path, "rb") as f:
        magic = f.read(len(MAGIC))
        if magic != MAGIC:
            raise ValueError("Not a valid .mrsc file")
        (hlen,) = struct.unpack(HEADER_LEN_FMT, f.read(4))
        header = json.loads(f.read(hlen).decode("utf-8"))
        payload = f.read()

    if header["molecular"]:
        seq = payload.decode("ascii")
        escapes = bytes.fromhex(header["molecular_meta"]["escape_bitmap_b64"])
        payload = bases_to_bytes(seq, escapes)
    blob = zstd_decompress(payload)

    pos = 0
    columns = {}
    for name in header["columns"]:
        meta = header["col_meta"][name]
        (length,) = struct.unpack_from(HEADER_LEN_FMT, blob, pos)
        pos += 4
        enc = blob[pos:pos + length]
        pos += length
        columns[name] = _decode_by_name(meta["strategy"], enc, [None] * meta["n"], meta["type"])

    n_rows = header["n_rows"]
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header["columns"])
        for i in range(n_rows):
            writer.writerow([columns[name][i] for name in header["columns"]])
    return {"rows": n_rows, "elapsed_seconds": round(time.time() - t0, 2)}


def inspect(path: str) -> dict:
    with open(path, "rb") as f:
        magic = f.read(len(MAGIC))
        if magic != MAGIC:
            raise ValueError("Not a valid .mrsc file")
        (hlen,) = struct.unpack(HEADER_LEN_FMT, f.read(4))
        header = json.loads(f.read(hlen).decode("utf-8"))
    header["file_size_bytes"] = Path(path).stat().st_size
    return header


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("compress")
    c.add_argument("input_csv")
    c.add_argument("output_mrsc")
    c.add_argument("--molecular", action="store_true",
                    help="also encode the compressed payload as a DNA base-4 sequence")
    c.add_argument("--level", type=int, default=19, help="zstd level (1-22)")

    d = sub.add_parser("decompress")
    d.add_argument("input_mrsc")
    d.add_argument("output_csv")

    i = sub.add_parser("inspect")
    i.add_argument("mrsc_file")

    args = ap.parse_args()
    if args.cmd == "compress":
        stats = compress_csv(args.input_csv, args.output_mrsc, args.molecular, args.level)
        print(json.dumps(stats, indent=2, default=str))
    elif args.cmd == "decompress":
        stats = decompress_csv(args.input_mrsc, args.output_csv)
        print(json.dumps(stats, indent=2, default=str))
    elif args.cmd == "inspect":
        print(json.dumps(inspect(args.mrsc_file), indent=2, default=str))


if __name__ == "__main__":
    main()
