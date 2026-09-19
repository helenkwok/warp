#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""test_engram_rows.py — `convert.py --engram-rows` must write a slice of an
Engram table that is byte-identical to the same rows of the whole table.

That is the property a distributed conversion stands on: DeepSeek-V4.1's two
tables are 98 GB apiece as published and do not fit one small disk, so each
machine takes a row range and the parts are concatenated in row order. If a
slice differed from the whole by so much as a chunk-boundary effect, the
gathered container would be silently wrong.

Real torch and the real quantizer, on a synthetic checkpoint: a table small
enough to build in a second, laid out the way the release lays it out
(F8_E4M3 payload, F8_E8M0 scales, DeepSeek's `.weight`/`.scale` spelling),
written as a safetensors shard by hand so the test needs no other library.
Skipped, not failed, where torch is missing.

  python3 tests/test_engram_rows.py
"""
import json
import os
import struct
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONVERT = os.path.join(REPO, "tools", "convert.py")

try:
    import torch  # noqa: F401
except ImportError:
    print("SKIP test_engram_rows: torch is not installed")
    sys.exit(0)

ROWS, DIM, LAYER = 1000, 64, 1
FAILS = []


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'} {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILS.append(name)


def build_checkpoint(d):
    """A one-shard checkpoint holding just the Engram table for layer 1."""
    # Bytes below 0x78 are finite e4m3 values in both signs of the low half;
    # NaN would only make the comparison depend on NaN handling, not test it.
    payload = bytes((i * 37 + 11) % 0x78 for i in range(ROWS * DIM))
    scale = bytes(118 + (i * 5) % 12 for i in range(ROWS * (DIM // 32)))
    a = f"layers.{LAYER}.engram.embed.weight"
    b = f"layers.{LAYER}.engram.embed.scale"
    hdr = {a: {"dtype": "F8_E4M3", "shape": [ROWS, DIM],
               "data_offsets": [0, len(payload)]},
           b: {"dtype": "F8_E8M0", "shape": [ROWS, DIM // 32],
               "data_offsets": [len(payload), len(payload) + len(scale)]}}
    h = json.dumps(hdr).encode()
    h += b" " * (-len(h) % 8)
    with open(os.path.join(d, "shard.safetensors"), "wb") as f:
        f.write(struct.pack("<Q", len(h)) + h + payload + scale)
    json.dump({"weight_map": {a: "shard.safetensors", b: "shard.safetensors"}},
              open(os.path.join(d, "model.safetensors.index.json"), "w"))
    json.dump({"engram_layer_ids": [LAYER],
               "architectures": ["DeepseekV41ForCausalLM"]},
              open(os.path.join(d, "config.json"), "w"))


def run(src, out, *extra):
    return subprocess.run([sys.executable, CONVERT, "--src", src, "--out", out,
                           "--engram-layer", str(LAYER), *extra],
                          capture_output=True, text=True)


def part(out, lo, hi):
    return os.path.join(out, f"engram-L{LAYER}.rows-{lo:010d}-{hi:010d}.part")


def main():
    # Import in-process for the whole-table reference, so it is the same
    # write_engram a full conversion calls.
    sys.path.insert(0, os.path.join(REPO, "tools"))
    import convert
    import mxfp4

    with tempfile.TemporaryDirectory() as tmp:
        src, whole, parts = (os.path.join(tmp, n) for n in ("src", "whole", "parts"))
        for d in (src, whole, parts):
            os.makedirs(d)
        build_checkpoint(src)

        st = mxfp4.ST(src)
        # chunk_rows=64 so the slices below straddle chunk boundaries of the
        # whole write in every possible way.
        convert.write_engram(st, whole, f"layers.{LAYER}.engram.embed.weight",
                             LAYER, chunk_rows=64)
        ref = open(os.path.join(whole, f"engram-L{LAYER}.bin"), "rb").read()
        row_bytes = DIM * 4 // 8 + (DIM // 32) * 2
        check("reference is rows x row_bytes", len(ref) == ROWS * row_bytes,
              f"{len(ref)} vs {ROWS * row_bytes}")

        # Uneven ranges, none aligned to the 64-row chunk of the reference or
        # to the 2^20 default of the CLI.
        cuts = [0, 1, 137, 138, 500, 777, ROWS]
        for lo, hi in zip(cuts, cuts[1:]):
            r = run(src, parts, "--engram-rows", f"{lo}:{hi}")
            check(f"rows {lo}:{hi} exits 0", r.returncode == 0, r.stderr.strip()[-120:])
            got = open(part(parts, lo, hi), "rb").read() if os.path.exists(part(parts, lo, hi)) else b""
            check(f"rows {lo}:{hi} equal the same rows of the whole table",
                  got == ref[lo * row_bytes: hi * row_bytes])

        joined = b"".join(open(part(parts, lo, hi), "rb").read()
                          for lo, hi in zip(cuts, cuts[1:]))
        check("parts in row order concatenate to the whole table", joined == ref)

        names = sorted(n for n in os.listdir(parts) if n.startswith("engram-L"))
        check("a lexical sort of the part names is a row sort",
              names == [os.path.basename(part(parts, lo, hi))
                        for lo, hi in zip(cuts, cuts[1:])])
        check("nothing but Engram parts was written",
              all(n.endswith(".part") for n in os.listdir(parts)), str(os.listdir(parts)))

        # resume: a finished slice is kept, and says so
        r = run(src, parts, "--engram-rows", "137:138")
        check("a finished slice is kept", "already written" in r.stdout, r.stdout.strip()[-80:])

        # refusals
        for label, args in (("rows past the end", ("--engram-rows", f"0:{ROWS + 1}")),
                            ("empty range", ("--engram-rows", "5:5")),
                            ("no such Engram layer", ("--engram-rows", "0:10", "--engram-layer", "9"))):
            r = subprocess.run([sys.executable, CONVERT, "--src", src, "--out", parts,
                                "--engram-layer", str(LAYER), *args],
                               capture_output=True, text=True)
            check(f"refuses: {label}", r.returncode != 0)
        r = subprocess.run([sys.executable, CONVERT, "--src", src, "--out", parts,
                            "--engram-rows", "0:10"], capture_output=True, text=True)
        check("refuses --engram-rows without --engram-layer", r.returncode != 0)

    print(f"\n{'PASS' if not FAILS else 'FAIL'}: {len(FAILS)} failed")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
