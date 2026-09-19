#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""ds41_gather.py — assemble a DeepSeek-V4.1 container converted in pieces.

The pieces, each made on a machine that never saw the others:

  layers/experts-L<N>.bin, codebooks-L<N>.bin   convert.py --layers N
                                                  --skip-trunk --codebook-base layer
  trunk/trunk.bin, manifest.json, engram.json,
        engram-tokmap.bin, tokenizer.model ...   tools/trunk_remote.py
  engram/engram-L<N>.rows-<LO>-<HI>.part         tools/engram_remote.py

Put them in one directory (--dir) and run this. It

  1. concatenates each Engram table's parts in row order, and refuses a gap,
     an overlap, a part of the wrong size, or a table that is not whole;
  2. runs convert.py --skip-trunk --codebook-base layer to merge the banks and
     codebooks into the manifest, which also records the Engram tables;
  3. checks every file the engine will open is there at the size the manifest
     says.

Step 2 is handed a checkpoint holding the small files and the *headers* of
the Engram shards, and nothing it could convert: convert.py keeps a
correctly sized engram-L<N>.bin without reading it, but it would write one
from a sparse shard's zeros if the file were missing. An empty
.download-state makes every shard read as not-downloaded, so it cannot.

Needs `pip install torch tokenizers numpy` (torch for convert.py, tokenizers
and numpy for the Engram index, which convert.py rebuilds at the end).

  python3 tools/ds41_gather.py --dir /data/ds41.waste \\
      --repo deepseek-ai/DeepSeek-V4.1-Flash --scratch /data/ds41small
"""

import argparse
import glob
import json
import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

PART = re.compile(r"^engram-L(\d+)\.rows-(\d{10})-(\d{10})\.part$")


def find_parts(d):
    """{layer: [(lo, hi, path)]} sorted by row."""
    out = {}
    for path in glob.glob(os.path.join(d, "engram-L*.rows-*.part")):
        m = PART.match(os.path.basename(path))
        if m:
            out.setdefault(int(m.group(1)), []).append(
                (int(m.group(2)), int(m.group(3)), path))
    return {L: sorted(p) for L, p in out.items()}


def gather_table(d, layer, parts, total_rows, row_bytes, keep=False):
    """Concatenate one table's parts into engram-L<layer>.bin.

    Every property a wrong table would have is refused before a byte of the
    result exists: rows must run 0..total_rows with no gap and no overlap,
    and each part must be exactly its rows long. Written to a .tmp and
    renamed, so a killed run leaves no table that looks finished.
    """
    at = 0
    for lo, hi, path in parts:
        if lo != at:
            kind = "gap" if lo > at else "overlap"
            raise SystemExit(f"engram L{layer}: {kind} at row {at}: the next "
                             f"part starts at {lo}")
        want = (hi - lo) * row_bytes
        if os.path.getsize(path) != want:
            raise SystemExit(f"engram L{layer}: {os.path.basename(path)} is "
                             f"{os.path.getsize(path)} bytes, its rows are {want}")
        at = hi
    if at != total_rows:
        raise SystemExit(f"engram L{layer}: parts end at row {at}, the table "
                         f"has {total_rows}")
    dest = os.path.join(d, f"engram-L{layer}.bin")
    tmp = dest + ".tmp"
    with open(tmp, "wb") as out:
        for _, _, path in parts:
            with open(path, "rb") as f:
                shutil.copyfileobj(f, out, 16 << 20)
            if not keep:
                os.remove(path)            # a table is never held twice on disk
        out.flush()
        os.fsync(out.fileno())
    os.replace(tmp, dest)
    return dest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="the container directory")
    ap.add_argument("--repo", default="deepseek-ai/DeepSeek-V4.1-Flash")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--scratch", required=True,
                    help="where the small files and shard headers go")
    ap.add_argument("--keep-parts", action="store_true",
                    help="do not delete parts once they are concatenated "
                         "(doubles the disk a table takes)")
    ap.add_argument("rest", nargs=argparse.REMAINDER,
                    help="after --, arguments passed through to convert.py")
    args = ap.parse_args()

    import engram_remote

    os.makedirs(args.scratch, exist_ok=True)
    raw_url = engram_remote.fetch_small_files(args.repo, args.revision, args.scratch)
    wm = json.load(open(os.path.join(args.scratch, "model.safetensors.index.json")))["weight_map"]
    cfg = json.load(open(os.path.join(args.scratch, "config.json")))
    cfg = cfg.get("text_config") or cfg
    layers = cfg.get("engram_layer_ids") or []
    if not layers:
        print(f"{args.repo} has no Engram layers; nothing here applies", file=sys.stderr)
        return 1

    # Headers only, and an empty .download-state so nothing reads as present.
    for L in layers:
        name = f"layers.{L}.engram.embed.weight"
        engram_remote.lay_out(raw_url, args.scratch, wm[name])
    open(os.path.join(args.scratch, ".download-state"), "w").close()

    import mxfp4
    st = mxfp4.ST(args.scratch)

    # ---- 1. the tables --------------------------------------------------
    parts = find_parts(args.dir)
    for L in layers:
        dest = os.path.join(args.dir, f"engram-L{L}.bin")
        rows, dim = st.shape(f"layers.{L}.engram.embed.weight")
        row_bytes = dim * 4 // 8 + (dim // 32) * 2
        if L in parts:
            gather_table(args.dir, L, parts[L], rows, row_bytes, args.keep_parts)
            print(f"engram L{L}: {len(parts[L])} parts -> {os.path.getsize(dest) / 1e9:.2f} GB")
        elif not (os.path.exists(dest) and os.path.getsize(dest) == rows * row_bytes):
            print(f"engram L{L}: no parts and no whole table in {args.dir}",
                  file=sys.stderr)
            return 1

    # ---- 2. banks and codebooks into the manifest ------------------------
    import convert
    rest = args.rest[1:] if args.rest[:1] == ["--"] else args.rest
    sys.argv = ["convert.py", "--src", args.scratch, "--out", args.dir,
                "--device", "cpu", "--jobs", "1", "--skip-trunk",
                "--codebook-base", "layer", *rest]
    rc = convert.main()
    if rc:
        return rc

    # ---- 3. every file the engine opens is there, at the size claimed ----
    m = json.load(open(os.path.join(args.dir, "manifest.json")))
    bad = []
    for L in layers:
        e = (m.get("engram") or {}).get(str(L))
        p = os.path.join(args.dir, f"engram-L{L}.bin")
        if not e:
            bad.append(f"manifest has no engram entry for layer {L}")
        elif not os.path.exists(p) or os.path.getsize(p) != e["bytes"]:
            bad.append(f"{p} is not the {e['bytes']} bytes the manifest says")
    have = {int(k) for k in (m.get("layers") or {})}
    want = {int(x.group(1)) for n in wm
            if (x := re.match(r"layers\.(\d+)\.ffn\.experts\.", n))}
    if have != want:
        bad.append(f"manifest lacks expert layers {sorted(want - have)[:8]}"
                   if want - have else
                   f"manifest has unexpected layers {sorted(have - want)[:8]}")
    for k, meta in (m.get("layers") or {}).items():
        p = os.path.join(args.dir, f"experts-L{k}.bin")
        if not os.path.exists(p) or os.path.getsize(p) != meta.get("bytes"):
            bad.append(f"{p} is missing or not {meta.get('bytes')} bytes")
    for fn in ("trunk.bin", "engram.json", "engram-tokmap.bin", "codebooks.bin"):
        if not os.path.exists(os.path.join(args.dir, fn)):
            bad.append(f"{fn} is missing")
    if bad:
        print("\n".join(bad), file=sys.stderr)
        return 1
    print(f"container complete: {len(m['layers'])} expert layers, "
          f"{len(layers)} Engram tables")
    return 0


if __name__ == "__main__":
    sys.exit(main())
