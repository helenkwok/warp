#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""engram_remote.py — convert one row range of an Engram table without
downloading the shard that holds it.

DeepSeek-V4.1's two Engram tables are 98 GB of fp8 apiece, each in a 101.5 GB
shard of its own. Downloaded, a shard plus the 55 GB it converts to does not
fit a free CI runner's disk. But `convert.py --engram-rows` needs only the
rows it was asked for, so this fetches only those.

It works the way tools/trunk_remote.py does. The shard is laid out as a
*sparse* file of its real size holding only its header; when ST.row_slice()
asks for rows, their bytes are fetched with one HTTP Range request, written
into the file where they belong, read back through the unmodified
ST.row_slice(), and the hole is punched again. Peak disk is the part being
written plus one chunk. Nothing about how a row is quantized lives here.

  python3 tools/engram_remote.py --repo deepseek-ai/DeepSeek-V4.1-Flash \\
      --src /data/ds41src --out /data/ds41parts --layer 1 --rows 0:48000000

writes engram-L1.rows-0000000000-0048000000.part into --out (see
tools/ds41_gather.py for what puts the parts back together).
"""

import argparse
import json
import os
import struct
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import trunk_remote                                              # noqa: E402

# Only the dtypes a row-sliced table can be. ST.row_slice knows more; this is
# just enough to turn a row range into a byte range before it is asked.
ITEMSIZE = {"U8": 1, "I8": 1, "F8_E8M0": 1, "F8_E4M3": 1, "F8_E5M2": 1,
            "BF16": 2, "F16": 2, "F32": 4}


def fetch_small_files(repo, revision, dest):
    """Config, index, tokenizer, templates: everything but the shards."""
    raw_url = f"https://huggingface.co/{repo}/resolve/{revision}/"
    api_url = f"https://huggingface.co/api/models/{repo}/revision/{revision}"
    with urllib.request.urlopen(api_url) as r:
        siblings = [s["rfilename"] for s in json.load(r)["siblings"]]
    for fn in siblings:
        if fn.endswith(".safetensors") or "/" in fn:
            continue
        with urllib.request.urlopen(raw_url + fn) as r:
            open(os.path.join(dest, fn), "wb").write(r.read())
    return raw_url


def lay_out(raw_url, dest, fn):
    """`fn` as a sparse file of its real size holding only its header."""
    path = os.path.join(dest, fn)
    with urllib.request.urlopen(urllib.request.Request(raw_url + fn, method="HEAD")) as r:
        size = int(r.headers["Content-Length"])
    (n,) = struct.unpack("<Q", trunk_remote.http_range(raw_url + fn, 0, 8))
    with open(path, "wb") as f:
        f.truncate(size)                            # sparse: no blocks allocated
        f.write(struct.pack("<Q", n) + trunk_remote.http_range(raw_url + fn, 8, 8 + n))


def install(mxfp4, raw_url, fetched):
    """Make ST.row_slice fetch the rows it is asked for. Returns the original,
    so a test can put it back."""
    orig = mxfp4.ST.row_slice

    def row_slice(self, name, r0, r1):
        fn = self.wm[name]
        hdr, b0 = self._header(fn)
        meta = hdr[name]
        shape = meta["shape"]
        beg, end = meta["data_offsets"]
        r0, r1 = max(0, r0), min(shape[0], r1)
        if r1 <= r0 or len(shape) != 2:
            return orig(self, name, r0, r1)         # let it say why
        stride = shape[1] * ITEMSIZE[meta["dtype"]]
        a, b = b0 + beg + r0 * stride, b0 + beg + r1 * stride
        if b0 + end < b:
            return orig(self, name, r0, r1)         # runs past the tensor
        path = os.path.join(self.dir, fn)
        data = trunk_remote.http_range(raw_url + fn, a, b)
        with open(path, "r+b") as f:
            f.seek(a)
            f.write(data)
        del data
        t = orig(self, name, r0, r1)                # copies the bytes out
        trunk_remote.punch_hole(path, a, b - a)
        fetched["n"] += 1
        fetched["bytes"] += b - a
        return t

    mxfp4.ST.row_slice = row_slice
    return orig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="HF repo id of the checkpoint")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--src", required=True, help="where the sparse checkpoint goes")
    ap.add_argument("--out", required=True)
    ap.add_argument("--layer", type=int, required=True, help="Engram layer")
    ap.add_argument("--rows", required=True, metavar="LO:HI")
    ap.add_argument("rest", nargs=argparse.REMAINDER,
                    help="after --, arguments passed through to convert.py")
    args = ap.parse_args()
    os.makedirs(args.src, exist_ok=True)
    os.makedirs(args.out, exist_ok=True)

    raw_url = fetch_small_files(args.repo, args.revision, args.src)
    wm = json.load(open(os.path.join(args.src, "model.safetensors.index.json")))["weight_map"]
    table = f"layers.{args.layer}.engram.embed.weight"
    if table not in wm:
        print(f"{table} is not in {args.repo}'s index", file=sys.stderr)
        return 1
    # The table and its scales, which DeepSeek ships side by side.
    for fn in sorted({wm[table], wm[table.replace(".weight", ".scale")]}):
        lay_out(raw_url, args.src, fn)

    import mxfp4
    import convert

    fetched = {"n": 0, "bytes": 0}
    install(mxfp4, raw_url, fetched)

    rest = args.rest[1:] if args.rest[:1] == ["--"] else args.rest
    sys.argv = ["convert.py", "--src", args.src, "--out", args.out,
                "--engram-layer", str(args.layer), "--engram-rows", args.rows,
                "--device", "cpu", "--jobs", "1", *rest]
    rc = convert.main()
    print(f"convert.py returned {rc}; fetched {fetched['n']} ranges, "
          f"{fetched['bytes'] / 1e9:.2f} GB", flush=True)

    # convert.py keeps a finished slice without reading it, so a run that
    # fetched nothing is only right when the part was already there.
    lo, hi = (int(x) for x in args.rows.split(":"))
    part = os.path.join(args.out, convert.engram_part_name(args.layer, lo, hi))
    if rc == 0 and not os.path.exists(part):
        print(f"{part} was not written", file=sys.stderr)
        return 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
