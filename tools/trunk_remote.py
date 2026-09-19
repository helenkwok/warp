#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""trunk_remote.py — build a container's trunk without downloading its experts.

K3's checkpoint is 1.42 TB, and 114 GB of it is trunk: every tensor that is
not a routed expert, spread over all 96 shards. The experts can be converted
one layer per machine (convert.py --codebook-base layer), but the trunk is
one pass over every shard, so it has been the part that needed the whole
download on one disk.

It does not need that. convert.py reads a tensor through mxfp4.ST.raw(),
which seeks to the tensor's offset and copies its bytes out. So this lays
each shard out as a *sparse* file of its real size holding only its header,
and fetches a tensor's bytes with one HTTP Range request at the moment
ST.raw() asks for them — then punches the hole back once they are copied.
Peak disk is the output plus one tensor; the bytes fetched are exactly the
trunk. convert.py itself runs unmodified.

Experts are hidden from this run (ST.have() answers False for them), so the
MoE layers come out "missing" and the manifest it publishes carries the
trunk alone. The gather step adds the layers:

  convert.py --src <small files> --out <trunk + banks + codebook parts>
             --skip-trunk --codebook-base layer

  python3 tools/trunk_remote.py --repo moonshotai/Kimi-K3 \\
      --src /data/k3src --out /data/k3.waste [--only N] [-- convert args]

--only N converts only the first N trunk tensors, to prove the plumbing in
minutes. Linux only (hole punching is fallocate(2)); elsewhere it keeps the
fetched bytes on disk, which is correct but needs the 114 GB.
"""

import argparse
import ctypes
import ctypes.util
import json
import os
import struct
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

FALLOC_FL_KEEP_SIZE, FALLOC_FL_PUNCH_HOLE = 0x01, 0x02


def punch_hole(path, off, length):
    """Give a fetched range's blocks back, keeping the file's size. Linux
    only (fallocate(2)); elsewhere a no-op that keeps the bytes on disk."""
    libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
    if not (hasattr(libc, "fallocate") and sys.platform.startswith("linux")):
        return False
    fd = os.open(path, os.O_RDWR)
    try:
        if libc.fallocate(fd, FALLOC_FL_PUNCH_HOLE | FALLOC_FL_KEEP_SIZE,
                          ctypes.c_long(off), ctypes.c_long(length)):
            raise OSError(ctypes.get_errno(), "fallocate punch hole")
    finally:
        os.close(fd)
    return True


def http_range(url, a, b, tries=6):
    """Bytes [a, b) of url. A short read is retried, never returned: a
    truncated tensor is exactly the silent failure this tool must not have."""
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={a}-{b - 1}"})
            with urllib.request.urlopen(req, timeout=300) as r:
                data = r.read()
            if len(data) == b - a:
                return data
            err = f"got {len(data)} of {b - a} bytes"
        except Exception as e:                      # noqa: BLE001 — retried
            err = repr(e)
        wait = min(120, 5 * 2 ** attempt)
        print(f"  range {a}-{b} failed ({err}); retry in {wait}s", flush=True)
        time.sleep(wait)
    raise RuntimeError(f"{url}: bytes {a}-{b} failed {tries} times")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="HF repo id of the checkpoint")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--src", required=True, help="where the sparse checkpoint goes")
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", type=int, default=0,
                    help="convert only the first N trunk tensors (smoke test)")
    ap.add_argument("--match", default="",
                    help="convert only trunk tensors whose name contains this "
                         "(smoke test; implies nothing about completeness)")
    ap.add_argument("rest", nargs=argparse.REMAINDER,
                    help="after --, arguments passed through to convert.py")
    args = ap.parse_args()
    raw_url = f"https://huggingface.co/{args.repo}/resolve/{args.revision}/"
    os.makedirs(args.src, exist_ok=True)

    # ---- the small files: config, index, tokenizer, templates, vision cfg --
    api_url = f"https://huggingface.co/api/models/{args.repo}/revision/{args.revision}"
    siblings = [s["rfilename"] for s in json.load(urllib.request.urlopen(api_url))["siblings"]]
    for fn in siblings:
        if fn.endswith(".safetensors") or "/" in fn:
            continue
        with urllib.request.urlopen(raw_url + fn) as r:
            open(os.path.join(args.src, fn), "wb").write(r.read())
    wm = json.load(open(os.path.join(args.src, "model.safetensors.index.json")))["weight_map"]

    # ---- each shard: its real size, its header, and nothing else ----------
    def lay_out(fn):
        path = os.path.join(args.src, fn)
        with urllib.request.urlopen(urllib.request.Request(raw_url + fn, method="HEAD")) as r:
            size = int(r.headers["Content-Length"])
        (n,) = struct.unpack("<Q", http_range(raw_url + fn, 0, 8))
        with open(path, "wb") as f:
            f.truncate(size)                        # sparse: no blocks allocated
            f.write(struct.pack("<Q", n) + http_range(raw_url + fn, 8, 8 + n))
        return fn

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(16) as ex:
        laid = list(ex.map(lay_out, sorted(set(wm.values()))))
    print(f"{len(laid)} sparse shards laid out", flush=True)

    import mxfp4
    import convert

    fetched = {"n": 0, "bytes": 0}
    only = {"left": args.only or None}
    orig_have, orig_raw = mxfp4.ST.have, mxfp4.ST.raw

    def have(self, name):
        # The experts are converted elsewhere, one layer per machine. Hidden
        # here, every MoE layer reads as absent — so this run neither fetches
        # nor converts them, and publishes a manifest with the trunk alone.
        if ".experts." in name:
            return False
        # DeepSeek-V4.1's Engram tables are 98 GB each and are converted by
        # tools/engram_remote.py, one row range per machine. Left visible,
        # build_engram would write them from this run's sparse shards — which
        # read as zeros — and publish a table of nothing. K3 has none.
        if ".engram.embed." in name:
            return False
        if only["left"] is not None and only["left"] <= 0:
            return False
        if args.match and args.match not in name:
            return False
        return orig_have(self, name)

    def raw(self, name):
        fn = self.wm[name]
        hdr, b0 = self._header(fn)
        beg, end = hdr[name]["data_offsets"]
        path = os.path.join(self.dir, fn)
        data = http_range(raw_url + fn, b0 + beg, b0 + end)
        with open(path, "r+b") as f:
            f.seek(b0 + beg)
            f.write(data)
        del data
        t = orig_raw(self, name)                    # copies the bytes out
        punch_hole(path, b0 + beg, end - beg)
        fetched["n"] += 1
        fetched["bytes"] += end - beg
        if only["left"] is not None:
            only["left"] -= 1
        if fetched["n"] % 100 == 0:
            print(f"  fetched {fetched['n']} tensors, {fetched['bytes'] / 1e9:.1f} GB",
                  flush=True)
        return t

    mxfp4.ST.have, mxfp4.ST.raw = have, raw

    rest = args.rest[1:] if args.rest[:1] == ["--"] else args.rest
    sys.argv = ["convert.py", "--src", args.src, "--out", args.out,
                "--device", "cpu", "--jobs", "1", *rest]
    rc = convert.main()
    print(f"convert.py returned {rc}; fetched {fetched['n']} tensors, "
          f"{fetched['bytes'] / 1e9:.2f} GB", flush=True)

    # ---- the verdict: every trunk tensor the checkpoint has is in the trunk --
    # build_trunk skips a tensor whose shard is absent without a word, so a
    # count is the only thing that tells a whole trunk from a short one.
    # Counting what was fetched would only prove the run agrees with itself,
    # so the full run is held to the index: every name build_trunk's own
    # filters keep. A subset run (--only/--match) can only be held to what
    # it read.
    def is_trunk(name):
        if ".experts." in name or name.endswith(
                (".weight_packed", ".weight_scale", ".weight_scale_inv")):
            return False
        if name.endswith(".scale") and name[: -len(".scale")] + ".weight" in wm:
            return False
        return not name.endswith(".engram.embed.weight")

    m = json.load(open(os.path.join(args.out, "manifest.json")))
    got = len(m.get("trunk") or [])
    subset = bool(args.only or args.match)
    want = fetched["n"] if subset else sum(1 for n in wm if is_trunk(n))
    if got != want or fetched["n"] != got or got == 0:
        print(f"trunk has {got} tensors; the index names {want} and "
              f"{fetched['n']} were fetched", file=sys.stderr)
        return 1
    print(f"trunk: {got} tensors, {os.path.getsize(os.path.join(args.out, 'trunk.bin')) / 1e9:.2f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
