#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""test_engram_remote.py — tools/engram_remote.py must produce the same bytes
as a local conversion while fetching only the rows it was asked for.

A local HTTP server with Range support stands in for the Hub. The shard is
laid out sparse (header only), which reads as zeros — so a conversion that
did not fetch would quantize zeros and the comparison would fail. The test
checks that too, so it cannot pass by never fetching.

Skipped, not failed, where torch is missing.

  python3 tests/test_engram_remote.py
"""
import http.server
import os
import re
import sys
import tempfile
import threading

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tests"))
sys.path.insert(0, os.path.join(REPO, "tools"))

try:
    import torch  # noqa: F401
except ImportError:
    print("SKIP test_engram_remote: torch is not installed")
    sys.exit(0)

from test_engram_rows import ROWS, DIM, LAYER, build_checkpoint      # noqa: E402

FAILS = []


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'} {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILS.append(name)


def serve(root):
    """A file server that honours Range, and counts the bytes it sent."""
    sent = {"bytes": 0}

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _path(self):
            return os.path.join(root, self.path.lstrip("/"))

        def do_HEAD(self):
            self.send_response(200)
            self.send_header("Content-Length", str(os.path.getsize(self._path())))
            self.end_headers()

        def do_GET(self):
            size = os.path.getsize(self._path())
            m = re.match(r"bytes=(\d+)-(\d+)", self.headers.get("Range", ""))
            a, b = (int(m.group(1)), int(m.group(2)) + 1) if m else (0, size)
            with open(self._path(), "rb") as f:
                f.seek(a)
                data = f.read(b - a)
            sent["bytes"] += len(data)
            self.send_response(206 if m else 200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, sent


def main():
    import convert
    import engram_remote
    import mxfp4

    with tempfile.TemporaryDirectory() as tmp:
        full, sparse = os.path.join(tmp, "full"), os.path.join(tmp, "sparse")
        whole, parts = os.path.join(tmp, "whole"), os.path.join(tmp, "parts")
        for d in (full, sparse, whole, parts):
            os.makedirs(d)
        build_checkpoint(full)
        name = f"layers.{LAYER}.engram.embed.weight"
        row_bytes = DIM * 4 // 8 + (DIM // 32) * 2

        convert.write_engram(mxfp4.ST(full), whole, name, LAYER, chunk_rows=64)
        ref = open(os.path.join(whole, f"engram-L{LAYER}.bin"), "rb").read()

        srv, sent = serve(full)
        raw_url = f"http://127.0.0.1:{srv.server_address[1]}/"

        # the sparse checkpoint: index and config as published, shard = header
        for fn in ("model.safetensors.index.json", "config.json"):
            open(os.path.join(sparse, fn), "wb").write(open(os.path.join(full, fn), "rb").read())
        engram_remote.lay_out(raw_url, sparse, "shard.safetensors")
        st = mxfp4.ST(sparse)

        # Without the fetch, a sparse shard reads as zeros — the failure this
        # test exists to catch.
        zeros = st.row_slice(name, 0, 4)
        check("an unpatched sparse shard reads as zeros", not zeros.float().any())

        fetched = {"n": 0, "bytes": 0}
        orig = engram_remote.install(mxfp4, raw_url, fetched)
        try:
            sent["bytes"] = 0
            cuts = [0, 137, 500, ROWS]
            for lo, hi in zip(cuts, cuts[1:]):
                rc = convert.engram_part(st, {"engram_layer_ids": [LAYER]}, parts,
                                         LAYER, lo, hi, 4)
                p = os.path.join(parts, convert.engram_part_name(LAYER, lo, hi))
                got = open(p, "rb").read() if os.path.exists(p) else b""
                check(f"rows {lo}:{hi} over HTTP equal the local conversion",
                      rc == 0 and got == ref[lo * row_bytes: hi * row_bytes])
        finally:
            mxfp4.ST.row_slice = orig
            srv.shutdown()

        # Payload is DIM bytes a row (fp8) and the scale DIM/32 bytes a row
        # (e8m0): that, and nothing else, is what was fetched.
        want = ROWS * (DIM + DIM // 32)
        check("fetched exactly the requested rows' bytes", fetched["bytes"] == want,
              f"{fetched['bytes']} vs {want}")
        check("...and the server sent no more than that",
              sent["bytes"] <= want, f"{sent['bytes']} sent")

    print(f"\n{'PASS' if not FAILS else 'FAIL'}: {len(FAILS)} failed")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
