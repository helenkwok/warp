#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""test_trunk_verdict.py — tools/trunk_remote.py's verdict on a finished trunk.

Written after the first DeepSeek-V4.1 run on GitHub Actions built the whole
trunk, correctly, in 17 minutes and would then have been failed by its own
verdict: it counted the 72 `mtp.*` tensors convert.py deliberately drops, and
it demanded one fetch per trunk tensor when a fp8 weight is fetched together
with its scale. Both held for K3 (bf16, no draft head) and neither for DS41.

The verdict still has to fail a trunk that is really short — that is what it
is for — so half of these are cases it must refuse.

No torch, no network.

  python3 tests/test_trunk_verdict.py
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))
from trunk_remote import trunk_verdict                              # noqa: E402

FAILS = []


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'} {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILS.append(name)


def drop_mtp(name):
    return name.startswith("mtp.")


def ds41_index(n_layers=3):
    """A DS41-shaped index: fp8 weights with `.scale`, an mtp head, experts,
    Engram tables. Returns (weight_map, number of trunk tensors it should yield)."""
    wm, trunk = {}, 0
    for L in range(n_layers):
        for k in ("attn.wq", "attn.wo", "ffn.gate"):
            wm[f"layers.{L}.{k}.weight"] = "s"
            wm[f"layers.{L}.{k}.scale"] = "s"          # companion: not a trunk tensor
            trunk += 1
        wm[f"layers.{L}.attn_norm.weight"] = "s"       # a norm has no scale
        trunk += 1
        wm[f"layers.{L}.ffn.experts.0.w1.weight"] = "s"
    wm["layers.1.engram.embed.weight"] = "s"
    wm["layers.1.engram.embed.scale"] = "s"
    for i in range(2):                                 # the DSpark head
        wm[f"mtp.{i}.attn.wq.weight"] = "s"
        wm[f"mtp.{i}.attn.wq.scale"] = "s"
    return wm, trunk


def main():
    wm, want = ds41_index()
    fetched = want + 3 * len(range(3))                 # weights + their scales

    ok, msg = trunk_verdict(wm, want, fetched, False, drop_mtp)
    check("a whole DS41 trunk passes", ok, msg)

    ok, msg = trunk_verdict(wm, want, want + 40, False, drop_mtp)
    check("...however many companion fetches there were", ok, msg)

    ok, msg = trunk_verdict(wm, want - 1, fetched, False, drop_mtp)
    check("a trunk one tensor short is refused", not ok, msg)

    ok, msg = trunk_verdict(wm, want + 1, fetched, False, drop_mtp)
    check("a trunk one tensor long is refused", not ok, msg)

    ok, msg = trunk_verdict(wm, want, fetched, False, None)
    check("without convert.py's drop rule the mtp names are counted (the bug)",
          not ok, msg)

    ok, msg = trunk_verdict(wm, want, want - 1, False, drop_mtp)
    check("fewer fetches than tensors is refused", not ok, msg)

    ok, msg = trunk_verdict(wm, 0, 0, False, drop_mtp)
    check("an empty trunk is refused", not ok, msg)

    # K3: bf16, one fetch per tensor, no drop rule — must keep passing
    k3 = {f"model.layers.{L}.self_attn.q_proj.weight": "s" for L in range(4)}
    k3["model.layers.0.mlp.experts.0.w1.weight"] = "s"
    ok, msg = trunk_verdict(k3, 4, 4, False, None)
    check("K3's trunk (bf16, no drop rule) still passes", ok, msg)

    # a subset run is held to what it read, not to the index
    ok, msg = trunk_verdict(wm, 3, 3, True, drop_mtp)
    check("a subset run is held to what it read", ok, msg)
    ok, msg = trunk_verdict(wm, 2, 3, True, drop_mtp)
    check("...and refused when the trunk holds fewer", not ok, msg)

    print(f"\n{'PASS' if not FAILS else 'FAIL'}: {len(FAILS)} failed")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
