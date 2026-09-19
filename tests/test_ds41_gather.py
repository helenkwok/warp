#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""test_ds41_gather.py — a table gathered from parts must be whole, or the
gather must refuse. A gap or an overlap gathered anyway is a container that
opens, runs, and reads the wrong rows of a 55 GB table without a word.

No torch, no network: this is tools/ds41_gather.py's own decision code.

  python3 tests/test_ds41_gather.py
"""
import os
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))
import ds41_gather                                                    # noqa: E402

RB, FAILS = 36, []


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'} {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILS.append(name)


def write_part(d, layer, lo, hi, fill=None, size=None):
    p = os.path.join(d, f"engram-L{layer}.rows-{lo:010d}-{hi:010d}.part")
    open(p, "wb").write(bytes([lo % 251]) * ((hi - lo) * RB if size is None else size))
    return p


def refuses(d, parts, total):
    try:
        ds41_gather.gather_table(d, 1, ds41_gather.find_parts(d)[1], total, RB)
    except SystemExit as e:
        return str(e)
    return None


def fresh():
    return tempfile.TemporaryDirectory()


def main():
    with fresh() as d:
        for lo, hi in ((0, 10), (10, 25), (25, 40)):
            write_part(d, 1, lo, hi)
        write_part(d, 14, 0, 5)                       # another table's part
        parts = ds41_gather.find_parts(d)
        check("parts are found per layer, in row order",
              sorted(parts) == [1, 14] and [p[:2] for p in parts[1]] == [(0, 10), (10, 25), (25, 40)])
        dest = ds41_gather.gather_table(d, 1, parts[1], 40, RB)
        got = open(dest, "rb").read()
        check("a whole table is concatenated in row order",
              len(got) == 40 * RB and got[:RB] == bytes([0]) * RB
              and got[10 * RB:11 * RB] == bytes([10]) * RB
              and got[25 * RB:26 * RB] == bytes([25]) * RB)
        check("the parts are removed once concatenated",
              not [n for n in os.listdir(d) if n.startswith("engram-L1.rows")])
        check("no .tmp is left behind", not [n for n in os.listdir(d) if n.endswith(".tmp")])
        check("another table's parts are untouched",
              os.path.exists(os.path.join(d, "engram-L14.rows-0000000000-0000000005.part")))

    with fresh() as d:
        write_part(d, 1, 0, 10); write_part(d, 1, 12, 20)
        msg = refuses(d, None, 20)
        check("a gap is refused", msg and "gap at row 10" in msg, msg)
        check("...and nothing is written", not os.path.exists(os.path.join(d, "engram-L1.bin")))

    with fresh() as d:
        write_part(d, 1, 0, 10); write_part(d, 1, 8, 20)
        msg = refuses(d, None, 20)
        check("an overlap is refused", msg and "overlap at row 10" in msg, msg)

    with fresh() as d:
        write_part(d, 1, 0, 10); write_part(d, 1, 10, 20, size=5)
        msg = refuses(d, None, 20)
        check("a truncated part is refused", msg and "its rows are" in msg, msg)

    with fresh() as d:
        write_part(d, 1, 0, 10); write_part(d, 1, 10, 20)
        msg = refuses(d, None, 30)
        check("a table that stops short is refused", msg and "parts end at row 20" in msg, msg)

    with fresh() as d:
        write_part(d, 1, 5, 20)
        msg = refuses(d, None, 20)
        check("a table that does not start at row 0 is refused", msg and "gap at row 0" in msg, msg)

    print(f"\n{'PASS' if not FAILS else 'FAIL'}: {len(FAILS)} failed")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
