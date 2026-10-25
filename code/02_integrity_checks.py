"""Step 1b: integrity checks on the Zenodo dataset.

Checks (i) whether scenarios share identical records (overlap between files),
(ii) duplicated (timestamp, link) keys within a file, and (iii) the time
segments of each file separated by gaps. Standard library only.

Usage: python3 -I code/02_integrity_checks.py data/zenodo_17255307 > results/02_integrity_checks.txt
"""
import csv
import os
import sys
from collections import Counter
from datetime import datetime, timezone

SCEN = {"D1": "dataset_01_TC33_27062025", "D2": "dataset_02_TC33_02072025",
        "D3": "dataset_03_TC33_05072025"}
VALS = ["packet_loss", "jitter", "outflow_Router", "inflow_Router", "resolve",
        "avalability", "ttl", "probe_duration", "rtt"]


def load(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def utc(t):
    return datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")  # noqa: UP017


def segments(ts, gap=300):
    segs, start = [], ts[0]
    for a, b in zip(ts, ts[1:]):
        if b - a > gap:
            segs.append((start, a))
            start = b
    segs.append((start, ts[-1]))
    return segs


def main(d):
    data = {s: load(os.path.join(d, b + "_labeled.csv")) for s, b in SCEN.items()}
    keys, full = {}, {}
    for s, rows in data.items():
        k = Counter((r["timestamp"], r["link"]) for r in rows)
        dup = sum(v - 1 for v in k.values() if v > 1)
        keys[s] = set(k)
        full[s] = Counter(tuple(r[c] for c in ["timestamp", "link"] + VALS) for r in rows)
        identical_dups = sum(v - 1 for v in full[s].values() if v > 1)
        ts = sorted({int(r["timestamp"]) for r in rows})
        print(f"\n== {s}: rows={len(rows)} unique (timestamp,link)={len(k)} duplicated keys={dup} "
              f"of which fully identical rows={identical_dups}")
        for a, b in segments(ts):
            n = sum(1 for r in rows if a <= int(r["timestamp"]) <= b)
            print(f"   segment {utc(a)} -> {utc(b)} UTC  ({(b-a)/3600:.2f} h, {n} rows)")

    print("\n== Overlap between scenarios (same timestamp and link)")
    names = list(SCEN)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            ov = keys[a] & keys[b]
            same = sum(min(full[a][t], full[b][t]) for t in full[a] if t in full[b])
            print(f"   {a} & {b}: shared keys={len(ov)}  identical full rows={same}")
            if ov:
                t = sorted(int(x[0]) for x in ov)
                print(f"      shared time span {utc(t[0])} -> {utc(t[-1])} UTC")


if __name__ == "__main__":
    main(sys.argv[1])
