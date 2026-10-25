"""Step 1c: apply the cleaning protocol, write processed CSVs, and compute
Table 3 plus SLA-budget violation rates.

Cleaning protocol (motivated by 02_integrity_checks):
  * D2: keep only records with timestamp >= 2025-07-01 00:00 UTC. Earlier
    D2 records are three identical copies of the last D1 segment
    (2025-06-27 13:29 -> 2025-06-28 01:29 UTC) and are dropped.
  * D1, D3: used as published (no duplicated keys, no cross-file overlap).
Missing records (all metrics empty) are kept; they are masked downstream.

Usage: python3 -I code/03_clean_dataset.py data/zenodo_17255307 data/processed > results/03_clean_dataset.txt
"""
import csv
import os
import statistics as st
import sys
from collections import Counter

SCEN = {"D1": "dataset_01_TC33_27062025", "D2": "dataset_02_TC33_02072025",
        "D3": "dataset_03_TC33_05072025"}
NUM = ["packet_loss", "jitter", "outflow_Router", "inflow_Router", "resolve",
       "avalability", "ttl", "probe_duration", "rtt"]
D2_START = 1751328000  # 2025-07-01 00:00:00 UTC
RTT_BUDGETS = [5.0, 10.0, 15.0]   # ms
LOSS_BUDGETS = [0.5, 1.0, 2.0]    # %


def load(path):
    with open(path, encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        return r.fieldnames, list(r)


def q(xs, p):
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def main(src, dst):
    os.makedirs(dst, exist_ok=True)
    print("Cleaning protocol: D2 restricted to timestamp >= 2025-07-01 UTC; D1, D3 unchanged.\n")
    rows_t5 = []
    for s, base in SCEN.items():
        cols, rows = load(os.path.join(src, base + "_labeled.csv"))
        n_raw = len(rows)
        if s == "D2":
            rows = [r for r in rows if int(r["timestamp"]) >= D2_START]
        keys = Counter((r["timestamp"], r["link"]) for r in rows)
        assert max(keys.values()) == 1, f"{s}: duplicated keys remain"
        with open(os.path.join(dst, f"{s}.csv"), "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)

        ts = sorted({int(r["timestamp"]) for r in rows})
        gaps = [b - a for a, b in zip(ts, ts[1:])]
        dur = sum(g for g in gaps if g <= 300) / 3600   # measured time, gaps excluded
        full = [r for r in rows if not all(r[c] == "" for c in NUM)]
        miss = 100 * (len(rows) - len(full)) / len(rows)
        lab = Counter(r["LABEL"] for r in full)
        tot = len(full)
        rtt = [float(r["rtt"]) for r in full]
        jit = [float(r["jitter"]) for r in full]
        loss = [float(r["packet_loss"]) for r in full]
        print(f"== {s}: raw rows={n_raw} kept={len(rows)} measured={tot} missing={miss:.1f}% "
              f"timestamps={len(ts)} measured duration={dur:.1f} h")
        print(f"   labels 0/1/2 among measured: "
              f"{100*lab['0.0']/tot:.2f} / {100*lab['1.0']/tot:.2f} / {100*lab['2.0']/tot:.2f} %")
        print(f"   RTT  p50={st.median(rtt):.2f} p95={q(rtt,.95):.2f} p99={q(rtt,.99):.2f} ms")
        print(f"   jitter p50={st.median(jit):.2f} p95={q(jit,.95):.2f} ms;  loss p50={st.median(loss):.2f} "
              f"p95={q(loss,.95):.2f} p99={q(loss,.99):.2f} %  mean={st.mean(loss):.3f}%")
        print("   share of measured records violating an SLA (RTT > R or loss > L), in %:")
        print("        " + "  ".join(f"L={L:>3}%" for L in LOSS_BUDGETS))
        for R in RTT_BUDGETS:
            v = []
            for L in LOSS_BUDGETS:
                n = sum(1 for a, b in zip(rtt, loss) if a > R or b > L)
                v.append(f"{100*n/tot:6.2f}")
            print(f"   R={R:>4} ms " + "  ".join(v))
        rows_t5.append((s, len(rows), dur, miss, lab, tot, st.median(rtt), st.median(jit),
                        q(rtt, .95), q(loss, .99)))

    print("\n== Table 3 (cleaned)")
    for s, n, dur, miss, lab, tot, r50, j50, r95, l99 in rows_t5:
        print(f"{s}: records={n} measured_duration={dur:.1f}h missing={miss:.1f}% "
              f"labels={100*lab['0.0']/tot:.1f}/{100*lab['1.0']/tot:.2f}/{100*lab['2.0']/tot:.2f}% "
              f"RTT50/jit50={r50:.2f}/{j50:.2f} ms RTT95={r95:.2f} ms loss99={l99:.2f}%")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
