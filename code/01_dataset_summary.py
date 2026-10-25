"""Step 1: full-file summary of the Zenodo dataset 10.5281/zenodo.17255307.

Produces the numbers for Table 3 (dataset summary), checks whether the
impairments described for D2/D3 are visible in the data, and reports what is
needed to set SLA budgets. Standard library only.

Usage: python3 -I code/01_dataset_summary.py data/zenodo_17255307 > results/01_dataset_summary.txt
"""
import csv
import os
import statistics as st
import sys
from collections import Counter, defaultdict

NUM = ["packet_loss", "jitter", "outflow_Router", "inflow_Router", "resolve",
       "avalability", "ttl", "probe_duration", "rtt"]
SCEN = {"D1": "dataset_01_TC33_27062025", "D2": "dataset_02_TC33_02072025",
        "D3": "dataset_03_TC33_05072025"}


def q(xs, p):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    k = (len(xs) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def load(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def is_empty(r):
    return all(r[c] == "" for c in NUM)


def fmt(x, d=2):
    return f"{x:.{d}f}"


def summarize(name, rows, wrows):
    print(f"\n==================== {name} ====================")
    ts = sorted({int(r["timestamp"]) for r in rows})
    steps = Counter(b - a for a, b in zip(ts, ts[1:]))
    links = sorted({r["link"] for r in rows})
    per_ts = Counter(int(r["timestamp"]) for r in rows)
    empty = [r for r in rows if is_empty(r)]
    full = [r for r in rows if not is_empty(r)]
    partial = sum(1 for r in full if any(r[c] == "" for c in NUM))
    dur_h = (ts[-1] - ts[0]) / 3600
    print(f"records={len(rows)}  timestamps={len(ts)}  links={len(links)}  duration={fmt(dur_h)} h")
    print(f"step sizes (s): {dict(steps.most_common(6))}")
    print(f"rows per timestamp: {dict(Counter(per_ts.values()).most_common(5))}")
    print(f"missing records: {len(empty)} ({100*len(empty)/len(rows):.1f}%)   partially missing: {partial}")
    lab = Counter(r["LABEL"] for r in full)
    tot = sum(lab.values())
    print("labels among measured records: " + ", ".join(
        f"{k}: {v} ({100*v/tot:.2f}%)" for k, v in sorted(lab.items())))
    print(f"labels on missing records (plain): {dict(Counter(r['LABEL'] for r in empty))}")

    # weighted variant consistency
    wl = Counter(r["LABEL"] for r in wrows)
    wempty = sum(1 for r in wrows if is_empty(r))
    print(f"weighted variant: records={len(wrows)} missing={wempty} labels={dict(sorted(wl.items()))}")
    key = lambda r: (r["timestamp"], r["link"])
    pm = {key(r): r["LABEL"] for r in rows}
    agree = diff = 0
    trans = Counter()
    for r in wrows:
        k = key(r)
        if k in pm:
            a = pm[k].replace(".0", "") or "missing"
            b = r["LABEL"]
            trans[(a, b)] += 1
    print(f"plain->weighted label mapping (top): {dict(trans.most_common(8))}")

    m = defaultdict(list)
    for r in full:
        for c in NUM:
            if r[c] != "":
                m[c].append(float(r[c]))
    print("metric quantiles over measured records [p05 / p50 / p95 / p99 / max]:")
    for c in ["rtt", "jitter", "packet_loss", "outflow_Router", "inflow_Router", "probe_duration", "ttl", "resolve", "avalability"]:
        v = m[c]
        print(f"  {c:15s} {fmt(q(v,.05))} / {fmt(q(v,.5))} / {fmt(q(v,.95))} / {fmt(q(v,.99))} / {fmt(max(v))}")

    # per-label medians
    print("per-label medians [rtt, jitter, loss, outflow]:")
    for L in sorted(lab):
        sel = [r for r in full if r["LABEL"] == L]
        med = lambda c: st.median(float(r[c]) for r in sel)
        print(f"  label {L}: n={len(sel)}  rtt={fmt(med('rtt'))}  jitter={fmt(med('jitter'))}  loss={fmt(med('packet_loss'))}  outflow={fmt(med('outflow_Router'))}")

    # per-link view
    print("per-link: missing% | median rtt | p95 rtt | median outflow | max outflow | label1+2 %")
    by = defaultdict(list)
    for r in rows:
        by[r["link"]].append(r)
    for l in links:
        rs = by[l]
        f_ = [r for r in rs if not is_empty(r)]
        rtt = [float(r["rtt"]) for r in f_]
        out = [float(r["outflow_Router"]) for r in f_]
        bad = sum(1 for r in f_ if r["LABEL"] in ("1.0", "2.0"))
        print(f"  {l:6s} {100*(len(rs)-len(f_))/len(rs):5.1f}% | {fmt(st.median(rtt))} | {fmt(q(rtt,.95))} | {fmt(st.median(out))} | {fmt(max(out))} | {100*bad/len(f_):.1f}%")

    # temporal view: is the impairment visible? hourly medians
    print("hourly view: hour | median rtt | p95 rtt | median jitter | mean loss | label1+2 % | missing %")
    t0 = ts[0]
    hb = defaultdict(list)
    for r in rows:
        hb[(int(r["timestamp"]) - t0) // 3600].append(r)
    for h in sorted(hb):
        rs = hb[h]
        f_ = [r for r in rs if not is_empty(r)]
        if not f_:
            print(f"  {h:3d} | no measurements")
            continue
        rtt = [float(r["rtt"]) for r in f_]
        jit = [float(r["jitter"]) for r in f_]
        loss = [float(r["packet_loss"]) for r in f_]
        bad = sum(1 for r in f_ if r["LABEL"] in ("1.0", "2.0"))
        print(f"  {h:3d} | {fmt(st.median(rtt))} | {fmt(q(rtt,.95))} | {fmt(st.median(jit))} | {fmt(st.mean(loss))} | {100*bad/len(f_):5.1f} | {100*(len(rs)-len(f_))/len(rs):5.1f}")

    # bandwidth ranges derived from outflow on loaded links
    rng = Counter()
    for r in full:
        o = float(r["outflow_Router"])
        rng[0 if o <= 10 else 1 if o <= 40 else 2 if o <= 70 else 3 if o <= 90 else 4] += 1
    print("derived bandwidth range of outflow (all measured records): " + str(dict(sorted(rng.items()))))
    return {"records": len(rows), "dur_h": dur_h, "missing": 100*len(empty)/len(rows),
            "lab": {k: 100*v/tot for k, v in lab.items()},
            "rtt50": st.median(m["rtt"]), "jit50": st.median(m["jitter"]),
            "loss50": st.median(m["packet_loss"]), "loss_mean": st.mean(m["packet_loss"])}


def main(d):
    out = {}
    for s, base in SCEN.items():
        rows = load(os.path.join(d, base + "_labeled.csv"))
        wrows = load(os.path.join(d, base + "_labeled_weighted.csv"))
        out[s] = summarize(s, rows, wrows)
    print("\n==================== Table 3 ====================")
    for s, v in out.items():
        lab = v["lab"]
        print(f"{s}: records={v['records']}  duration={v['dur_h']:.1f} h  missing={v['missing']:.1f}%  "
              f"labels 0/1/2 = {lab.get('0.0',0):.2f}/{lab.get('1.0',0):.2f}/{lab.get('2.0',0):.2f}%  "
              f"median RTT/jitter = {v['rtt50']:.2f}/{v['jit50']:.2f} ms  median loss = {v['loss50']:.2f}% (mean {v['loss_mean']:.3f}%)")


if __name__ == "__main__":
    main(sys.argv[1])
