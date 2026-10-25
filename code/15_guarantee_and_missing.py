"""What the gate guarantees, and how missing measurements affect the results (public dataset, published twin).

1. Scope of the guarantee. Proposition 1 bounds the marginal probability that an approved link violates the SLA,
   averaged over links and epochs. This script measures how far the corresponding conditional rates (per link, per
   hour, per bin of predicted risk) depart from it, for TwinGuard's verdicts.
2. Missing measurements. About a fifth of the outcomes are missing, more often under congestion. The false-approval
   rate is re-estimated with the missing outcomes of OK verdicts imputed (a) under missing at random given what the
   controller knows at decision time (link, last measured load and SLA state, age of the last measurement), with
   cell frequencies estimated on the measured outcomes of the same verdict; (b) under missing not at random, with
   the odds of a violation multiplied by delta, and the tipping point delta* at which the rate would reach epsilon;
   (c) in the worst case (every missing outcome is a violation), as published.
Outputs: results/twinguard/guarantee_scope.csv, results/twinguard/missing_outcomes.csv
Run from the repository root:  python code/15_guarantee_and_missing.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from twinguard import gate as G, realdata as rd  # noqa: E402

OUT = ROOT / "results" / "twinguard"
EPS = 0.05
LOAD_BINS = [0, 10, 40, 70, 90, np.inf]


def covariates(pipe, split):
    """Per decision epoch and link: last measured outflow (Mbps), steps since the last measurement."""
    load, age, k = [], [], 0
    while f"{split}_t_end_{k}" in pipe.files:
        V, idx = pipe[f"{split}_V_{k}"], pipe[f"{split}_t_end_{k}"]
        out = pd.DataFrame(V[..., 3]).ffill().to_numpy()
        obs = ~np.isnan(V[..., 0]); a = np.zeros(V.shape[:2]); run = np.zeros(V.shape[1])
        for t in range(len(V)):
            run = np.where(obs[t], 0, run + 1); a[t] = run
        load.append(out[idx]); age.append(a[idx]); k += 1
    return np.nan_to_num(np.concatenate(load), nan=0.0), np.concatenate(age)


def imputed_rate(v, ok, cells, delta=1.0, min_n=20):
    """False-approval rate among OK verdicts with missing outcomes imputed from the measured outcomes of OK verdicts
    in the same cell (coarser cells when a cell has fewer than min_n measured outcomes)."""
    meas = ~np.isnan(v); num = np.nansum(v[ok & meas]); den = ok.sum()
    miss = ok & ~meas
    p_hat = np.zeros(miss.sum()); todo = np.ones(miss.sum(), bool)
    for c in cells:                                                  # finest to coarsest
        key_m = c[ok & meas]; val_m = v[ok & meas]
        stats = pd.DataFrame({"k": key_m, "v": val_m}).groupby("k").v.agg(["sum", "count"])
        km = c[miss]
        s = stats.reindex(km)
        good = todo & (s["count"].to_numpy() >= min_n)
        p_hat[good] = ((s["sum"] + 0.5) / (s["count"] + 1)).to_numpy()[good]; todo &= ~good
    p_hat[todo] = np.nanmean(v[ok & meas])
    odds = delta * p_hat / np.clip(1 - p_hat, 1e-9, None); p = odds / (1 + odds)
    return float((num + p.sum()) / den), float(p.mean())


def main():
    pipe = np.load(ROOT / "data" / "processed" / "graph_pipeline.npz")
    cal = rd.load("cal", pipe=pipe); A_, B_ = rd.cal_halves(cal); calib = G.calibrate(A_, B_)
    scope, miss_rows = [], []
    for split in rd.TEST_SPLITS:
        d = rd.load(split, pipe=pipe)
        r = G.run(d, calib["cp"], calib["q_static"], calib["ref"], calib["phi_min"])
        ver = r["verdict"]; v = G.violated(d, rd.RTT_MAX, rd.PLR_MAX); s = G.scores(d)
        ok = ver == G.OK; meas = ~np.isnan(v); N, L = v.shape
        cov = s <= r["margin"][:, None]
        # ---- 1. marginal vs. conditional rates
        fa = lambda m: float(np.nansum(v[m & ok & meas]) / (m & ok & meas).sum()) if (m & ok & meas).sum() else np.nan
        cv = lambda m: float(cov[m & meas].mean()) if (m & meas).sum() else np.nan
        hours = ((d["ts"] - d["ts"][0]) // 3600).astype(int)
        per_link = [fa(np.broadcast_to(np.arange(L) == e, (N, L))) for e in range(L)]
        per_hour = [fa(np.broadcast_to((hours == h)[:, None], (N, L))) for h in np.unique(hours)]
        cov_link = [cv(np.broadcast_to(np.arange(L) == e, (N, L))) for e in range(L)]
        slack = np.minimum(np.log1p(rd.RTT_MAX) - d["q"][:, :, 0, 0, 2], np.log1p(rd.PLR_MAX) - d["q"][:, :, 0, 2, 2])
        qs = np.nanquantile(slack[ok], [0.1, 0.5])
        near = ok & (slack <= qs[0]); far = ok & (slack > qs[1])
        pl = np.array([x for x in per_link if not np.isnan(x)]); ph = np.array([x for x in per_hour if not np.isnan(x)])
        scope.append({"split": split, "false_approval_marginal": fa(np.ones((N, L), bool)),
                      "coverage_marginal": cv(np.ones((N, L), bool)),
                      "fa_link_max": float(pl.max()), "links_above_eps": int((pl > EPS).sum()), "links": int(len(pl)),
                      "coverage_link_min": float(np.nanmin(cov_link)),
                      "fa_hour_max": float(ph.max()), "hours_above_eps": int((ph > EPS).sum()), "hours": int(len(ph)),
                      "fa_near_budget": fa(near), "fa_far_from_budget": fa(far),
                      "approved_epochs": int(ok.any(1).sum()), "epochs": int(N)})
        # ---- 2. missing outcomes of OK verdicts
        load, age = covariates(pipe, split)
        lb = np.digitize(load, LOAD_BINS) - 1; cur = d["cur_v"].astype(int); ab = np.minimum(age, 2).astype(int)
        link = np.broadcast_to(np.arange(L), (N, L))
        cells = [link * 1000 + lb * 100 + cur * 10 + ab, link * 1000 + lb * 100 + cur * 10, lb * 100 + cur * 10, lb * 100]
        mar, p_miss = imputed_rate(v, ok, cells)
        tip = np.nan
        for delta in np.exp(np.linspace(0, np.log(1000), 400)):
            if imputed_rate(v, ok, cells, delta)[0] > EPS:
                tip = float(delta); break
        miss_rows.append({"split": split, "ok_missing_share": float((ok & ~meas).sum() / ok.sum()),
                          "missing_share": float((~meas).mean()),
                          "fa_measured": fa(np.ones((N, L), bool)), "fa_imputed_mar": mar,
                          "p_violation_imputed_mean": p_miss,
                          "fa_mnar_delta2": imputed_rate(v, ok, cells, 2.0)[0],
                          "fa_mnar_delta5": imputed_rate(v, ok, cells, 5.0)[0],
                          "tipping_delta": tip,
                          "fa_worst_case": float(np.nansum(np.where(meas, v, 1.0)[ok]) / ok.sum())})
        print(split, "done", flush=True)
    sc = pd.DataFrame(scope); sc.to_csv(OUT / "guarantee_scope.csv", index=False)
    mi = pd.DataFrame(miss_rows); mi.to_csv(OUT / "missing_outcomes.csv", index=False)
    pd.set_option("display.width", 250)
    print(sc.round(4).to_string(index=False)); print(mi.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
