"""Metrics of the simulator experiments: action-consequence accuracy (counterfactual), coverage of the actions the
agents select, and closed-loop control.

Inputs : results/sim[_<topology>]/{counterfactual.npz, counterfactual_cal.npz, closed_loop.csv}, data/sim[...]/cal.npz
Outputs: counterfactual_metrics.csv, effect_metrics.csv, decision_metrics.csv, coverage_actions.csv,
         selection_coverage.csv, closed_loop_summary.csv, sim_results.json (in results/sim[_<topology>])
Run from the repository root:  [TG_TOPOLOGY=nsfnet] python code/11_sim_results.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from code_sim_common import RES as OUT, TWINS, load_sim_twin, sim, sim_calibration  # noqa: E402
from twinguard import gate as G  # noqa: E402
from twinguard.model import auroc  # noqa: E402

LR, LP = np.log1p(sim.RTT_MAX), np.log1p(sim.PLR_MAX)
TW = {"GNN twin": "twin", "Per-link twin": "link"}
WB = "White-box queueing model"


def labels(z, h):
    y, ym = z["y"][:, :, h], z["ym"][:, :, h]
    v = ((y[..., 0] > sim.RTT_MAX) | (y[..., 2] > sim.PLR_MAX)).astype(float)
    return np.where(ym, v, np.nan)


def scores(z, h):
    """Violation scores of every model for the candidate outcomes at horizon index h."""
    pr = np.nan_to_num(z["persist"], nan=0.0)
    return {**{n: z[p + "_p"][:, :, h] for n, p in TW.items()},
            WB: np.maximum(z["ana_rtt"] / sim.RTT_MAX, z["ana_loss"] / sim.PLR_MAX),
            "Persistence": np.maximum(pr[..., 0] / sim.RTT_MAX, pr[..., 1] / sim.PLR_MAX)}


def rtt_median(z, h):
    return {**{n: np.expm1(z[p + "_q"][:, :, h, 0, 1]) for n, p in TW.items()},
            WB: z["ana_rtt"], "Persistence": z["persist"][..., 0]}


def accuracy(z):
    rows = []
    for regime in ("nominal", "shift"):
        for subset in ("all", "actions", "no-op"):
            sel = z["regime"] == regime
            if subset == "actions":
                sel &= z["kind"] != "noop"
            elif subset == "no-op":
                sel &= z["kind"] == "noop"
            for h, hs in ((0, 15), (2, 60)):
                v = labels(z, h)[sel]; sc = scores(z, h); med = rtt_median(z, h)
                y = z["y"][sel][:, :, h, 0]; ym = z["ym"][sel][:, :, h]
                for name in sc:
                    s = sc[name][sel]; m = ~np.isnan(v)
                    mae = np.nanmean(np.abs(np.where(ym, med[name][sel] - y, np.nan)))
                    row = {"regime": regime, "subset": subset, "horizon_s": hs, "model": name,
                           "auroc": auroc(v[m], s[m]), "mae_rtt_ms": mae, "n": int(m.sum()),
                           "violation_rate": float(np.nanmean(v))}
                    if name in TW:
                        q95 = np.expm1(z[TW[name] + "_q"][sel][:, :, h, 0, 2]); u = y - q95
                        row["pinball_rtt"] = float(np.nanmean(np.where(ym, np.maximum(0.95 * u, -0.05 * u), np.nan)))
                        row["brier"] = float(np.nanmean((s[m] - v[m]) ** 2))
                    rows.append(row)
    return pd.DataFrame(rows)


def effects(z):
    """Does the model predict how an action changes the RTT of every link (relative to the no-op)?"""
    rows = []
    for regime in ("nominal", "shift"):
        r = z["regime"] == regime; names = list(TW) + [WB]
        true, pred = {k: [] for k in names}, {k: [] for k in names}
        for e in np.unique(z["epoch_id"][r]):
            idx = np.where(z["epoch_id"] == e)[0]; n0 = idx[z["kind"][idx] == "noop"][0]
            for i in idx[z["kind"][idx] != "noop"]:
                both = z["ym"][i, :, 0] & z["ym"][n0, :, 0]
                dt = np.log1p(z["y"][i, :, 0, 0]) - np.log1p(z["y"][n0, :, 0, 0])
                dp = {**{n: z[p + "_q"][i, :, 0, 0, 1] - z[p + "_q"][n0, :, 0, 0, 1] for n, p in TW.items()},
                      WB: np.log1p(z["ana_rtt"][i]) - np.log1p(z["ana_rtt"][n0])}
                for k in names:
                    true[k].append(dt[both]); pred[k].append(dp[k][both])
        for k in names:
            t = np.concatenate(true[k]); p = np.concatenate(pred[k]); big = np.abs(t) > 0.1
            rows.append({"regime": regime, "model": k, "spearman": spearmanr(t, p).statistic,
                         "spearman_large": spearmanr(t[big], p[big]).statistic,
                         "sign_agreement_large": float(np.mean(np.sign(t[big]) == np.sign(p[big]))),
                         "n_link_effects": int(len(t)), "n_large": int(big.sum())})
    return pd.DataFrame(rows)


def decisions(z):
    """At epochs where the no-op leads to a violation 15 s later, does the action chosen with each model resolve it?"""
    rows = []
    for regime in ("nominal", "shift"):
        r = z["regime"] == regime; v = labels(z, 0); sc = scores(z, 0)
        res = {k: [] for k in sc}; best, n_ep = [], 0
        for e in np.unique(z["epoch_id"][r]):
            idx = np.where(z["epoch_id"] == e)[0]; n0 = idx[z["kind"][idx] == "noop"][0]
            nv = np.nansum(v[idx], 1)
            if nv[np.where(idx == n0)[0][0]] == 0:
                continue
            n_ep += 1; best.append(nv.min() == 0)
            for k in sc:
                exp_v = (sc[k][idx] > (0.5 if k in TW else 1.0)).sum(1) + 0.01 * z["cost"][idx].astype(float)
                res[k].append(nv[np.argmin(exp_v)] == 0)
        for k in res:
            rows.append({"regime": regime, "model": k, "epochs_with_noop_violation": n_ep,
                         "resolved_share": float(np.mean(res[k])), "oracle_resolvable_share": float(np.mean(best))})
    return pd.DataFrame(rows)


def joint_scores(z, p, sel, h=0):
    q95 = z[p + "_q"][sel][:, :, h, :, 2]; y = z["y"][sel][:, :, h]; ym = z["ym"][sel][:, :, h]
    s = np.maximum(np.log1p(np.clip(y[..., 0], 0, None)) - q95[..., 0], np.log1p(np.clip(y[..., 2], 0, None)) - q95[..., 2])
    return np.where(ym, s, np.nan), q95


def coverage(z, calib):
    rows = []
    for name, p in TW.items():
        for regime in ("nominal", "shift"):
            for subset in ("no-op", "actions"):
                sel = (z["regime"] == regime) & ((z["kind"] == "noop") if subset == "no-op" else (z["kind"] != "noop"))
                s, _ = joint_scores(z, p, sel); m = ~np.isnan(s)
                rows.append({"twin": name, "regime": regime, "subset": subset, "raw_q95": float(np.mean(s[m] <= 0)),
                             "static_conformal": float(np.mean(s[m] <= calib[name]["q_static"]))})
    return pd.DataFrame(rows)


def selection_coverage(z, zc, calib, eps=0.05):
    """Coverage of the action the gate would select among the planner's candidates (cheapest candidate whose bounds
    all hold), for four calibrations: exploration data (the published calibration) or the planner's own candidates
    on the calibration scenarios (counterfactual forks), each at level eps or with the selection correction
    eps / (K + 1). Mondrian: planner calibration per action type. Coverage = share of measured link outcomes of the
    selected action within its bound; violation = share of selected actions followed by a violation 15 s later."""
    rows = []
    for name, p in TW.items():
        sc_c, _ = joint_scores(zc, p, zc["src"] == "planner")
        plan_cp = G.Conformal(sc_c.ravel())
        kinds_c = np.repeat(zc["kind"][zc["src"] == "planner"][:, None], sc_c.shape[1], 1)
        mond = {k: G.Conformal(sc_c[kinds_c == k]) for k in np.unique(kinds_c)}
        variants = {"exploration": calib[name]["cp"], "planner": plan_cp, "planner, per action type": None}
        for regime in ("nominal", "shift"):
            r = z["regime"] == regime
            for vname, cp in variants.items():
                for corr in (False, True):
                    cov_n = cov_d = viol = n_sel = 0
                    for e in np.unique(z["epoch_id"][r]):
                        idx = np.where((z["epoch_id"] == e) & (z["src"] == "planner"))[0]
                        if not len(idx):
                            continue
                        K = len(idx); a = eps / (K + 1) if corr else eps
                        s, q95 = joint_scores(z, p, idx)
                        slack = np.minimum(LR - q95[..., 0], LP - q95[..., 2])
                        m = np.array([(mond.get(z["kind"][i], plan_cp) if cp is None else cp).q(a) for i in idx])
                        feas = np.where((slack >= m[:, None]).all(1))[0]
                        if not len(feas):
                            continue
                        j = feas[np.argmin(z["cost"][idx][feas].astype(float))]
                        ok = ~np.isnan(s[j]); cov_n += int((s[j][ok] <= m[j]).sum()); cov_d += int(ok.sum())
                        vv = labels(z, 0)[idx[j]]; viol += int(np.nansum(vv) > 0); n_sel += 1
                    rows.append({"twin": name, "regime": regime, "calibration": vname, "selection_correction": corr,
                                 "selected_actions": n_sel, "coverage_selected": cov_n / cov_d if cov_d else np.nan,
                                 "violation_after_selected": viol / n_sel if n_sel else np.nan})
    return pd.DataFrame(rows)


def default_routing(seeds=range(800, 806), hours=6.0):
    """SLA violations under static shortest-path routing (nominal law) and their dependence on utilization."""
    V, R = [], []
    for seed in seeds:
        ex = sim.make_exo(seed, hours, "test")
        out, _ = sim.Network(ex).run(lambda k: sim.Config(), 0, ex.rates.shape[1] // sim.STEP)
        V.append(sim.violation(out)); R.append(out["load"] / sim.C_NOM)
    V, R = np.concatenate(V), np.concatenate(R); m = ~np.isnan(V)
    bins = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.8), (0.8, 0.85), (0.85, 0.9), (0.9, 0.95), (0.95, 1.0), (1.0, 9.0)]
    return {"violation_rate": float(np.nanmean(V)),
            "by_utilization": {f"{lo:.2f}-{hi:.2f}": float(V[m & (R >= lo) & (R < hi)].mean()) for lo, hi in bins}}


def closed_summary(cl):
    agg = {c: (c, "mean") for c in cl.columns if c not in ("regime", "missing", "seed", "controller")
           and pd.api.types.is_numeric_dtype(cl[c])}
    summ = cl.groupby(["regime", "missing", "controller"]).agg(**agg).reset_index()
    sd = cl.groupby(["regime", "missing", "controller"]).violation_rate.std().rename("violation_sd").reset_index()
    return summ.merge(sd, on=["regime", "missing", "controller"])


def main():
    pd.set_option("display.width", 250)
    calib = {n: sim_calibration(load_sim_twin(m)) for n, m in zip(TW, TWINS)}
    info = {}
    if (OUT / "counterfactual.npz").exists():
        z = dict(np.load(OUT / "counterfactual.npz", allow_pickle=True))
        acc = accuracy(z); acc.to_csv(OUT / "counterfactual_metrics.csv", index=False)
        eff = effects(z); eff.to_csv(OUT / "effect_metrics.csv", index=False)
        dec = decisions(z); dec.to_csv(OUT / "decision_metrics.csv", index=False)
        cov = coverage(z, calib); cov.to_csv(OUT / "coverage_actions.csv", index=False)
        print(acc[(acc.horizon_s == 15) & (acc.subset == "all")].round(4).to_string(index=False))
        print(eff.round(3).to_string(index=False)); print(dec.round(3).to_string(index=False))
        print(cov.round(4).to_string(index=False))
        if (OUT / "counterfactual_cal.npz").exists():
            zc = dict(np.load(OUT / "counterfactual_cal.npz", allow_pickle=True))
            selc = selection_coverage(z, zc, calib); selc.to_csv(OUT / "selection_coverage.csv", index=False)
            print(selc.round(4).to_string(index=False))
        info.update({"n_candidates": int(len(z["kind"])), "n_epochs": int(len(np.unique(z["epoch_id"])))})
    if (OUT / "closed_loop.csv").exists():
        summ = closed_summary(pd.read_csv(OUT / "closed_loop.csv"))
        summ.to_csv(OUT / "closed_loop_summary.csv", index=False)
        print(summ[["regime", "missing", "controller", "violation_rate", "violation_rate_all", "actions_per_h",
                    "abstention_rate", "false_approval", "coverage_act"]].round(4).to_string(index=False))
    json.dump({**info, **{f"phi_min_{n}": c["phi_min"] for n, c in calib.items()},
               **{f"q_static_{n}": c["q_static"] for n, c in calib.items()},
               "default_routing_nominal": default_routing()}, open(OUT / "sim_results.json", "w"), indent=1)


if __name__ == "__main__":
    main()
