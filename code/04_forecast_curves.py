"""Forecast quality of the GNN twin from the saved predictions (no retraining).

fig_roc_pr.png      : ROC and precision-recall curves of violation prediction on D1 test, Delta = 15 s.
fig_reliability.png : reliability diagram of the twin's violation probability on D1 test, D3, and D2.
fig_horizon_gap.png : AUROC difference twin - per-link GRU against the horizon, with paired block-bootstrap
                      95% intervals (both models scored on the same resampled windows).

Inputs : results/preds/{GNNTwin,LinkGRU}_{split}.npz   (experiments notebook, ensemble members)
         data/processed/graph_pipeline.npz             (persistence baseline: last observed violation state)
Outputs: results/figs/fig_{roc_pr,reliability,horizon_gap}.png and results/model_curves.json
Run from the repository root:  python code/04_forecast_curves.py
"""
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
RES, PREDS = ROOT / "results", ROOT / "results" / "preds"
FIGS = RES / "figs"
PAPER_W, DPI = 4.75, 600
RTT_MAX, PLR_MAX = 10.0, 1.0
HORIZONS_S = [15, 30, 60]
N_BOOT, BLOCK = 300, 40                      # as for the confidence intervals in forecast_metrics.csv
COLORS = {"GNNTwin": "#0072B2", "LinkGRU": "#D55E00"}
LABELS = {"GNNTwin": "GNN twin", "LinkGRU": "Per-link GRU (B2)"}
SPLIT_LABELS = {"test_D1": "D1 test", "D1_quiet": "D1 quiet", "D3": "D3, moderate shift", "D2": "D2, severe shift"}
mpl.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "legend.fontsize": 8, "legend.frameon": False, "savefig.dpi": DPI, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False, "mathtext.fontset": "cm",
})


def load(model, split):
    # ensemble mean of the violation probability, [N, L, H], and the violation labels (NaN = not measured)
    z = np.load(PREDS / f"{model}_{split}.npz")
    return z["P"].astype(np.float32).mean(0), z["V"].astype(np.float32)


def persistence(split):
    # last observed violation state at the end of each window, [N, L] (as in the experiments notebook)
    z = np.load(ROOT / "data" / "processed" / "graph_pipeline.npz")
    out, i = [], 0
    while f"{split}_V_{i}" in z.files:
        raw = z[f"{split}_V_{i}"][..., :3]                      # rtt, jitter, packet_loss; NaN when missing
        ff = pd.DataFrame(raw.reshape(len(raw), -1)).ffill().to_numpy().reshape(raw.shape)
        lv = ((ff[..., 0] > RTT_MAX) | (ff[..., 2] > PLR_MAX)).astype(np.float32)
        out.append(lv[z[f"{split}_t_end_{i}"]]); i += 1
    return np.concatenate(out)


def auroc(y, s):
    pos = y == 1; n1 = pos.sum(); n0 = len(y) - n1
    r = rankdata(s)
    return (r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def roc_pr(y, s):
    o = np.argsort(-s, kind="mergesort"); y, s = y[o], s[o]
    last = np.r_[np.where(np.diff(s))[0], len(s) - 1]           # one point per distinct threshold
    tp = np.cumsum(y)[last]; fp = (last + 1) - tp
    tpr, fpr, prec = tp / y.sum(), fp / (len(y) - y.sum()), tp / (last + 1)
    ap = np.sum(np.diff(np.r_[0, tpr]) * prec)                  # average precision (step-wise PR-AUC)
    return np.r_[0, fpr], np.r_[0, tpr], tpr, prec, ap


def flat(p, v, h):
    m = ~np.isnan(v[:, :, h])
    return v[:, :, h][m], p[:, :, h][m]


def figure12(stats):
    fig, (a, b) = plt.subplots(1, 2, figsize=(PAPER_W, 2.5))
    for split in ("test_D1", "D3"):
        for m in ("GNNTwin", "LinkGRU"):
            y, s = flat(*load(m, split), 0)
            fpr, tpr, rec, prec, ap = roc_pr(y, s)
            stats.setdefault("auprc", {})[f"{m}_{split}"] = round(float(ap), 3)
            stats.setdefault("prevalence", {})[split] = round(float(y.mean()), 4)
            k = np.searchsorted(fpr, 0.05, side="right") - 1          # operating point at 5% false alarms
            stats.setdefault("tpr_at_fpr05", {})[f"{m}_{split}"] = round(float(tpr[k]), 3)
            if split == "test_D1":
                a.plot(fpr, tpr, color=COLORS[m], lw=1.3, label=LABELS[m])
                b.plot(rec, prec, color=COLORS[m], lw=1.1)
        _, v = load("GNNTwin", split)
        pv = persistence(split)[:, :, None].repeat(v.shape[2], 2)
        y, s = flat(pv, v, 0)
        tpr_p, fpr_p, prec_p = s[y == 1].mean(), s[y == 0].mean(), y[s == 1].mean()
        stats.setdefault("persistence", {})[split] = dict(tpr=round(float(tpr_p), 3), fpr=round(float(fpr_p), 3),
                                                          precision=round(float(prec_p), 3))
        if split == "test_D1":
            a.plot(fpr_p, tpr_p, "o", color="0.3", ms=5, zorder=5, label="Persistence (B1)")
            b.plot(tpr_p, prec_p, "o", color="0.3", ms=5, zorder=5)
    a.plot([0, 1], [0, 1], color="0.7", lw=0.8, ls=":", label="random")
    a.set_xlim(0, 0.3); a.set_ylim(0, 1); a.set_xticks([0, 0.1, 0.2, 0.3])
    b.set_xlim(0, 1); b.set_ylim(0, 1)
    a.set_xlabel("False-positive rate"); a.set_ylabel("True-positive rate")
    b.set_xlabel("Recall"); b.set_ylabel("Precision")
    for ax, t in [(a, "(a) ROC"), (b, "(b) Precision-recall")]:
        ax.set_title(t, loc="left", fontsize=8)
        ax.grid(True, color="0.9", lw=0.6); ax.set_axisbelow(True)
    h, l = a.get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.08), columnspacing=1.2,
               handlelength=1.6, fontsize=7.5)
    fig.tight_layout(w_pad=2.4)
    fig.savefig(FIGS / "fig_roc_pr.png"); plt.close(fig)


def figure13(stats):
    # log-spaced bins of the predicted probability; log axes because most probabilities are close to 0
    fig = plt.figure(figsize=(0.88 * PAPER_W, 2.9))
    ax = fig.add_axes([0.13, 0.15, 0.52, 0.80])           # square axes; legend on the right
    styles = {"test_D1": ("#0072B2", "o"), "D3": ("#E69F00", "s"), "D2": ("#CC3311", "^")}
    lo = 3e-4
    for split, (c, mk) in styles.items():
        y, s = flat(*load("GNNTwin", split), 0)
        edges = np.array([0, 0.003, 0.006, 0.01, 0.02, 0.04, 0.08, 0.15, 0.3, 1.0])
        b = np.clip(np.searchsorted(edges, s, side="right") - 1, 0, len(edges) - 2)
        n = np.array([(b == i).sum() for i in range(len(edges) - 1)])
        pm = np.array([s[b == i].mean() if n[i] else np.nan for i in range(len(edges) - 1)])
        om = np.array([y[b == i].mean() if n[i] else np.nan for i in range(len(edges) - 1)])
        ece = float(np.nansum(n * np.abs(pm - om)) / n.sum())
        stats.setdefault("ece", {})[split] = round(ece, 4)
        stats.setdefault("mean_prob_vs_rate", {})[split] = dict(mean_prob=round(float(s.mean()), 4),
                                                                rate=round(float(y.mean()), 4))
        keep = (n >= 50) & (om > 0)                                          # bins without violations cannot be drawn on log axes
        ax.plot(pm[keep], om[keep], "-" + mk, color=c, ms=3.8, lw=1.1,
                mfc=c, mec="white", mew=0.5, label=SPLIT_LABELS[split])
    ax.plot([lo, 1], [lo, 1], color="0.6", lw=0.8, ls="--", label="perfect calibration")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo, 1); ax.set_ylim(lo, 1); ax.set_aspect("equal")
    ax.set_xlabel("Predicted violation probability")
    ax.set_ylabel("Observed violation frequency")
    ax.grid(True, which="major", color="0.9", lw=0.6); ax.set_axisbelow(True)
    ax.legend(loc="center left", bbox_to_anchor=(1.03, 0.5), fontsize=7.5)
    fig.savefig(FIGS / "fig_reliability.png"); plt.close(fig)


def figure14(stats):
    # paired block bootstrap over windows: both models are scored on the same resampled windows
    fig, ax = plt.subplots(figsize=(0.7 * PAPER_W, 2.3))
    styles = {"test_D1": ("#0072B2", "o", -1.2), "D3": ("#E69F00", "s", 0.0), "D2": ("#CC3311", "^", 1.2)}
    rng = np.random.default_rng(0)
    for split, (c, mk, dx) in styles.items():
        pt, v = load("GNNTwin", split); pg, _ = load("LinkGRU", split)
        N = len(v); nb = int(np.ceil(N / BLOCK))
        idxs = []
        for _ in range(N_BOOT):
            starts = rng.integers(0, max(1, N - BLOCK), nb)
            idxs.append(np.concatenate([np.arange(s, min(s + BLOCK, N)) for s in starts])[:N])
        d, lo, hi = [], [], []
        for h in range(len(HORIZONS_S)):
            def gap(ix):
                y, st = flat(pt[ix], v[ix], h); _, sg = flat(pg[ix], v[ix], h)
                return auroc(y, st) - auroc(y, sg)
            g = gap(np.arange(N)); bs = np.array([gap(ix) for ix in idxs])
            l, u = np.percentile(bs, [2.5, 97.5])
            d.append(g); lo.append(l); hi.append(u)
            stats.setdefault("auroc_gap", {})[f"{split}_{HORIZONS_S[h]}s"] = dict(
                gap=round(float(g), 4), lo=round(float(l), 4), hi=round(float(u), 4),
                p_le_0=round(float((bs <= 0).mean()), 4))
        x = np.array(HORIZONS_S) + dx
        d, lo, hi = map(np.array, (d, lo, hi))
        ax.errorbar(x, d, yerr=[d - lo, hi - d], fmt="-" + mk, color=c, ms=4, lw=1.1, capsize=2.5,
                    elinewidth=0.9, mfc=c, mec="white", mew=0.5, label=SPLIT_LABELS[split])
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set_xticks(HORIZONS_S); ax.set_xlim(8, 67)
    ax.set_xlabel("Prediction horizon (s)")
    ax.set_ylabel("AUROC, twin minus GRU")
    ax.grid(True, axis="y", color="0.9", lw=0.6); ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_horizon_gap.png"); plt.close(fig)


if __name__ == "__main__":
    FIGS.mkdir(parents=True, exist_ok=True)
    stats = {}
    figure12(stats); figure13(stats); figure14(stats)
    (RES / "model_curves.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))
