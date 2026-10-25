"""Result figures of the simulator experiments and of the online recalibration on the public dataset.

fig_recal   : fidelity score on D3 and D2, as published and with online recalibration (bound and references)
fig_effects : predicted vs. true change of link RTT caused by an action (counterfactual, simulator)
fig_closed  : closed-loop SLA violations and actions per hour of the controllers (simulator)
Run from the repository root:
  python code/12_result_figures.py effects=results/figs/fig_action_effects.png closed=results/figs/fig_closed_loop.png \
      recal=results/figs/fig_recalibration.png
"""
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from twinguard import realdata as rd, sim  # noqa: E402

PAPER_W, DPI = 4.75, 600
mpl.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "legend.fontsize": 8, "legend.frameon": False, "savefig.dpi": DPI, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False, "mathtext.fontset": "cm",
})


def median15(t_h, x):
    """15-min medians (60 epochs) for readability; gaps longer than 1 h are left empty."""
    bins = np.floor(t_h * 4) / 4
    df = pd.DataFrame({"b": bins, "x": x}).groupby("b").x.median()
    return df.index.to_numpy(), df.to_numpy()


def fig_recal(out):
    z = np.load(ROOT / "results" / "twinguard" / "recalibration_phi.npz")
    phi_min = float(z["phi_min"])
    fig, axes = plt.subplots(2, 1, figsize=(PAPER_W, 3.6), sharey=True)
    for ax, split, title in zip(axes, ["D3", "D2"], ["(a) D3, moderate shift", "(b) D2, severe shift"]):
        ts = rd.meta(split)["ts"]; t_h = (ts - ts[0]) / 3600
        for key, lab, c, lw in [("phi", "as published", "0.55", 0.9), ("phi_refs", "online recalibration", "#0072B2", 1.1)]:
            x, y = median15(t_h, z[f"{split}/{key}"])
            ax.plot(x, y, color=c, lw=lw, label=lab)
        ax.axhline(phi_min, color="#b22222", ls="--", lw=0.9, label=r"$\Phi_{\min}$")
        ax.set_title(title, loc="left", fontsize=8); ax.set_ylim(-0.03, 1.03)
        ax.set_ylabel(r"Fidelity $\Phi_t$"); ax.grid(True, color="0.9", lw=0.6); ax.set_axisbelow(True)
    axes[-1].set_xlabel("Hours since the start of the scenario")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(h_pad=1.2)
    fig.savefig(ROOT / out); plt.close(fig)


def fig_effects(out):
    z = dict(np.load(ROOT / "results" / "sim" / "counterfactual.npz", allow_pickle=True))
    fig, axes = plt.subplots(2, 2, figsize=(PAPER_W, 4.6), sharex=True, sharey=True)
    for row, (key, name) in enumerate([("twin", "GNN twin"), ("link", "per-link twin")]):
        for col, (regime, rt) in enumerate([("nominal", "nominal"), ("shift", "shift")]):
            ax = axes[row, col]; r = z["regime"] == regime; T, P = [], []
            for e in np.unique(z["epoch_id"][r]):
                idx = np.where(z["epoch_id"] == e)[0]; n0 = idx[z["kind"][idx] == "noop"][0]
                for i in idx[z["kind"][idx] != "noop"]:
                    both = z["ym"][i, :, 0] & z["ym"][n0, :, 0]
                    T.append((np.log1p(z["y"][i, :, 0, 0]) - np.log1p(z["y"][n0, :, 0, 0]))[both])
                    P.append((z[key + "_q"][i, :, 0, 0, 1] - z[key + "_q"][n0, :, 0, 0, 1])[both])
            T, P = np.concatenate(T), np.concatenate(P)
            ax.hexbin(T, P, gridsize=45, bins="log", cmap="Blues", mincnt=1, extent=(-1.6, 1.6, -1.6, 1.6), linewidths=0)
            ax.plot([-1.6, 1.6], [-1.6, 1.6], color="0.35", lw=0.8, ls="--")
            ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.6, 1.6); ax.set_aspect("equal")
            ax.set_title(f"({'abcd'[2 * row + col]}) {name[0].upper() + name[1:]}, {rt}", loc="left", fontsize=7.5)
            if row == 1:
                ax.set_xlabel(r"True change of $\log(1+\mathrm{RTT})$")
            if col == 0:
                ax.set_ylabel("Predicted change")
    fig.tight_layout(w_pad=1.5, h_pad=1.2)
    fig.savefig(ROOT / out); plt.close(fig)


ORDER = ["static", "reactive", "analytical what-if", "twin, ungated (B3)", "twin, uncalibrated gate (B4)",
         "TwinGuard", "TwinGuard, improved fallback", "TwinGuard, improved fallback, per-link twin", "oracle"]
LABELS = {"static": "Static routing", "reactive": "Reactive rule", "analytical what-if": "Queueing what-if",
          "twin, ungated (B3)": "Twin, ungated", "twin, uncalibrated gate (B4)": "Twin, uncalibrated gate",
          "TwinGuard": r"TwinGuard ($\pi_{\mathrm{safe}}$)", "TwinGuard, improved fallback": r"TwinGuard ($\pi_{\mathrm{safe}}^{+}$)",
          "TwinGuard, improved fallback, per-link twin": r"TwinGuard ($\pi_{\mathrm{safe}}^{+}$), per-link twin",
          "oracle": "Oracle"}


def fig_closed(out):
    cl = pd.read_csv(ROOT / "results" / "sim" / "closed_loop.csv"); cl = cl[cl.missing == "nominal"]
    fig, axes = plt.subplots(1, 2, figsize=(PAPER_W, 2.9), sharey=True)
    y = np.arange(len(ORDER))[::-1]
    for ax, regime, title in zip(axes, ["nominal", "shift"], ["(a) Nominal regime", "(b) Shift regime"]):
        g = cl[cl.regime == regime].groupby("controller").violation_rate
        m, s = g.mean().reindex(ORDER) * 100, g.std().reindex(ORDER) * 100
        cols = ["#0072B2" if "TwinGuard" in c else ("0.3" if c == "oracle" else "0.65") for c in ORDER]
        ax.barh(y, m, xerr=s, color=cols, height=0.62, error_kw=dict(lw=0.8, capsize=2), zorder=3)
        ax.set_title(title, loc="left", fontsize=8); ax.set_xlim(0, 6.6); ax.set_xticks([0, 2, 4, 6])
        ax.grid(True, axis="x", color="0.9", lw=0.6); ax.set_axisbelow(True)
    axes[0].set_yticks(y); axes[0].set_yticklabels([LABELS[c] for c in ORDER], fontsize=7.5)
    fig.supxlabel("SLA violations (% of measured link-epochs; mean and standard deviation over 4 scenarios)",
                  fontsize=8, y=0.02)
    fig.tight_layout(w_pad=1.5)
    fig.savefig(ROOT / out); plt.close(fig)


if __name__ == "__main__":
    args = dict(a.split("=") for a in sys.argv[1:])
    if "recal" in args:
        fig_recal(args["recal"])
    if "effects" in args:
        fig_effects(args["effects"])
    if "closed" in args:
        fig_closed(args["closed"])
    print("written:", ", ".join(args.values()))
