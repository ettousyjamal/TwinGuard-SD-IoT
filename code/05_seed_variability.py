"""Variability over training seeds (mean +- standard deviation), for the forecasts and for TwinGuard.

1. Forecasts: every metric of results/forecast_metrics.csv for each of the five single-seed models of the twin and
   of the per-link GRU (same definitions as the experiments notebook), at Delta = 15 s.
2. TwinGuard: the code of notebooks/TwinGuard_gate.ipynb is run five times, each time on a four-member ensemble
   that leaves out one seed; calibration, Phi_min, and all verdicts are recomputed in every run.

Inputs : results/preds/*.npz, data/processed/graph_pipeline.npz, notebooks/TwinGuard_gate.ipynb
Outputs: results/seed_variability_forecast.csv, results/seed_variability_gate.csv, results/seed_variability.json
Run from the repository root:  python code/05_seed_variability.py
"""
import contextlib
import io
import json
import os
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

os.environ.setdefault("MPLBACKEND", "Agg")
warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")   # plt.show() in the notebook
ROOT = Path(__file__).resolve().parents[1]
RES, PREDS = ROOT / "results", ROOT / "results" / "preds"
SPLITS = ["test_D1", "D1_quiet", "D3", "D2"]
N_SEEDS = 5


def auroc(y, s):
    pos = y == 1; n1 = pos.sum(); n0 = len(y) - n1
    return (rankdata(s)[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def forecast_metrics(q_log, p, z, h=0):
    # as forecast_metrics() in the experiments notebook; q_log: [N, L, H, M, 3] log1p quantiles, p: [N, L, H]
    q_ms = np.expm1(q_log[..., 0, :])
    y, ym, v = z["Y"][:, :, h, 0], z["Ym"][:, :, h], z["V"][:, :, h]
    vm = ~np.isnan(v)
    u = y - q_ms[:, :, h, 2]
    pin = np.where(ym, np.maximum(0.95 * u, -0.05 * u), np.nan)
    mae = np.where(ym, np.abs(y - q_ms[:, :, h, 1]), np.nan)
    br = np.where(vm, (p[:, :, h] - np.nan_to_num(v)) ** 2, np.nan)
    return dict(auroc=auroc(v[vm], p[:, :, h][vm]), brier=np.nanmean(br), pinball=np.nanmean(pin), mae=np.nanmean(mae))


def forecast_variability():
    rows = []
    for model in ("GNNTwin", "LinkGRU"):
        for split in SPLITS:
            z = np.load(PREDS / f"{model}_{split}.npz")
            Q, P = z["Q"].astype(np.float32), z["P"].astype(np.float32)
            ens = forecast_metrics(Q.mean(0), P.mean(0), z)
            for s in range(len(Q)):
                rows.append({"model": model, "split": split, "seed": s, **forecast_metrics(Q[s], P[s], z)})
            rows.append({"model": model, "split": split, "seed": "ensemble", **ens})
    df = pd.DataFrame(rows)
    df.to_csv(RES / "seed_variability_forecast.csv", index=False)
    return df


def run_gate(keep, out_dir):
    # executes the gate notebook with the ensemble restricted to the members in `keep`; outputs go to out_dir
    nb = json.loads((ROOT / "notebooks" / "TwinGuard_gate.ipynb").read_text())
    code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    patches = [('ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()', f"ROOT = Path({str(ROOT)!r})"),
               ('OUT = ROOT / "results" / "twinguard"', f"OUT = Path({str(out_dir)!r})"),
               ("FIG_OUT = ROOT ", f"FIG_OUT = Path({str(out_dir)!r}) "),
               ('Q = z["Q"].astype(np.float32)', f'Q = z["Q"].astype(np.float32)[{keep}]'),
               ('z["P"].astype(np.float32).mean(0)', f'z["P"].astype(np.float32)[{keep}].mean(0)')]
    src = "\n\n".join(code)
    for a, b in patches:
        if a.startswith("FIG_OUT") and a not in src:                 # figure folder only in some versions
            continue
        assert a in src, f"patch target not found: {a}"
        src = src.replace(a, b)
    g = {"__name__": "__gate__"}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(src, "TwinGuard_gate.ipynb", "exec"), g)
    ver = pd.read_csv(out_dir / "gate_verdicts.csv")
    cov = pd.read_csv(out_dir / "coverage.csv")
    summ = json.loads((out_dir / "summary.json").read_text())
    return ver, cov, summ


def gate_variability():
    rows = []
    for left_out in range(N_SEEDS):
        keep = [s for s in range(N_SEEDS) if s != left_out]
        with tempfile.TemporaryDirectory() as tmp:
            ver, cov, summ = run_gate(keep, Path(tmp))
        print(f"  gate run without seed {left_out}: Phi_min = {summ['phi_min']:.3f}", flush=True)
        for _, r in ver.iterrows():
            rows.append({"left_out": left_out, "kind": "verdict", **r.to_dict()})
        for _, r in cov.iterrows():
            rows.append({"left_out": left_out, "kind": "coverage", **r.to_dict()})
        rows.append({"left_out": left_out, "kind": "summary", "phi_min": summ["phi_min"],
                     **{k: v for k, v in summ.items() if isinstance(v, (int, float)) and k != "phi_min"}})
    df = pd.DataFrame(rows)
    df.to_csv(RES / "seed_variability_gate.csv", index=False)
    return df


if __name__ == "__main__":
    fc = forecast_variability()
    single = fc[fc.seed != "ensemble"]
    agg = single.groupby(["model", "split"])[["auroc", "brier", "pinball", "mae"]].agg(["mean", "std"])
    print(agg.round(4).to_string())
    print("\nensemble:\n", fc[fc.seed == "ensemble"].round(4).to_string(index=False))
    print("\nTwinGuard, leave-one-seed-out ensembles:")
    gate = gate_variability()
    stats = {"forecast": {f"{m}|{s}": {k: [float(agg.loc[(m, s), (k, "mean")]), float(agg.loc[(m, s), (k, "std")])]
                                       for k in ["auroc", "brier", "pinball", "mae"]} for m, s in agg.index}}
    (RES / "seed_variability.json").write_text(json.dumps(stats, indent=1))
