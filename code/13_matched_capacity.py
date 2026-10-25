"""GNN twin against the per-link GRU at equal capacity, on the public dataset.

The selected models differ in size (78,454 and 34,038 parameters). Here each architecture is also trained with the
other's selected configuration, which gives two pairs of nearly equal size: GNN twin 78,454 vs per-link GRU 78,070
(GNN configuration) and GNN twin 34,166 vs per-link GRU 34,038 (GRU configuration). Same windows, splits, seeds,
training protocol, and metrics as the published models; the published checkpoints provide the other two members
of each pair. A paired block bootstrap compares the AUROC within each pair on the same windows.
Outputs: results/matched_capacity/{metrics.csv, paired_auroc.csv, models/*.pt}
Run from the repository root:  python code/13_matched_capacity.py
"""
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from twinguard import model as M  # noqa: E402

OUT = ROOT / "results" / "matched_capacity"; MOD = OUT / "models"
SEEDS = range(5)
SPLITS = ["test_D1", "D1_quiet", "D3", "D2"]
HP = json.load(open(ROOT / "results" / "best_hyperparameters.json"))
# (label, architecture, configuration) of the four models; the first two are the published ones
VARIANTS = [("GNN twin, GNN config.", "GNNTwin", "GNNTwin"), ("GRU, GRU config.", "LinkGRU", "LinkGRU"),
            ("GRU, GNN config.", "LinkGRU", "GNNTwin"), ("GNN twin, GRU config.", "GNNTwin", "LinkGRU")]
PAIRS = [("GNN twin, GNN config.", "GRU, GNN config."), ("GNN twin, GRU config.", "GRU, GRU config.")]


def windows(pipe, split):
    X, Y, Ym, V, k = [], [], [], [], 0
    while f"{split}_t_end_{k}" in pipe.files:
        Xs, idx = pipe[f"{split}_X_{k}"], pipe[f"{split}_t_end_{k}"]; W = int(pipe["W"])
        X.append(np.stack([Xs[t - W + 1:t + 1] for t in idx]))
        Y.append(pipe[f"{split}_Y_{k}"].transpose(0, 2, 1, 3)); Ym.append(pipe[f"{split}_Ym_{k}"].transpose(0, 2, 1))
        V.append(pipe[f"{split}_vio_{k}"].transpose(0, 2, 1)); k += 1
    return {"X": np.concatenate(X), "Y": np.concatenate(Y), "Ym": np.concatenate(Ym), "V": np.concatenate(V)}


def path(arch, cfg, seed):
    if arch == cfg:
        return ROOT / "results" / "models" / f"{arch}_s{seed}.pt"
    return MOD / f"{arch}_{cfg}config_s{seed}.pt"


def job(args):
    arch, cfg, seed = args
    torch.set_num_threads(2)
    p = path(arch, cfg, seed)
    if p.exists():
        return arch, cfg, seed, "exists"
    pipe = np.load(ROOT / "data" / "processed" / "graph_pipeline.npz")
    tr = windows(pipe, "train"); cut = int(0.85 * len(tr["X"]))
    fit = {k: v[:cut] for k, v in tr.items()}; val = {k: v[cut:] for k, v in tr.items()}
    model, info = M.train(M.MODELS[arch], HP["best"][cfg], HP["fixed"], fit, val, pipe["A"], seed)
    torch.save({"state": model.state_dict(), "hp": HP["best"][cfg], **info}, p)
    return arch, cfg, seed, f"best epoch {info['best_epoch']}/{info['epochs_run']}, val loss " \
                            f"{info['best_val_loss']:.4f}, {info['seconds'] / 60:.0f} min"


def load(arch, cfg, pipe):
    ms = []
    for s in SEEDS:
        ck = torch.load(path(arch, cfg, s), map_location="cpu", weights_only=False)
        m = M.MODELS[arch](10, HP["best"][cfg], pipe["A"]); m.load_state_dict(ck["state"]); m.eval(); ms.append(m)
    return ms


def scores(q_ms, p, d, h):
    y, ym, v = d["Y"][:, :, h, 0], d["Ym"][:, :, h], d["V"][:, :, h]; vm = ~np.isnan(v)
    u = y - q_ms[:, :, h, 2]
    return {"auroc": M.auroc(v[vm], p[:, :, h][vm]),
            "brier": float(np.nanmean(np.where(vm, (p[:, :, h] - np.nan_to_num(v)) ** 2, np.nan))),
            "pinball": float(np.nanmean(np.where(ym, np.maximum(0.95 * u, -0.05 * u), np.nan))),
            "mae": float(np.nanmean(np.where(ym, np.abs(y - q_ms[:, :, h, 1]), np.nan)))}


def paired_auroc(pa, pb, v, n_boot=300, block=40, seed=0):
    """pa, pb, v: [N windows, L links]; blocks of consecutive windows (all links), as in the paper."""
    rng = np.random.default_rng(seed); N = len(v); nb = int(np.ceil(N / block)); out = []
    for _ in range(n_boot):
        idx = np.concatenate([np.arange(s, min(s + block, N)) for s in rng.integers(0, N - block, nb)])[:N]
        vv = v[idx].ravel(); m = ~np.isnan(vv)
        out.append(M.auroc(vv[m], pa[idx].ravel()[m]) - M.auroc(vv[m], pb[idx].ravel()[m]))
    v, pa, pb = v.ravel(), pa.ravel(), pb.ravel(); m = ~np.isnan(v)
    return M.auroc(v[m], pa[m]) - M.auroc(v[m], pb[m]), np.percentile(out, [2.5, 97.5])


def main():
    MOD.mkdir(parents=True, exist_ok=True)
    jobs = [(a, c, s) for _, a, c in VARIANTS if a != c for s in SEEDS]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=2) as ex:
        for a, c, s, msg in ex.map(job, jobs):
            print(f"[{(time.time() - t0) / 60:5.1f} min] {a} ({c} configuration) seed {s}: {msg}", flush=True)
    pipe = np.load(ROOT / "data" / "processed" / "graph_pipeline.npz")
    torch.set_num_threads(4)
    rows, pair_rows = [], []
    for split in SPLITS:
        d = windows(pipe, split); P = {}
        for label, a, c in VARIANTS:
            ms = load(a, c, pipe); Q, Pr = M.predict(ms, d["X"]); P[label] = Pr.mean(0)
            n_par = sum(p.numel() for p in ms[0].parameters())
            for h, hs in [(0, 15), (2, 60)]:
                ens = scores(np.expm1(Q.mean(0)[..., 0, :]), Pr.mean(0), d, h)
                per = [scores(np.expm1(Q[s][..., 0, :]), Pr[s], d, h) for s in range(len(ms))]
                rows.append({"split": split, "horizon_s": hs, "model": label, "parameters": n_par, **ens,
                             **{f"{k}_sd": float(np.std([p[k] for p in per])) for k in ens}})
        for a, b in PAIRS:
            for h, hs in [(0, 15), (2, 60)]:
                diff, (lo, hi) = paired_auroc(P[a][:, :, h], P[b][:, :, h], d["V"][:, :, h])
                pair_rows.append({"split": split, "horizon_s": hs, "pair": f"{a} - {b}", "auroc_diff": diff,
                                  "lo": lo, "hi": hi})
        print(f"{split}: done", flush=True)
    df = pd.DataFrame(rows); df.to_csv(OUT / "metrics.csv", index=False)
    pr = pd.DataFrame(pair_rows); pr.to_csv(OUT / "paired_auroc.csv", index=False)
    pd.set_option("display.width", 250)
    print(df.round(4).to_string(index=False)); print(pr.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
