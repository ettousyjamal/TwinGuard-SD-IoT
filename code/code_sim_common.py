"""Shared helpers of the simulator experiments (topology, paths, models, normalization, calibration of the gate).

The topology is chosen with the environment variable TG_TOPOLOGY (default: "dataset", the 9-router topology of the
public dataset; also "nsfnet" and "two_tier", see twinguard/sim.py). Data and results of the default topology live
in data/sim and results/sim, those of another topology in data/sim_<name> and results/sim_<name>.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from twinguard import gate as G, model as M, sim, simdata as sd  # noqa: E402

TOPO = os.environ.get("TG_TOPOLOGY", "dataset")
SUFFIX = "" if TOPO == "dataset" else f"_{TOPO}"
DATA, RES = ROOT / "data" / f"sim{SUFFIX}", ROOT / "results" / f"sim{SUFFIX}"
_meta = DATA / "topology.json"
sim.set_topology(TOPO, json.load(open(_meta))["load_scale"] if _meta.exists() else None)

# The simulator twins: the action-conditioned graph-attention twin and its per-link variant, two seeds each
TWINS = {"GNNTwin": (0, 1), "LinkGRU": (0, 1)}


def adjacency():
    A = np.eye(sim.L, dtype=np.float32)
    for i, a in enumerate(sim.LINKS):
        for j, b in enumerate(sim.LINKS):
            if i != j and set(a.split("-")) & set(b.split("-")):
                A[i, j] = 1.0
    return A


def load_ensemble(name, seeds=range(5)):
    ms = []
    for s in seeds:
        ck = torch.load(RES / "models" / f"{name}_s{s}.pt", map_location="cpu", weights_only=False)
        m = M.MODELS[name](11, ck["hp"], adjacency()); m.load_state_dict(ck["state"]); m.eval(); ms.append(m)
    return ms


def load_sim_twin(name="GNNTwin"):
    return load_ensemble(name, TWINS[name])


def norm():
    z = np.load(DATA / "norm.npz")
    n = sd.Normalizer.__new__(sd.Normalizer); n.mu, n.sd = z["mu"], z["sd"]
    return n


def sim_calibration(models, eps=0.05):
    """Calibration of TwinGuard on the simulator's calibration scenarios (cal-A: conformal scores of the first
    half; cal-B: signal references and Phi_min on the second half), with action windows included."""
    z = np.load(DATA / "cal.npz")
    Q, _ = M.predict(models, z["X"])
    d = {"q": Q.mean(0), "sd": Q[..., 0, 1].std(0), "y": z["Y"], "ym": z["Ym"]}
    n = len(z["X"]); half = n // 2
    cut = lambda sl: {k: v[sl] for k, v in d.items()}
    return G.calibrate(cut(slice(0, half)), cut(slice(half, None)), eps=eps)
