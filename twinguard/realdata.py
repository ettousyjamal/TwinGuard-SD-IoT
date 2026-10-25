"""Loading of the public dataset's test sets in the convention of twinguard.gate (as in the gate notebook)."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TEST_SPLITS = ["test_D1", "D1_quiet", "D3", "D2"]
RTT_MAX, PLR_MAX = 10.0, 1.0


def meta(split, pipe=None):
    # timestamps of the decision epochs and the last measured SLA state (fallback / reactive rule)
    pipe = np.load(ROOT / "data" / "processed" / "graph_pipeline.npz") if pipe is None else pipe
    ts, cur_v, k = [], [], 0
    while f"{split}_t_end_{k}" in pipe.files:
        t_end, T, V = pipe[f"{split}_t_end_{k}"], pipe[f"{split}_ts_{k}"], pipe[f"{split}_V_{k}"]
        raw = V[..., [0, 2]]
        ff = pd.DataFrame(raw.reshape(len(raw), -1)).ffill().to_numpy().reshape(raw.shape)
        state = (ff[..., 0] > RTT_MAX) | (ff[..., 1] > PLR_MAX)
        ts.append(T[t_end]); cur_v.append(state[t_end]); k += 1
    return {"ts": np.concatenate(ts), "cur_v": np.concatenate(cur_v)}


def from_preds(Q, y, ym, v, m):
    """Q: [S,N,L,H,M,3] member quantiles (log1p) -> test-set dict of twinguard.gate."""
    Q = Q.astype(np.float32)
    return {"q": Q.mean(0), "sd": Q[..., 0, 1].std(0), "y": y, "ym": ym, "v": v, **m}


def load(split, model="GNNTwin", members=None, pipe=None):
    z = np.load(ROOT / "results" / "preds" / f"{model}_{split}.npz")
    Q = z["Q"].astype(np.float32)
    if members is not None:
        Q = Q[members]
    return from_preds(Q, z["Y"], z["Ym"], z["V"], meta(split, pipe))


def cal_halves(cal):
    n = len(cal["y"]); half = n // 2
    cut = lambda sl: {k: (v[sl] if isinstance(v, np.ndarray) and len(v) == n else v) for k, v in cal.items()}
    return cut(slice(0, half)), cut(slice(half, None))
