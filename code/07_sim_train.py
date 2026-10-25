"""Train the simulator's action-conditioned twins on the exploration data (CPU).

Both models use the architecture and hyperparameters selected on the public dataset (results/best_hyperparameters.json;
no new search), with two seeds each: the action-conditioned graph-attention twin (GNNTwin) and its per-link variant
(LinkGRU). Early stopping on the validation scenarios.
Outputs: results/sim[_<topology>]/models/{GNNTwin,LinkGRU}_s{0,1}.pt and their training logs
Run from the repository root:  [TG_TOPOLOGY=nsfnet] python code/07_sim_train.py
"""
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from code_sim_common import DATA, RES, ROOT, TWINS, M, adjacency  # noqa: E402

OUT = RES / "models"
FIXED = dict(max_epochs=60, patience=8, lr_patience=4, lr_factor=0.5, clip=1.0)


def job(args):
    name, seed = args
    torch.set_num_threads(2)
    path = OUT / f"{name}_s{seed}.pt"
    if path.exists():
        return name, seed, "exists"
    hp = json.load(open(ROOT / "results" / "best_hyperparameters.json"))["best"][name]
    load = lambda s: {k: v for k, v in np.load(DATA / f"{s}.npz").items() if k in ("X", "Y", "Ym", "V")}
    logf = open(OUT / f"{name}_s{seed}.log", "w")
    model, info = M.train(M.MODELS[name], hp, FIXED, load("train"), load("val"), adjacency(), seed,
                          log=lambda m: (logf.write(m + "\n"), logf.flush()))
    torch.save({"state": model.state_dict(), "hp": hp, "fixed": FIXED, **{k: v for k, v in info.items()}}, path)
    return name, seed, f"best epoch {info['best_epoch']}/{info['epochs_run']}, val loss {info['best_val_loss']:.4f}, " \
                       f"{info['seconds'] / 60:.0f} min"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(name, s) for name, seeds in TWINS.items() for s in seeds]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=2) as ex:
        for name, seed, msg in ex.map(job, jobs):
            print(f"[{(time.time() - t0) / 60:5.1f} min] {name} seed {seed}: {msg}", flush=True)


if __name__ == "__main__":
    main()
