"""Simulator data for the action-conditioned twin: scenarios run under a random exploration policy.

Disjoint seeds per split. train/val/cal use the training regime (storms, mild capacity degradations); the test
scenarios of the closed-loop and counterfactual experiments are generated in their own scripts. For a topology
other than the dataset's (TG_TOPOLOGY), the traffic volume is first scaled so that static shortest-path routing
violates the SLA on about 3% of the link-epochs, as on the dataset topology (3.1%).
Outputs: data/sim[_<topology>]/{train,val,cal}.npz (windows, targets), norm.npz, and topology.json
Run from the repository root:  [TG_TOPOLOGY=nsfnet] python code/06_sim_data.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from code_sim_common import DATA as OUT, TOPO, sim, sd  # noqa: E402

SPLITS = {"train": (range(100, 112), 8.0), "val": (range(200, 203), 6.0), "cal": (range(300, 304), 6.0)}
if TOPO != "dataset":                            # other topologies: 8 training scenarios (CPU budget)
    SPLITS["train"] = (range(100, 108), 8.0)
TARGET = 0.031                                   # static-routing violation rate of the dataset topology


def static_violations(scale, seeds=range(800, 804), hours=6.0):
    sim.LOAD_SCALE = scale; V = []
    for s in seeds:
        ex = sim.make_exo(s, hours, "test")
        V.append(sim.violation(sim.Network(ex).run(lambda k: sim.Config(), 0, ex.rates.shape[1] // sim.STEP)[0]))
    return float(np.nanmean(np.concatenate(V)))


def calibrate_scale(lo=0.5, hi=4.0, iters=9):
    for _ in range(iters):                       # violations increase with the traffic volume: bisection
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if static_violations(mid) < TARGET else (lo, mid)
    return (lo + hi) / 2


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    scale = sim.LOAD_SCALE if TOPO == "dataset" else calibrate_scale()
    sim.LOAD_SCALE = scale
    json.dump({"topology": TOPO, "load_scale": scale, "routers": sim.G.number_of_nodes(), "links": sim.L,
               "static_violation_rate": static_violations(scale)}, open(OUT / "topology.json", "w"), indent=1)
    print(open(OUT / "topology.json").read(), flush=True)
    runs = {}
    for name, (seeds, hours) in SPLITS.items():
        t0 = time.time()
        runs[name] = [sd.explore(sim.make_exo(s, hours, "train"), seed=s + 7) for s in seeds]
        print(f"{name}: {len(seeds)} scenarios x {hours} h simulated in {time.time() - t0:.0f} s", flush=True)
    norm = sd.Normalizer(np.concatenate([r["F"] for r in runs["train"]]))
    np.savez(OUT / "norm.npz", mu=norm.mu, sd=norm.sd)
    for name, rs in runs.items():
        ws = [sd.windows(r, norm) for r in rs]
        cat = {k: np.concatenate([w[k] for w in ws]) for k in ("X", "Y", "Ym", "V", "acted")}
        cat["scenario"] = np.concatenate([np.full(len(w["X"]), i) for i, w in enumerate(ws)])
        np.savez_compressed(OUT / f"{name}.npz", **cat)
        v = cat["V"][:, :, 0]
        print(f"{name}: {len(cat['X'])} windows, {100 * cat['acted'].mean():.1f}% with an action, "
              f"SLA violations {100 * np.nanmean(v):.2f}%, measured {100 * cat['Ym'][:, :, 0].mean():.1f}%", flush=True)


if __name__ == "__main__":
    main()
