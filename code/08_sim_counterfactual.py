"""Can the twin predict the consequences of different actions? Counterfactual evaluation on the simulator.

Unseen test scenarios (nominal regime and shift regime) are run under the exploration policy; at sampled decision
epochs the candidate actions of the Planning agent plus random actions and the no-op are each simulated 60 s ahead
from the same network state with the same random draws (exact counterfactual outcomes). The predictions of the
action-conditioned graph-attention twin, its per-link variant, the white-box queueing model, and persistence are
scored against them. With the argument "cal", the same forks are collected on the calibration scenarios (training
law), which provides calibration scores of the candidates the Planning agent actually proposes.
Outputs: results/sim[_<topology>]/counterfactual.npz (or counterfactual_cal.npz)
Run from the repository root:  [TG_TOPOLOGY=nsfnet] python code/08_sim_counterfactual.py [cal]
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from code_sim_common import RES as OUT, TWINS, load_sim_twin, norm, sim, sd  # noqa: E402
from twinguard import agents as ag  # noqa: E402

SCEN = {"nominal": (range(400, 404), "test"), "shift": (range(500, 504), "shift")}
SCEN_CAL = {"cal": (range(300, 304), "train")}
HOURS, EVERY, N_RANDOM = 6.0, 6, 3
H_IDX = {1: 0, 2: 1, 4: 2}               # horizon (epochs) -> index in the model output
PREFIX = {"GNNTwin": "twin", "LinkGRU": "link"}


def collect(exo, seed, twins):
    """Run the scenario under the exploration policy; at decision epochs, fork every candidate action."""
    rng = np.random.default_rng(seed); net = sim.Network(exo); E = net.E
    cfg = sim.Config(); state = None; hist = []; last = -4; recs = []
    for k in range(E):
        o, state = net.run(lambda _k: cfg, k, k + 1, state)
        F_t = sd.raw_features(o, [cfg], exo, k)[0]; hist.append(F_t)
        sl = slice(k * sim.STEP, (k + 1) * sim.STEP); rates = exo.rates[:, sl].mean(1)
        failed = exo.capf[:, sl].min(1) <= 0
        if k >= 2 * sd.W and k + 4 < E and (k % EVERY == 0):
            util = sd.controller_loads(cfg, rates, failed).sum(0) / sim.C_NOM
            alerts = list(np.argsort(-util)[:2])
            planned = ag.plan(cfg, alerts, rates, failed)
            acts = [sd.NOOP] + planned
            pool = [a for a in sd.all_actions(cfg) if a not in acts]
            rand = [pool[i] for i in rng.choice(len(pool), min(N_RANDOM, len(pool)), replace=False)]
            acts += rand
            src = ["noop"] + ["planner"] * len(planned) + ["random"] * len(rand)
            cfgs = [sd.apply(cfg, a) for a in acts]
            y = np.zeros((len(acts), sim.L, 3, 3)); ym = np.zeros((len(acts), sim.L, 3), bool)
            for i, c in enumerate(cfgs):
                oc, _ = net.run(lambda _k: c, k + 1, k + 5, state)
                for h, j in H_IDX.items():
                    y[i, :, j] = np.stack([oc["rtt"][h - 1], oc["jitter"][h - 1], oc["loss"][h - 1]], -1)
                    ym[i, :, j] = oc["measured"][h - 1]
            rec = {"k": k, "kinds": [a.kind for a in acts], "src": src, "y": y, "ym": ym,
                   "costs": [sd.cost(cfg, a, sd.controller_loads(cfg, rates, failed)[2]) for a in acts]}
            for name, tw in twins.items():
                p = tw.predict(tw.windows(hist[:-1], F_t, cfgs, rates, failed))
                rec[name + "_q"] = p["q"]; rec[name + "_sd"] = p["sd"]; rec[name + "_p"] = p["p"]
            ana = ag.Analytical().predict_raw(hist[:-1], F_t, cfgs, rates, failed)
            rec["ana_rtt"] = np.stack([a[0] for a in ana]); rec["ana_loss"] = np.stack([a[1] for a in ana])
            rec["persist"] = np.stack([F_t[:, 0], F_t[:, 2]], -1)          # last measurement, NaN if missing
            recs.append(rec)
        # exploration policy (as in the training data), so that configurations vary
        a = sd.NOOP
        if k - last >= 4 and rng.uniform() < 0.2:
            cand = sd.all_actions(cfg); a = cand[int(rng.integers(len(cand)))]; last = k
        cfg = sd.apply(cfg, a)
    return recs


def job(args):
    regime, s, law = args
    torch.set_num_threads(2)
    twins = {PREFIX[n]: ag.Twin(load_sim_twin(n), norm()) for n in TWINS}
    return regime, s, collect(sim.make_exo(s, HOURS, law), s, twins)


def main():
    cal = len(sys.argv) > 1 and sys.argv[1] == "cal"
    scen = SCEN_CAL if cal else SCEN
    keys = ["regime", "epoch_id", "kind", "src", "cost", "y", "ym", "ana_rtt", "ana_loss", "persist"] + \
        [f"{p}_{x}" for p in PREFIX.values() for x in ("q", "sd", "p")]
    rows = {k: [] for k in keys}; eid = 0
    jobs = [(regime, s, law) for regime, (seeds, law) in scen.items() for s in seeds]
    with ProcessPoolExecutor(max_workers=4) as ex:
        for regime, s, recs in ex.map(job, jobs):
            for r in recs:
                for i, kind in enumerate(r["kinds"]):
                    rows["regime"].append(regime); rows["epoch_id"].append(eid); rows["kind"].append(kind)
                    rows["src"].append(r["src"][i]); rows["cost"].append(r["costs"][i])
                    rows["y"].append(r["y"][i]); rows["ym"].append(r["ym"][i])
                    for key in keys[7:]:
                        rows[key].append(r[key][i] if key != "persist" else r["persist"])
                eid += 1
            print(f"{regime} seed {s}: {len(recs)} decision epochs, {eid} in total", flush=True)
    name = "counterfactual_cal.npz" if cal else "counterfactual.npz"
    np.savez_compressed(OUT / name, **{k: np.array(v) for k, v in rows.items()})
    print("saved", OUT / name, len(rows["kind"]), "candidate outcomes")


if __name__ == "__main__":
    main()
