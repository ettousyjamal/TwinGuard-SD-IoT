"""Closed-loop evaluation on the simulator: agents, TwinGuard, enforcement, rollback, and fallback, against baselines.

Every controller runs on the same unseen scenarios (same random draws), four in the nominal regime and four in the
shift regime (severe capacity degradations, link failures, stronger alarm storms). The learned controllers use the
action-conditioned graph-attention twin unless stated. Further runs: TwinGuard with the per-link twin, the agent
ablations, and the same scenarios with missing measurements as frequent and as load-dependent as in the public
dataset ("missing" profile, about 20% of the records). On another topology (TG_TOPOLOGY), the core controllers run.
Outputs: results/sim[_<topology>]/closed_loop.csv (one row per controller and scenario), closed_loop_logs.npz
Run from the repository root:  [TG_TOPOLOGY=nsfnet] python code/09_sim_closed_loop.py
"""
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from code_sim_common import RES as OUT, TOPO, TWINS, load_sim_twin, norm, sim, sim_calibration  # noqa: E402
from twinguard import agents as ag  # noqa: E402

SCEN = [("nominal", s, "test") for s in range(600, 604)] + [("shift", s, "shift") for s in range(700, 704)]
HOURS = float(os.environ.get("TG_HOURS", 6.0))       # scenario length (shorter runs on other topologies)
S = ag.Spec
CORE = [
    S("static", twin="none"),
    S("reactive", twin="none"),
    S("analytical what-if", twin="analytical", margin="point", fidelity=False),
    S("twin, uncalibrated gate (B4)", twin="gnn", margin="raw", fidelity=False),
    S("TwinGuard", twin="gnn"),
    S("TwinGuard, improved fallback", twin="gnn", fallback="improved"),
    S("TwinGuard, improved fallback, per-link twin", twin="gnn", fallback="improved", model="LinkGRU"),
    S("oracle", twin="none", oracle=True),
]
FULL = CORE[:3] + [
    S("twin, ungated (B3)", twin="gnn", margin="point", fidelity=False, rollback=False),
    CORE[3],
    S("TwinGuard, no fidelity check (A1)", twin="gnn", fidelity=False),
    S("TwinGuard, no rollback", twin="gnn", rollback=False),
] + CORE[4:] + [
    S("agents: no Monitoring agent", twin="gnn", fallback="improved", mode="no_monitoring"),
    S("agents: random candidates (no Diagnosis)", twin="gnn", fallback="improved", mode="no_diagnosis"),
    S("no agents: exhaustive action search", twin="gnn", fallback="improved", mode="monolithic"),
] + [S(f"{n}, per-link twin", twin="gnn", fallback="improved", mode=m, model="LinkGRU") for n, m in [
    ("agents: no Monitoring agent", "no_monitoring"), ("agents: random candidates (no Diagnosis)", "no_diagnosis"),
    ("no agents: exhaustive action search", "monolithic")]]
MISSING = [CORE[0], CORE[1], CORE[4], CORE[5]]          # runs under the dataset-like missing profile


def jobs():
    only = os.environ.get("TG_ONLY")            # run only the controllers whose name contains this text, and merge
    out = []
    for regime, seed, law in SCEN:
        out += [(regime, seed, law, "nominal", sp) for sp in (FULL if TOPO == "dataset" else CORE)]
        if TOPO == "dataset":
            out += [(regime, seed, law, "dataset", sp) for sp in MISSING]
    return [j for j in out if only is None or only in j[4].name]


def job(args):
    regime, seed, law, missing, spec = args
    torch.set_num_threads(1)
    twin = calib = None
    if spec.twin == "gnn":
        twin = ag.Twin(MODELS[spec.model], norm()); calib = CALIB[spec.model]
    exo = sim.make_exo(seed, HOURS, law)
    t0 = time.time()
    log, summ = ag.run_loop(exo, spec, twin, calib, net=sim.Network(exo, missing=missing), seed=seed)
    return {"regime": regime, "missing": missing, "seed": seed, "controller": spec.name, **summ,
            "seconds": round(time.time() - t0, 1)}, \
        (f"{regime}|{missing}|{seed}|{spec.name}", np.array(log.verdict), np.array(log.phi), np.array(log.vio),
         np.array(log.vio_all))


def init(calib):
    global CALIB, MODELS
    CALIB = calib; MODELS = {n: load_sim_twin(n) for n in TWINS}


def main():
    torch.set_num_threads(4)
    calib = {n: sim_calibration(load_sim_twin(n)) for n in TWINS}
    for n, c in calib.items():
        print(f"{TOPO} calibration, {n}: Phi_min = {c['phi_min']:.3f}, static margin = {c['q_static']:.4f}", flush=True)
    rows, logs = [], {}
    with ProcessPoolExecutor(max_workers=4, initializer=init, initargs=(calib,)) as ex:
        for row, (key, ver, phi, vio, vio_all) in ex.map(job, jobs()):
            rows.append(row)
            logs[key + "|verdict"] = ver; logs[key + "|phi"] = phi; logs[key + "|vio"] = vio
            logs[key + "|vio_all"] = vio_all
            print(f"{row['regime']:7s} {row['missing']:7s} {row['seed']} {row['controller']:44s} "
                  f"viol {100 * row['violation_rate']:5.2f}% (all {100 * row['violation_rate_all']:5.2f}%)  "
                  f"actions/h {row['actions_per_h']:5.1f}  abstain {100 * row['abstention_rate']:4.1f}%  "
                  f"({row['seconds']:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    if os.environ.get("TG_ONLY") and (OUT / "closed_loop.csv").exists():
        old = pd.read_csv(OUT / "closed_loop.csv")
        df = pd.concat([old[~old.controller.isin(df.controller.unique())], df])
        logs = {**dict(np.load(OUT / "closed_loop_logs.npz")), **logs}
    df.to_csv(OUT / "closed_loop.csv", index=False)
    np.savez_compressed(OUT / "closed_loop_logs.npz", **logs)


if __name__ == "__main__":
    main()
