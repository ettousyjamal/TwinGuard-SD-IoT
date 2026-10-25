"""Reproducibility check of the repository (CPU, a few minutes). Prints PASS/FAIL per check; exit code 1 on failure.

1. every shipped checkpoint loads into its architecture and runs a forward pass;
2. the conformal quantile and the online gate behave as specified on synthetic data;
3. the simulator is deterministic: a fork replayed under the unchanged configuration reproduces the run bit for bit;
4. a 1-h closed loop of the agents with TwinGuard on the simulator reproduces the stored reference summary;
5. if the dataset has been prepared (data/processed/graph_pipeline.npz and results/preds/), TwinGuard re-run on the
   saved predictions reproduces the published gate metrics (results/twinguard/gate_verdicts.csv) exactly.
Run from the repository root:  python code/verify.py            (add --write-reference to refresh check 4)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "code"))
from twinguard import agents as ag, gate as G, model as M, sim  # noqa: E402

FAIL = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""), flush=True)
    if not ok:
        FAIL.append(name)


def checkpoints():
    n = 0
    for p in sorted(ROOT.glob("results/**/models/*.pt")):
        ck = torch.load(p, map_location="cpu", weights_only=False)
        arch = "GNNTwin" if p.name.startswith("GNNTwin") else "LinkGRU"
        hp = ck.get("hp") or json.load(open(ROOT / "results" / "best_hyperparameters.json"))["best"][arch]
        try:
            emb = ck["state"]["emb.weight"] if "state" in ck else ck["emb.weight"]
            L = emb.shape[0]
            f_in = (ck["state"] if "state" in ck else ck)
            first = [v for k, v in f_in.items() if k.startswith("encoder.0.lin.weight")][0]
            m = M.MODELS[arch](first.shape[1] - 8, hp, np.eye(L, dtype=np.float32))
            m.load_state_dict(f_in); m.eval()
            with torch.no_grad():
                q, lg = m(torch.zeros(2, 20, L, first.shape[1] - 8))
            ok = q.shape == (2, L, 3, 3, 3) and lg.shape == (2, L, 3) and torch.isfinite(q).all().item()
        except Exception as e:                                         # noqa: BLE001
            ok, q = False, e
        check(f"checkpoint {p.relative_to(ROOT)}", ok); n += 1
    check("checkpoints found", n > 0, f"{n} checkpoints")


def gate_unit():
    cp = G.Conformal(np.arange(1, 100, dtype=float))                  # scores 1..99, n = 99
    check("conformal quantile", cp.q(0.05) == 95.0 and cp.q(0.001) == np.inf, f"q(0.05) = {cp.q(0.05)}")
    rng = np.random.default_rng(0)
    tg = G.TwinGuard(G.Conformal(rng.normal(size=5000)), 1.6, {k: 1.0 for k in G.SIGNALS}, 0.0, use=[])
    cov = []
    for _ in range(2000):
        q95 = np.zeros((13, 3)); tg.decide(q95, np.zeros(13))
        y = np.zeros((13, 3)); y[:, 0] = np.expm1(rng.normal(size=13) + 0.5)
        cov.append(np.mean(np.log1p(y[:, 0]) <= tg.margin)); tg.observe(y, np.ones(13, bool), q95, np.zeros(13))
    check("ACI tracks the target coverage under shift", abs(np.mean(cov[500:]) - 0.95) < 0.01,
          f"coverage {np.mean(cov[500:]):.3f} (target 0.95)")


def determinism():
    ex = sim.make_exo(123, 0.5, "shift"); net = sim.Network(ex)
    a, st = net.run(lambda k: sim.Config(), 0, 60)
    b, _ = net.run(lambda k: sim.Config(), 60, 120, st)
    full, _ = net.run(lambda k: sim.Config(), 0, 120)
    ok = all(np.array_equal(np.concatenate([a[k], b[k]]), full[k], equal_nan=True) for k in a)
    check("simulator fork replays bit for bit", ok)


def closed_loop(write):
    from code_sim_common import DATA, load_sim_twin, norm, sim_calibration
    if not (DATA / "cal.npz").exists():
        print("[SKIP] closed-loop reference (data/sim/cal.npz missing; run code/06_sim_data.py)"); return
    torch.set_num_threads(1)
    models = load_sim_twin("GNNTwin"); calib = sim_calibration(models)
    spec = ag.Spec("TwinGuard, improved fallback", twin="gnn", fallback="improved")
    _, s = ag.run_loop(sim.make_exo(650, 1.0, "shift"), spec, ag.Twin(models, norm()), calib)
    keys = ["violation_rate", "actions_per_h", "abstention_rate", "rollbacks_per_h", "false_approval"]
    ref_p = ROOT / "results" / "sim" / "verify_reference.json"
    got = {k: float(s[k]) for k in keys}
    if write:
        json.dump(got, open(ref_p, "w"), indent=1); print("reference written", got)
        return
    ref = json.load(open(ref_p))
    exact = all(abs(got[k] - ref[k]) < 1e-9 or (np.isnan(got[k]) and np.isnan(ref[k])) for k in keys)
    # other CPUs or PyTorch builds can change the last digits of the twin's outputs and flip a few decisions
    tol = {"violation_rate": 5e-4, "actions_per_h": 1.0, "abstention_rate": 0.02, "rollbacks_per_h": 0.5,
           "false_approval": 5e-4}
    close = all(abs(got[k] - ref[k]) <= tol[k] or (np.isnan(got[k]) and np.isnan(ref[k])) for k in keys)
    check("1-h closed loop reproduces the reference", close, ("exactly: " if exact else "within tolerance: ") + f"{got}")


def published_gate():
    from twinguard import realdata as rd
    if not (ROOT / "data" / "processed" / "graph_pipeline.npz").exists() or not (ROOT / "results" / "preds").exists():
        print("[SKIP] published gate metrics (prepare the data and predictions first, see README)"); return
    pipe = np.load(ROOT / "data" / "processed" / "graph_pipeline.npz")
    cal = rd.load("cal", pipe=pipe); A_, B_ = rd.cal_halves(cal); calib = G.calibrate(A_, B_)
    pub = pd.read_csv(ROOT / "results" / "twinguard" / "gate_verdicts.csv").set_index(["split", "config"])
    for split in rd.TEST_SPLITS:
        d = rd.load(split, pipe=pipe)
        r = G.run(d, calib["cp"], calib["q_static"], calib["ref"], calib["phi_min"])
        s = G.summarize(G.epoch_counts(r["verdict"], d, d["cur_v"]))
        p = pub.loc[(split, "B5 TwinGuard")]
        check(f"TwinGuard gate metrics, {split}", all(abs(s[k] - p[k]) < 1e-12 for k in
                                                     ("false_approval", "abstention_rate", "intervention_rate")),
              f"false approvals {s['false_approval']:.4f}, abstention {s['abstention_rate']:.4f}")


if __name__ == "__main__":
    write = "--write-reference" in sys.argv
    if not write:
        checkpoints(); gate_unit(); determinism()
    closed_loop(write)
    if not write:
        published_gate()
        print("ALL CHECKS PASSED" if not FAIL else f"{len(FAIL)} CHECK(S) FAILED: {FAIL}")
        sys.exit(1 if FAIL else 0)
