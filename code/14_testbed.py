"""Closed-loop validation on an emulated SDN testbed: Mininet with Open vSwitch (OpenFlow 1.3), real packets, real
queues, and real flow-table, meter, and queue reconfiguration, driven by the unchanged agents and TwinGuard.

The data plane runs in a privileged container (testbed/dataplane.py; image with Mininet 2.3, Open vSwitch 2.13,
iperf3); links are shaped to 10 Mbps (the simulator's rates and capacities scaled by 0.1). The twins are those
trained in the simulator; they are not retrained on testbed data. Instead, TwinGuard is recalibrated on one testbed
calibration run (conformal scores and fidelity references, as in the online recalibration of the public dataset).
  cal                      : calibration run under the exploration policy (training law, seed 300)
  run <controller> <regime>: closed loop of one controller on the nominal (seed 633) or shift (seed 707) scenario
  summary                  : results table (results/testbed/summary.csv) and decision-cycle timing
Each run lasts 75 min of wall-clock time (15-s epochs; the first 10 min are the warm-up of the twin's window).
Outputs: results/testbed/<run>/{log.npz, summary.json, timing.csv}
Run from the repository root:  python code/14_testbed.py cal | run "TwinGuard, improved fallback" shift
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from code_sim_common import ROOT, load_sim_twin, norm, sim, sd  # noqa: E402
from twinguard import agents as ag, gate as G, testbed as T  # noqa: E402

OUT = ROOT / "results" / "testbed"
HOURS = 1.25
SCEN = {"cal": (300, "train"), "nominal": (633, "test"), "shift": (707, "shift")}
SPECS = {s.name: s for s in [
    ag.Spec("static", twin="none"), ag.Spec("reactive", twin="none"),
    ag.Spec("TwinGuard", twin="gnn"),
    ag.Spec("TwinGuard, improved fallback", twin="gnn", fallback="improved"),
]}
PORT = {"cal": 9100, "static": 9101, "reactive": 9102, "TwinGuard": 9103, "TwinGuard, improved fallback": 9104}


def slug(s):
    return s.lower().replace(", ", "_").replace(" ", "_")


def calibration_run():
    seed, law = SCEN["cal"]; exo = sim.make_exo(seed, HOURS, law); d = OUT / "cal"
    tb = T.Testbed(exo, d, port=PORT["cal"], name="tg_tb_cal")
    try:
        run = sd.explore(tb.exo, seed=seed + 7, net=tb)
    finally:
        tb.close()
    np.savez_compressed(d / "run.npz", F=run["F"], post=run["post"], acts=np.array([str(a) for a in run["acts"]]),
                        **{f"out_{k}": v for k, v in run["out"].items()})
    json.dump([[list(map(float, x)) for x in c.key()] for c in run["cfgs"]], open(d / "cfgs.json", "w"))
    pd.DataFrame(tb.timing).to_csv(d / "timing.csv", index=False)
    print("calibration run saved", flush=True)


def testbed_calibration(models):
    """TwinGuard calibrated on the testbed calibration run: cal-A (first half) conformal scores, cal-B references
    and Phi_min; no model is retrained."""
    z = np.load(OUT / "cal" / "run.npz")
    keys = json.load(open(OUT / "cal" / "cfgs.json"))
    run = {"F": z["F"], "post": z["post"], "cfgs": [sim.Config(np.array(k[0], int), np.array(k[1], int),
                                                                np.array(k[2], float), np.array(k[3], bool)) for k in keys],
           "acts": [sd.NOOP if a == str(sd.NOOP) else a for a in z["acts"]],
           "out": {k[4:]: z[k] for k in z.files if k.startswith("out_")}}
    w = sd.windows(run, norm())
    from twinguard.model import predict
    Q, P = predict(models, w["X"])
    d = {"q": Q.mean(0), "sd": Q[..., 0, 1].std(0), "y": w["Y"], "ym": w["Ym"]}
    half = len(w["X"]) // 2; cut = lambda sl: {k: v[sl] for k, v in d.items()}
    calib = G.calibrate(cut(slice(0, half)), cut(slice(half, None)))
    from twinguard.model import auroc
    v = w["V"][:, :, 0]; m = ~np.isnan(v)
    return calib, {"windows": int(len(w["X"])), "auroc_cal": float(auroc(v[m], P.mean(0)[:, :, 0][m])),
                   "violation_rate_cal": float(np.nanmean(v))}


def closed_run(name, regime):
    spec = SPECS[name]; seed, law = SCEN[regime]; exo = sim.make_exo(seed, HOURS, law)
    d = OUT / f"{slug(name)}_{regime}"
    twin = calib = None; info = {}
    if spec.twin == "gnn":
        torch.set_num_threads(2)
        models = load_sim_twin(spec.model); twin = ag.Twin(models, norm())
        calib, info = testbed_calibration(models)
    tb = T.Testbed(exo, d, port=PORT[name] + (10 if regime == "shift" else 0), name=f"tg_tb_{slug(name)}_{regime}")
    try:
        log, summ = ag.run_loop(tb.exo, spec, twin, calib, net=tb, seed=seed)
    finally:
        tb.close()
    json.dump({"controller": name, "regime": regime, "seed": seed, **info, **summ}, open(d / "summary.json", "w"),
              indent=1, default=float)
    np.savez_compressed(d / "log.npz", verdict=np.array(log.verdict), vio=np.array(log.vio), phi=np.array(log.phi),
                        vio_all=np.array(log.vio_all), decide_s=np.array(log.decide_s))
    pd.DataFrame(tb.timing).to_csv(d / "timing.csv", index=False)
    print(name, regime, {k: summ[k] for k in ("violation_rate", "actions_per_h", "abstention_rate")}, flush=True)


def summary():
    rows = [json.load(open(p)) for p in sorted(OUT.glob("*/summary.json"))]
    df = pd.DataFrame(rows); df.to_csv(OUT / "summary.csv", index=False)
    tim = pd.concat([pd.read_csv(p).assign(run=p.parent.name) for p in OUT.glob("*/timing.csv")])
    ch = tim[tim.switches > 0]
    cyc = {"install_ms_median": float(ch.install_ms.median()), "install_ms_p95": float(ch.install_ms.quantile(0.95)),
           "switches_per_change_mean": float(ch.switches.mean()), "compile_ms_median": float(tim.compile_ms.median()),
           "telemetry_ms_median": float(tim.measure_ms.median()), "telemetry_ms_p95": float(tim.measure_ms.quantile(0.95)),
           "n_changes": int(len(ch))}
    dec = np.concatenate([np.load(p)["decide_s"] for p in OUT.glob("*/log.npz")])
    if len(dec):
        cyc.update({"decision_ms_median": 1000 * float(np.median(dec)), "decision_ms_p95": 1000 * float(np.percentile(dec, 95))})
    json.dump(cyc, open(OUT / "cycle_timing.json", "w"), indent=1)
    pd.set_option("display.width", 250)
    cols = ["controller", "regime", "violation_rate", "violation_rate_all", "actions_per_h", "rollbacks_per_h",
            "abstention_rate", "false_approval", "coverage_act", "coverage_ok", "delivered_best_effort",
            "flow_mods_per_h", "auroc_cal"]
    print(df[[c for c in cols if c in df]].round(4).to_string(index=False)); print(cyc)


def probe_window(e0=40, e1=220):
    """SLA violations of every run computed from the probe logs of the data plane over a common window of epochs
    (all runs, including those stopped before their end): a link-epoch is measured if probes returned and violates
    the SLA if its mean RTT exceeds 10 ms or its probe loss 1%."""
    rows = []
    for d in sorted(OUT.glob("*/io")):
        run = d.parent.name
        if run == "cal":
            continue
        n = v = miss = 0
        rec = {}
        for f in d.glob("probe_*.log"):
            for line in open(f):
                l, k, sent, recv, m, _ = line.split()
                rec[(l, int(k))] = (int(sent), int(recv), float(m))
        for k in range(e0, e1):
            for l in sim.LINKS:
                sent, recv, m = rec.get((l, k), (0, 0, np.nan))
                if recv == 0:
                    miss += 1; continue
                n += 1; v += int(m > sim.RTT_MAX or 100.0 * (sent - recv) / sent > sim.PLR_MAX)
        ctrl, regime = run.rsplit("_", 1)
        rows.append({"controller": ctrl, "regime": regime, "epochs": e1 - e0, "measured_link_epochs": n,
                     "violation_rate": v / n, "missing_share": miss / (n + miss)})
    df = pd.DataFrame(rows); df.to_csv(OUT / f"window_{e0}_{e1}.csv", index=False)
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    if sys.argv[1] == "cal":
        calibration_run()
    elif sys.argv[1] == "run":
        t0 = time.time(); closed_run(sys.argv[2], sys.argv[3]); print(f"{(time.time() - t0) / 60:.0f} min")
    elif sys.argv[1] == "window":
        probe_window()
    else:
        summary()
