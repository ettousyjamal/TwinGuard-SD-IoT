"""Lower abstention without retraining: online recalibration of TwinGuard on the public dataset, plus the improved
fallback. No model is trained; the saved predictions of the five-member twin are used.

Variants: TwinGuard as published; + online recalibration of the conformal bound (on the most recent hour of
measured outcomes, triggered when abstention over the last hour exceeds 10%); + recalibration of the bound and of
the fidelity references. For each, system-level metrics with the reactive fallback and with the improved fallback
(intervene where the measured state violates the SLA or the calibrated no-op bound exceeds the budget).
A consistency check requires that the variant without recalibration reproduces the published verdicts exactly.
Outputs: results/twinguard/recalibration.csv, results/twinguard/recalibration_phi.npz
Run from the repository root:  python code/10_real_recalibration.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from twinguard import adapt, gate as G, realdata as rd  # noqa: E402

OUT = ROOT / "results" / "twinguard"


def metrics(r, d):
    q95 = d["q"][:, :, 0, :, 2]
    slack = np.minimum(np.log1p(rd.RTT_MAX) - q95[..., 0], np.log1p(rd.PLR_MAX) - q95[..., 2])
    fb_imp = np.where(d["cur_v"] | (slack < r["margin"][:, None]), G.ACT, G.OK)
    react = G.summarize(G.epoch_counts(r["verdict"], d, d["cur_v"]))
    imp = G.summarize(G.epoch_counts(r["verdict"], d, d["cur_v"], fallback=fb_imp))
    s = G.scores(d); o = ~np.isnan(s)
    return {**{k: react[k] for k in ("false_approval", "false_approval_worst", "missed_violations",
                                     "intervention_rate", "abstention_rate", "n_ok")},
            "coverage": float(np.mean((s <= r["margin"][:, None])[o])),
            "sys_fa_reactive": react["system_false_approval"], "sys_missed_reactive": react["system_missed_violations"],
            "sys_int_reactive": react["system_intervention_rate"],
            "sys_fa_improved": imp["system_false_approval"], "sys_missed_improved": imp["system_missed_violations"],
            "sys_int_improved": imp["system_intervention_rate"]}


def main():
    pipe = np.load(ROOT / "data" / "processed" / "graph_pipeline.npz")
    cal = rd.load("cal", pipe=pipe); A_, B_ = rd.cal_halves(cal); calib = G.calibrate(A_, B_)
    published = pd.read_csv(OUT / "gate_verdicts.csv").set_index(["split", "config"])
    rows, arr = [], {}
    for split in rd.TEST_SPLITS:
        d = rd.load(split, pipe=pipe)
        ref = G.run(d, calib["cp"], calib["q_static"], calib["ref"], calib["phi_min"])
        r0 = adapt.run_recalibrated(d, calib, trigger=np.inf)
        assert np.array_equal(r0["verdict"], ref["verdict"]), f"{split}: online loop differs from the gate"
        m0 = metrics(r0, d)
        assert abs(m0["false_approval"] - published.loc[(split, "B5 TwinGuard"), "false_approval"]) < 1e-12
        rows.append({"split": split, "config": "TwinGuard", "recalibrations": 0, **m0})
        arr[split + "/phi"] = r0["phi"]
        for name, tag, kw in [("TwinGuard + recalibrated bound", "bound", {}),
                              ("TwinGuard + recalibrated bound and references", "refs", {"refit_refs": True})]:
            r = adapt.run_recalibrated(d, calib, log=lambda m: None, **kw)
            rows.append({"split": split, "config": name, "recalibrations": len(r["adapt_at"]), **metrics(r, d)})
            arr[f"{split}/phi_{tag}"] = r["phi"]; arr[f"{split}/at_{tag}"] = np.array(r["adapt_at"])
        print(f"{split}: done", flush=True)
    # which fidelity signals hold abstention up on D3 when only the bound is recalibrated
    d = rd.load("D3", pipe=pipe); rec = []; decide = G.TwinGuard.decide

    def logged(self, q95, sd):
        v = decide(self, q95, sd)
        rec.append([G.excess(self.sig[k], self.ref[k]) > 0 for k in G.SIGNALS] + [self.phi < self.phi_min])
        return v
    G.TwinGuard.decide = logged
    adapt.run_recalibrated(d, calib, log=lambda m: None)
    G.TwinGuard.decide = decide
    a = np.array(rec); ab = a[:, -1]
    drivers = {k: float(a[ab, i].mean()) for i, k in enumerate(G.SIGNALS)}
    pd.Series(drivers, name="share_of_abstained_D3_epochs_with_signal_above_reference").to_csv(
        OUT / "recalibration_abstention_drivers.csv")
    print("signals above their reference on abstained D3 epochs (bound recalibrated only):", drivers)
    df = pd.DataFrame(rows); df.to_csv(OUT / "recalibration.csv", index=False)
    np.savez(OUT / "recalibration_phi.npz", **arr, phi_min=calib["phi_min"])
    pd.set_option("display.width", 250)
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
