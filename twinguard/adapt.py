"""Online recalibration of TwinGuard when it keeps abstaining (no model is retrained).

Strictly causal: at decision epoch t only epochs whose outcomes are already measured are used.
"""
import numpy as np

from . import gate as G


def _replay(tg, d, idx):
    """Feed known outcomes of the epochs idx to a gate (fidelity history and ACI), without deciding."""
    sig = []
    for i in idx:
        tg.fidelity(d["sd"][i, :, 0]); tg.margin = tg.margin_now(); sig.append([tg.sig[k] for k in G.SIGNALS])
        tg.observe(d["y"][i, :, 0], d["ym"][i, :, 0], d["q"][i, :, 0, :, 2], d["q"][i, :, 0, 0, 1])
    return np.array(sig)


def run_recalibrated(d, calib, R=240, trigger=0.10, cal_h=1, min_h=2, eps=0.05, refit_refs=False, log=print):
    """Online recalibration without any model training. When the share of abstained epochs over the last R epochs
    exceeds `trigger`, the conformal scores are re-estimated on the most recent `cal_h` hours of measured
    outcomes (epochs whose targets are known at decision time) and ACI restarts from eps. With refit_refs, the
    fidelity references are also re-estimated on that hour (95th percentile, as on cal-B), so that Phi measures the
    twin's quality relative to its recent, recalibrated behavior; Phi_min is unchanged."""
    N, L = d["y"].shape[:2]; hmax = 4
    tg = G.TwinGuard(calib["cp"], calib["q_static"], calib["ref"], calib["phi_min"], eps=eps)
    verdict = np.empty((N, L), int); phi = np.empty(N); margin = np.empty(N); adapt_at = []
    for t in range(N):
        if t > 0 and t % R == 0 and t >= min_h * 240 and np.mean(verdict[t - R:t, 0] == G.ABST) > trigger:
            done = t - hmax; cal_idx = np.arange(done - cal_h * 240, done)
            dc = {"y": d["y"][cal_idx], "ym": d["ym"][cal_idx], "q": d["q"][cal_idx]}
            cp = G.Conformal(G.scores(dc)); q_static = float(cp.q(eps)); ref = calib["ref"]
            if refit_refs:
                probe = G.TwinGuard(cp, q_static, {k: 1.0 for k in G.SIGNALS}, 0.0, eps=eps, use=[])
                s = _replay(probe, d, cal_idx)[tg.H:]
                ref = {k: max(float(np.percentile(s[:, i], 95)), 1e-6) for i, k in enumerate(G.SIGNALS)}
            tg = G.TwinGuard(cp, q_static, ref, calib["phi_min"], eps=eps)
            _replay(tg, d, range(max(0, t - tg.H), t))
            adapt_at.append(t)
            log(f"    recalibration {len(adapt_at)} at epoch {t}: margin {q_static:.4f}")
        verdict[t] = tg.decide(d["q"][t, :, 0, :, 2], d["sd"][t, :, 0])
        phi[t], margin[t] = tg.phi, tg.margin
        tg.observe(d["y"][t, :, 0], d["ym"][t, :, 0], d["q"][t, :, 0, :, 2], d["q"][t, :, 0, 0, 1])
    return {"verdict": verdict, "phi": phi, "margin": margin, "adapt_at": adapt_at}
