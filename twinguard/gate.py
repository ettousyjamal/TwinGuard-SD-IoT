"""TwinGuard safety engine: split conformal + adaptive conformal inference (ACI), the fidelity score, and the gate.

Same logic as notebooks/TwinGuard_gate.ipynb (class TwinGuard), with the settings passed as arguments.
Data convention (per test set d): q [N,L,H,M,3] log1p quantiles of (rtt, jitter, loss); sd [N,L,H] ensemble
disagreement of the RTT median; y [N,L,H,M] outcomes in raw units; ym [N,L,H] outcome measured.
"""
from collections import deque

import numpy as np

OK, ACT, ABST = 0, 1, 2
SIGNALS = ["error", "coverage", "novelty", "missing"]


def scores(d, h=0):
    # joint nonconformity score of RTT and loss on the log1p scale; NaN where the outcome is missing
    y = d["y"][:, :, h]; q95 = d["q"][:, :, h, :, 2]
    s = np.maximum(np.log1p(np.clip(y[..., 0], 0, None)) - q95[..., 0],
                   np.log1p(np.clip(y[..., 2], 0, None)) - q95[..., 2])
    return np.where(d["ym"][:, :, h], s, np.nan)


def violated(d, rtt_max, plr_max, h=0):
    y = d["y"][:, :, h]
    v = ((y[..., 0] > rtt_max) | (y[..., 2] > plr_max)).astype(np.float32)
    return np.where(d["ym"][:, :, h], v, np.nan)


class Conformal:
    def __init__(self, cal_scores):
        self.s = np.sort(cal_scores[~np.isnan(cal_scores)]); self.n = len(self.s)

    def q(self, alpha):
        if alpha <= 1.0 / (self.n + 1):
            return np.inf
        k = int(np.ceil((self.n + 1) * (1 - alpha)))
        return self.s[min(max(k, 1), self.n) - 1]


def excess(x, r):
    return min(1.0, max(0.0, (x - r) / r))


class TwinGuard:
    """Online gate. decide() uses only information available at decision time; observe() receives the outcome."""

    def __init__(self, cp, q_static, ref, phi_min, eps=0.05, gamma=0.005, H=20, use=SIGNALS, aci=True,
                 rtt_max=10.0, plr_max=1.0):
        self.cp, self.q_static, self.ref, self.phi_min = cp, q_static, ref, phi_min
        self.eps, self.gamma, self.H, self.use, self.aci = eps, gamma, H, list(use), aci
        self.budget = (np.log1p(rtt_max), np.log1p(plr_max))
        self.alpha = eps
        self.hist = deque(maxlen=H)          # (n measured, sum |error|, n static-bound failures, n missing, n links)
        self.nov = deque(maxlen=H)           # ensemble disagreement, current epoch included

    def signals(self):
        n = sum(h[0] for h in self.hist); links = sum(h[4] for h in self.hist)
        return {"error": sum(h[1] for h in self.hist) / n if n else 0.0,
                "coverage": sum(h[2] for h in self.hist) / n if n else 0.0,
                "novelty": float(np.mean(self.nov)),
                "missing": sum(h[3] for h in self.hist) / links if links else 0.0}

    def fidelity(self, sd):
        self.nov.append(float(np.mean(sd)))
        self.sig = self.signals()
        self.phi = 1.0
        for k in SIGNALS:
            if k in self.use:
                self.phi *= 1.0 - excess(self.sig[k], self.ref[k])
        return self.phi

    def margin_now(self, level=None):
        if level is not None:
            return self.cp.q(level)
        return self.cp.q(self.alpha) if self.aci else self.cp.q(self.eps)

    def slack(self, q95):
        return np.minimum(self.budget[0] - q95[..., 0], self.budget[1] - q95[..., 2])

    def decide(self, q95, sd):
        # q95: [L, 3] log-scale 0.95 quantiles (rtt, jitter, loss); sd: [L] ensemble disagreement
        self.fidelity(sd)
        self.margin = self.margin_now()
        verdict = np.where(self.slack(q95) >= self.margin, OK, ACT)
        if self.use and self.phi < self.phi_min:
            verdict[:] = ABST                     # twin not trusted: the fallback acts
        return verdict

    def observe(self, y, measured, q95, q50_rtt):
        o = measured
        lr = np.log1p(np.clip(y[:, 0], 0, None)); lp = np.log1p(np.clip(y[:, 2], 0, None))
        s = np.maximum(lr - q95[:, 0], lp - q95[:, 2])
        if o.any() and self.aci:
            self.alpha += self.gamma * (self.eps - np.mean(s[o] > self.margin))
        self.hist.append((int(o.sum()), float(np.abs(lr - q50_rtt)[o].sum(dtype=np.float64)),
                          int((s[o] > self.q_static).sum()), int((~o).sum()), len(o)))


def run(d, cp, q_static, ref, phi_min, h=0, **kw):
    """Run the online gate over a test set at horizon index h (the decision uses the Δ-ahead forecast)."""
    tg = TwinGuard(cp, q_static, ref, phi_min, **kw)
    N, L = d["y"].shape[:2]
    verdict = np.empty((N, L), int); phi = np.empty(N); margin = np.empty(N); sig = np.empty((N, len(SIGNALS)))
    for t in range(N):
        q95 = d["q"][t, :, h, :, 2]
        verdict[t] = tg.decide(q95, d["sd"][t, :, h])
        phi[t], margin[t], sig[t] = tg.phi, tg.margin, [tg.sig[k] for k in SIGNALS]
        tg.observe(d["y"][t, :, h], d["ym"][t, :, h], q95, d["q"][t, :, h, 0, 1])
    return {"verdict": verdict, "phi": phi, "margin": margin, "sig": sig}


def calibrate(cal_a, cal_b, eps=0.05, abstain_budget=0.05, H=20, **kw):
    """cal-A: conformal scores; cal-B: signal references (95th percentile) and Phi_min (abstention budget)."""
    cp = Conformal(scores(cal_a)); q_static = float(cp.q(eps))
    r = run(cal_b, cp, q_static, {k: 1.0 for k in SIGNALS}, 0.0, use=[], eps=eps, H=H, **kw)
    ref = {k: float(np.percentile(r["sig"][H:, i], 95)) for i, k in enumerate(SIGNALS)}
    # a network without missing records in calibration: one missing record in the look-back window is the reference
    ref["missing"] = max(ref["missing"], 1.0 / (H * cal_b["y"].shape[1]))
    phi_b = run(cal_b, cp, q_static, ref, 0.0, eps=eps, H=H, **kw)["phi"]
    phi_min = float(np.percentile(phi_b, 100 * abstain_budget, method="lower"))
    return {"cp": cp, "q_static": q_static, "ref": ref, "phi_min": phi_min}


def epoch_counts(verdict, d, cur_v, rtt_max=10.0, plr_max=1.0, fallback=None):
    """Per-epoch counts for the metrics of the paper. fallback: verdicts [N,L] used on abstained epochs
    (default: the reactive rule, intervene where the last measured state violates the SLA)."""
    v = violated(d, rtt_max, plr_max); obs = ~np.isnan(v); vio = obs & (v == 1)
    ok = verdict == OK
    fb = np.where(cur_v, ACT, OK) if fallback is None else fallback
    sys_ok = np.where(verdict == ABST, fb == OK, ok)
    return np.stack([(ok & obs).sum(1), (ok & vio).sum(1), ok.sum(1), (ok & (vio | ~obs)).sum(1), vio.sum(1),
                     (verdict == ACT).sum(1), (verdict == ABST).any(1), np.full(len(v), v.shape[1]),
                     (sys_ok & obs).sum(1), (sys_ok & vio).sum(1), sys_ok.sum(1)], 1).astype(float)


def summarize(c):
    S = c.sum(0); div = lambda a, b: a / b if b > 0 else np.nan
    return {"false_approval": div(S[1], S[0]), "false_approval_worst": div(S[3], S[2]),
            "missed_violations": div(S[1], S[4]), "intervention_rate": div(S[5], S[7]),
            "abstention_rate": S[6] / len(c), "system_false_approval": div(S[9], S[8]),
            "system_missed_violations": div(S[9], S[4]), "system_intervention_rate": 1 - div(S[10], S[7]),
            "n_ok": S[2]}
