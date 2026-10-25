"""Closed loop of TwinGuard-SD-IoT on the simulator: Monitoring, Diagnosis, Planning, and Recovery agents, the
action-conditioned twin, the TwinGuard gate, rollback, and the fallback policy; plus the baseline controllers.

Decision timing: at the end of epoch t the controller observes the epoch, decides, and the chosen configuration
is in force during epoch t+1. The gate verifies the forecast at horizon Delta = 15 s (epoch t+1), as in the paper.
"""
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from . import gate as G
from . import sim, simdata as sd

T_VERIFY = 2              # epochs over which Recovery verifies an enforced action before keeping it
QUIET_REVERT = 20         # epochs without alert after which Recovery proposes to restore the default configuration
FALLBACK_HOLD = 8         # epochs a fallback rule stays in place after its trigger disappears
CAP_FALLBACK = 0.3        # best-effort meter of the pre-validated fallback rule
K_MAX = 8                 # candidates per planning round
TABU = 20                 # epochs during which a rolled-back configuration is not proposed again
R_MAX = 2                 # planning rounds (re-planning after a rejection)


# ----------------------------------------------------------------------------------------------- twin runtime
class Twin:
    """Ensemble of action-conditioned models; builds decision-time windows exactly as the training windows."""

    def __init__(self, models, norm):
        self.models, self.norm = models, norm

    def windows(self, hist, F_t, cfgs, rates, failed):
        """hist: raw features of epochs <= t-1 (at least 2W-1 of them); F_t: raw features of epoch t."""
        base = np.asarray(hist[-(2 * sd.W - 1):])
        X = [self.norm(np.concatenate([base, sd.post_step(F_t, c, rates, failed)[None]]))[-sd.W:] for c in cfgs]
        return np.stack(X).astype(np.float32)

    def predict(self, X):
        from .model import predict
        Q, P = predict(self.models, X)
        return {"q": Q.mean(0), "sd": Q[..., 0, 1].std(0), "p": P.mean(0)}     # q [K,L,H,M,3]


class Analytical:
    """White-box queueing what-if model (a classic NDT baseline): steady-state M/M/1-type delay and M/M/1/K loss of
    the post-action utilization at nominal capacity, plus the link's recent minimum RTT as propagation delay."""

    def predict_raw(self, hist, F_t, cfgs, rates, failed):
        H = np.asarray(list(hist)[-sd.W:] + [F_t])
        base = np.nanmin(np.where(np.isnan(H[..., 0]), np.inf, H[..., 0]), 0)
        base = np.where(np.isfinite(base), np.minimum(base, 2.0), 1.2)
        out = []
        for c in cfgs:
            load = sd.controller_loads(c, rates, failed)
            hp = np.where(c.prio, load[0], load.sum(0)); rho = np.minimum(hp / sim.C_NOM, 0.99)
            rtt = base + sim.S_BURST * rho / (1 - rho); loss = 100 * sim.p_block(rho)
            out.append((rtt, loss))
        return out


# ----------------------------------------------------------------------------------------------- agents
def flows_on(cfg, e, failed):
    return [f for f in range(sim.NF) if e in sim.path_of(f, cfg, failed)]


def plan(cfg, alerts, rates, failed):
    """Diagnosis + Planning: for the most critical alerted links, the flows that load them, and candidate actions."""
    cands = []
    for e in alerts[:2]:
        fl = sorted(flows_on(cfg, e, failed), key=lambda f: -rates[f])
        for f in fl[:3]:                                     # reroute the largest contributors around the link
            opts = sim.PATHS[(sim.src_of(f, cfg), sim.FLOWS[f].dst)]
            for i, p in enumerate(opts):
                if i != cfg.route[f] and e not in p:
                    cands.append(sd.Action("reroute", f, i)); break
        lc2 = sd.controller_loads(cfg, rates, failed)[2][e]
        if lc2 > 1.0 and cfg.cap[e] != sd.CAP_LEVELS[0]:
            cands.append(sd.Action("cap", e, sd.CAP_LEVELS[0]))
        if not cfg.prio[e]:
            cands.append(sd.Action("prio", e, 1.0))
        grp = [sim.FLOWS[f].group for f in fl if sim.FLOWS[f].group >= 0]
        if grp:
            g = max(set(grp), key=lambda x: sum(rates[f] for f in fl if sim.FLOWS[f].group == x))
            cands.append(sd.Action("gateway", g, 1 - cfg.gw[g]))
    seen, out = set(), []
    for a in cands:
        if a not in seen:
            seen.add(a); out.append(a)
    return out[:K_MAX]


class Fallback:
    """Fallback policy pi_safe, applied while TwinGuard abstains or when no candidate passes the gate.

    reactive (the rule of Section 4, also the reactive baseline): on observed violations, shortest-path routing with
        best-effort meters on the violating links; released after FALLBACK_HOLD quiet epochs.
    improved: keeps the last approved configuration (no rerouting) and, on links with an observed violation or whose
        calibrated no-op bound exceeds the budget, enables strict priority for the critical class and meters
        best-effort traffic; on release it restores the configuration it started from.
    """

    def __init__(self, mode="reactive"):
        self.mode, self.active, self.hold, self.base = mode, False, 0, None

    def step(self, cfg, vio_t, bound_alerts=()):
        trig = set(np.where(vio_t == 1)[0])
        if self.mode == "improved":
            trig |= set(bound_alerts)
        if trig:
            if not self.active:
                self.base = cfg.copy()
            if self.mode == "reactive":
                c = sim.Config(); c.gw = cfg.gw.copy()
                if self.active:
                    c.cap = cfg.cap.copy()
            else:
                c = cfg.copy()
            for e in sorted(trig):
                c.cap[e] = max(c.cap[e], CAP_FALLBACK) if c.cap[e] else CAP_FALLBACK
                if self.mode == "improved":
                    c.prio[e] = True
            self.active, self.hold = True, FALLBACK_HOLD
            return c, ("fallback" if c.key() != cfg.key() else "ok")
        if self.active:
            self.hold -= 1
            if self.hold <= 0:
                self.active = False
                if self.mode == "reactive":
                    c = sim.Config(); c.gw = cfg.gw.copy()
                else:
                    c = self.base
                return c, ("fallback" if c.key() != cfg.key() else "ok")
        return cfg, "ok"

    def release(self):
        """TwinGuard trusts the twin again: the agents take over from the current configuration."""
        self.active, self.hold = False, 0


# ----------------------------------------------------------------------------------------------- closed loop
@dataclass
class Spec:
    name: str
    twin: str = "gnn"            # gnn, gru, analytical, none
    margin: str = "aci"          # aci (TwinGuard), raw (uncalibrated q95), point (median, no bound)
    fidelity: bool = True
    rollback: bool = True
    fallback: str = "reactive"   # reactive (pi_safe of Section 4) or improved (see class Fallback)
    oracle: bool = False
    agents: bool = True
    # agent ablations: "agents" (Monitoring, Diagnosis, Planning, Recovery); "no_monitoring" (planning is triggered
    # only by observed violations, not by the twin's no-op forecast); "no_diagnosis" (K random candidates instead of
    # diagnosed, targeted ones); "monolithic" (no agent decomposition: the whole action space is scored by the twin
    # and the gate at every alert, with the selection correction for that many candidates)
    mode: str = "agents"
    model: str = "GNNTwin"       # learned twin used by the experiment scripts (GNNTwin or its per-link variant)


@dataclass
class Log:
    vio: list = field(default_factory=list)          # [L] measured violation (NaN missing) per epoch
    verdict: list = field(default_factory=list)       # 'ok', 'act', 'abstain', 'fallback', 'reject'
    enforced: int = 0
    rollbacks: int = 0
    cost: float = 0.0
    withheld: float = 0.0                             # best-effort Mbps-epochs withheld by meters
    approved_vio: list = field(default_factory=list)  # per approved decision: (n measured links, n violating) at t+1
    phi: list = field(default_factory=list)
    vio_all: list = field(default_factory=list)       # [L] violation of the would-be measurement, missing included
    approved_vio_all: list = field(default_factory=list)  # per approved decision: (n links, n violating), all outcomes
    approved_cov: list = field(default_factory=list)  # per approved decision: (verdict, n measured, n within bound)
    cfgs: list = field(default_factory=list)          # configuration key in force during each epoch
    thr: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))  # class x (offered, after meters, delivered)
    decide_s: list = field(default_factory=list)      # wall time of every decision that consulted the twin


def _selection_margin(tg, k):
    return tg.cp.q(max(tg.alpha, 1e-9) / (k + 1)) if k > 0 else tg.margin_now()


def run_loop(exo, spec, twin=None, calib=None, eps=0.05, warmup=None, net=None, seed=0):
    """Simulate one scenario under a controller. Returns the log and a summary. net: the network environment
    (default: the simulator; the Mininet testbed provides the same interface and fills exo as it measures)."""
    net = sim.Network(exo) if net is None else net
    E = net.E; warm = 2 * sd.W if warmup is None else warmup
    rng = np.random.default_rng(seed)
    cfg = sim.Config(); state = None; hist = []; log = Log()
    tg = None
    if spec.twin in ("gnn", "gru") and spec.margin != "point":
        tg = G.TwinGuard(calib["cp"], calib["q_static"], calib["ref"], calib["phi_min"], eps=eps,
                         use=G.SIGNALS if spec.fidelity else [], aci=(spec.margin == "aci"))
    ana = Analytical() if spec.twin == "analytical" else None
    pending = None              # prediction for the configuration in force during the next epoch (for observe)
    verify = None               # (previous cfg, epochs left, approved links) of an action under verification
    quiet = 0
    fb = Fallback("reactive" if spec.fallback == "reactive" else "improved")
    tabu = {}                   # configuration key -> epoch until which it is excluded (after a rollback)
    prev_verdict = None
    budget = (np.log1p(sim.RTT_MAX), np.log1p(sim.PLR_MAX))
    for k in range(E):
        o, state = net.run(lambda _k: cfg, k, k + 1, state)
        F_t = sd.raw_features(o, [cfg], exo, k)[0]
        vio_t = sim.violation(o)[0]; log.vio.append(vio_t); log.cfgs.append(cfg.key())
        sl = slice(k * sim.STEP, (k + 1) * sim.STEP); rates = exo.rates[:, sl].mean(1)
        failed = exo.capf[:, sl].min(1) <= 0
        lam = sim.class_loads(cfg, exo.rates[:, sl], failed)[1]
        log.withheld += float((exo.rates[sim.CLS == 2, sl] - lam[sim.CLS == 2]).sum() / sim.STEP)
        va = ((o["rtt"][0] > sim.RTT_MAX) | (o["loss"][0] > sim.PLR_MAX)).astype(float); va[failed] = np.nan
        log.vio_all.append(va)
        if k >= warm:
            log.thr += _throughput(cfg, rates, lam.mean(1), o["true_loss"][0] / 100.0, failed)
        meas = ~np.isnan(vio_t)
        # outcome of the previous decision (epoch t = k is the t+1 of the decision taken at k-1)
        if pending is not None:
            if tg is not None:
                y = np.stack([o["rtt"][0], o["jitter"][0], o["loss"][0]], -1)
                if prev_verdict in ("ok", "act"):        # coverage of the bound under which the decision was approved
                    s_ = np.maximum(np.log1p(np.clip(y[:, 0], 0, None)) - pending["q95"][:, 0],
                                    np.log1p(np.clip(y[:, 2], 0, None)) - pending["q95"][:, 2])
                    m_ = o["measured"][0]
                    log.approved_cov.append((prev_verdict, int(m_.sum()), int((s_[m_] <= tg.margin).sum())))
                tg.observe(y, o["measured"][0], pending["q95"], pending["q50"])
        if prev_verdict in ("ok", "act"):
            log.approved_vio.append((int(meas.sum()), int(np.nansum(vio_t))))
            log.approved_vio_all.append((int((~np.isnan(va)).sum()), int(np.nansum(va))))
        # Recovery: verification of an enforced action
        if verify is not None and spec.rollback:
            prev, left, links = verify
            if np.any(vio_t[links] == 1):
                tabu[cfg.key()] = k + TABU
                cfg = prev; log.rollbacks += 1; verify = None
                hist.append(F_t); log.verdict.append("rollback"); pending = None; prev_verdict = "rollback"
                if tg is not None:
                    log.phi.append(tg.phi)
                continue
            verify = (prev, left - 1, links) if left > 1 else None
        hist.append(F_t)
        if k < warm or spec.name == "static":
            log.verdict.append("none"); pending = None; prev_verdict = None; continue
        # ---------------- decision for epoch k+1
        t_dec = time.perf_counter()
        new_cfg, verdict, approved_links, pend = cfg, "ok", None, None
        if spec.oracle:
            new_cfg, verdict = _oracle(net, state, k, cfg, vio_t, rates, failed, log)
        elif spec.twin == "none":                                   # reactive rule only
            new_cfg, verdict = fb.step(cfg, vio_t)
        else:
            if spec.twin == "analytical":
                pr = ana.predict_raw(hist[:-1], F_t, [cfg], rates, failed)[0]
                ub = np.log1p(pr[0]); lb = np.log1p(pr[1]); slack0 = np.minimum(budget[0] - ub, budget[1] - lb)
                margin0 = 0.0; phi_ok = True
            else:
                X = twin.windows(hist[:-1], F_t, [cfg], rates, failed); p0 = twin.predict(X)
                qi = 1 if spec.margin == "point" else 2
                q95 = p0["q"][0, :, 0, :, qi]
                slack0 = np.minimum(budget[0] - q95[:, 0], budget[1] - q95[:, 2])
                phi_ok = True
                if tg is not None:
                    tg.fidelity(p0["sd"][0, :, 0]); log.phi.append(tg.phi)
                    phi_ok = not (spec.fidelity and tg.phi < tg.phi_min)
                    margin0 = 0.0 if spec.margin == "raw" else tg.margin_now()
                    tg.margin = margin0             # margin of the decision (ACI scores its outcome against it)
                else:
                    margin0 = 0.0
                pend = {"q95": p0["q"][0, :, 0, :, 2], "q50": p0["q"][0, :, 0, 0, 1]}
            if not phi_ok:                                          # TwinGuard abstains: the fallback acts
                new_cfg, _ = fb.step(cfg, vio_t, list(np.where(slack0 < margin0)[0]))
                verdict = "abstain"
            else:
                fb.release()
                predicted = set() if spec.mode == "no_monitoring" else set(np.where(slack0 < margin0)[0])
                alerts = sorted(predicted | set(np.where(vio_t == 1)[0]), key=lambda e: slack0[e] - margin0)
                if alerts and spec.agents:
                    quiet = 0
                    new_cfg, verdict, approved_links, pend2 = _plan_and_verify(
                        spec, twin, ana, tg, hist[:-1], F_t, cfg, alerts, rates, failed, budget, log,
                        tabu={c for c, until in tabu.items() if until > k}, proposer=_proposer(spec, rng))
                    if pend2 is not None:
                        pend = pend2
                    if verdict == "reject":                         # no candidate passes the gate: fallback
                        new_cfg, verdict = fb.step(cfg, vio_t, list(np.where(slack0 < margin0)[0]))
                        verdict = "fallback" if new_cfg.key() != cfg.key() else "reject"
                elif not cfg.is_default():
                    quiet += 1
                    if quiet >= QUIET_REVERT:                       # try to restore the default configuration
                        quiet = 0
                        new_cfg, verdict, approved_links, pend2 = _plan_and_verify(
                            spec, twin, ana, tg, hist[:-1], F_t, cfg, [], rates, failed, budget, log,
                            fixed=[sd.Action("revert")], tabu={c for c, until in tabu.items() if until > k})
                        if verdict == "reject":
                            verdict = "ok"
                        if pend2 is not None:
                            pend = pend2
        if new_cfg.key() != cfg.key() and verdict in ("act",):
            log.enforced += 1
            if spec.rollback and approved_links is not None:
                verify = (cfg, T_VERIFY, approved_links)
        if tg is not None and pend is not None and new_cfg.key() != cfg.key() and verdict != "act":
            X = twin.windows(hist[:-1], F_t, [new_cfg], rates, failed); pn = twin.predict(X)
            pend = {"q95": pn["q"][0, :, 0, :, 2], "q50": pn["q"][0, :, 0, 0, 1]}
        pending = pend if tg is not None else None
        if spec.twin != "none" and not spec.oracle:
            log.decide_s.append(time.perf_counter() - t_dec)
        prev_verdict = verdict
        cfg = new_cfg; log.verdict.append(verdict)
    return log, summarize(log, E, warm)


def _proposer(spec, rng):
    """Candidate generation of the agent ablations (None: the Diagnosis and Planning agents)."""
    if spec.mode == "no_diagnosis":
        def random_k(cfg, alerts, rates, failed):
            pool = sd.all_actions(cfg)
            return [pool[i] for i in rng.choice(len(pool), min(K_MAX, len(pool)), replace=False)]
        return random_k
    if spec.mode == "monolithic":
        return lambda cfg, alerts, rates, failed: sd.all_actions(cfg)
    return None


def _plan_and_verify(spec, twin, ana, tg, hist, F_t, cfg, alerts, rates, failed, budget, log, fixed=None,
                     tabu=frozenset(), proposer=None):
    """Planning rounds and TwinGuard verification. Returns (config, verdict, approved links, pending prediction)."""
    tried = []
    for rnd in range(R_MAX):
        if fixed is not None:
            acts = fixed if rnd == 0 else []
        elif rnd == 0:
            acts = (proposer or plan)(cfg, alerts, rates, failed)
        else:                                                         # re-plan: pairs of the best single actions
            best = [a for a, _ in sorted(tried, key=lambda x: x[1])[:3]]
            acts = [(a, b) for i, a in enumerate(best) for b in best[i + 1:] if a.kind != b.kind or a.target != b.target]
        if not acts:
            break
        cfgs, keep = [], []
        for a in acts:
            c = cfg
            for b in (a if isinstance(a, tuple) else (a,)):
                c = sd.apply(c, b)
            if c.key() not in tabu and c.key() != cfg.key():
                cfgs.append(c); keep.append(a)
        acts = keep
        if not acts:
            continue
        K = len(acts)
        if spec.twin == "analytical":
            pr = ana.predict_raw(hist, F_t, cfgs, rates, failed)
            sl = np.array([np.minimum(budget[0] - np.log1p(r), budget[1] - np.log1p(l)) for r, l in pr]); m = 0.0
            preds = None
        else:
            X = twin.windows(hist, F_t, cfgs, rates, failed); preds = twin.predict(X)
            qi = 1 if spec.margin == "point" else 2
            q = preds["q"][:, :, 0, :, qi]
            sl = np.minimum(budget[0] - q[..., 0], budget[1] - q[..., 2])            # [K, L]
            m = 0.0 if tg is None or spec.margin == "raw" else _selection_margin(tg, K)
        n_bad = (sl < m).sum(1)
        costs = np.array([sum(sd.cost(cfg, b, sd.controller_loads(cfg, rates, failed)[2])
                              for b in (a if isinstance(a, tuple) else (a,))) for a in acts])
        feasible = np.where(n_bad == 0)[0]
        if len(feasible):
            i = feasible[np.argmin(costs[feasible])]
            if tg is not None and spec.margin == "aci":
                tg.margin = m                       # the selection-corrected margin under which it was approved
            log.cost += float(costs[i])
            pend = None if preds is None else {"q95": preds["q"][i, :, 0, :, 2], "q50": preds["q"][i, :, 0, 0, 1]}
            return cfgs[i], "act", np.arange(sim.L), pend
        tried += [(a, n) for a, n in zip(acts, n_bad) if not isinstance(a, tuple)]
        if spec.margin == "point" and fixed is None:                  # ungated twin: apply the best prediction
            i = int(np.argmin(n_bad * 100 + costs)); log.cost += float(costs[i])
            pend = None if preds is None else {"q95": preds["q"][i, :, 0, :, 2], "q50": preds["q"][i, :, 0, 0, 1]}
            return cfgs[i], "act", None, pend
    return cfg, "reject", None, None


def _oracle(net, state, k, cfg, vio_t, rates, failed, log):
    """Upper bound: the planner's candidates scored by exact counterfactual simulation (true outcomes, 60 s)."""
    o0, _ = net.run(lambda _k: cfg, k + 1, min(k + 5, net.E), state)
    v0 = np.nansum(sim.violation(o0))
    if v0 == 0 and cfg.is_default():
        return cfg, "ok"
    alerts = list(np.where(np.nansum(sim.violation(o0), 0) > 0)[0]) or list(np.where(vio_t == 1)[0])
    acts = plan(cfg, alerts, rates, failed) + ([sd.Action("revert")] if not cfg.is_default() else [])
    best, best_v, best_c = cfg, v0, 0.0
    for a in acts:
        c = sd.apply(cfg, a)
        o, _ = net.run(lambda _k: c, k + 1, min(k + 5, net.E), state)
        v = np.nansum(sim.violation(o)); cst = sd.cost(cfg, a)
        if v < best_v or (v == best_v and best is not cfg and cst < best_c):
            best, best_v, best_c = c, v, cst
    if best is not cfg:
        log.cost += best_c; return best, "act"
    return cfg, "ok"


def _throughput(cfg, rates, lam, loss, failed):
    """Offered, admitted (after best-effort meters), and delivered traffic per class (Mbps, epoch means)."""
    out = np.zeros((3, 3))
    for f in range(sim.NF):
        p = sim.path_of(f, cfg, failed); c = sim.CLS[f]
        out[c] += (rates[f], lam[f], lam[f] * np.prod(1.0 - loss[list(p)]) if p else 0.0)
    return out


def rules(key):
    """OpenFlow state of a configuration: one forwarding entry per flow and hop, one meter per metered link, and
    one queue assignment per prioritized link."""
    route, gw, cap, prio = key
    cfg = sim.Config(np.array(route), np.array(gw), np.array(cap, float), np.array(prio, bool))
    r = {("fwd", f, e) for f in range(sim.NF) for e in sim.path_of(f, cfg, None)}
    return r | {("meter", e, c) for e, c in enumerate(cap) if c > 0} | {("queue", e) for e, p in enumerate(prio) if p}


def stability(cfgs, hours, flap_window=20):
    """Configuration churn: changes, flow-table modifications, route changes, and route flaps (a flow returning to
    its previous path within flap_window epochs) per hour, and the mean dwell time of a configuration."""
    changes = mods = rchg = flaps = 0; last = {}; dwell = []; since = 0
    for k in range(1, len(cfgs)):
        since += 1
        if cfgs[k] == cfgs[k - 1]:
            continue
        changes += 1; dwell.append(since); since = 0
        mods += len(rules(cfgs[k]) ^ rules(cfgs[k - 1]))
        for f, (a, b) in enumerate(zip(cfgs[k - 1][0], cfgs[k][0])):
            if a != b:
                rchg += 1
                if f in last and last[f][0] == b and k - last[f][1] <= flap_window:
                    flaps += 1
                last[f] = (a, k)
    return {"config_changes_per_h": changes / hours, "flow_mods_per_h": mods / hours,
            "route_changes_per_h": rchg / hours, "route_flaps_per_h": flaps / hours,
            "mean_dwell_min": float(np.mean(dwell)) * sim.STEP / 60 if dwell else np.nan}


def summarize(log, E, warm):
    V = np.array(log.vio[warm:]); meas = ~np.isnan(V)
    VA = np.array(log.vio_all[warm:]); known = ~np.isnan(VA); unm = known & ~meas
    ver = np.array(log.verdict[warm:])
    av = np.array(log.approved_vio) if log.approved_vio else np.zeros((0, 2))
    aa = np.array(log.approved_vio_all) if log.approved_vio_all else np.zeros((0, 2))
    cov = {v: np.array([c[1:] for c in log.approved_cov if c[0] == v]).reshape(-1, 2) for v in ("ok", "act")}
    hours = (E - warm) * sim.STEP / 3600
    T = log.thr
    extra = {"violation_rate_all": float(np.nansum(VA) / known.sum()), "unmeasured_share": float(unm.sum() / known.sum()),
             "violation_rate_unmeasured": float(np.nansum(VA[unm]) / unm.sum()) if unm.any() else np.nan,
             "false_approval_all": float(aa[:, 1].sum() / aa[:, 0].sum()) if aa[:, 0].sum() else np.nan,
             **{f"coverage_{v}": float(c[:, 1].sum() / c[:, 0].sum()) if c[:, 0].sum() else np.nan
                for v, c in cov.items()},
             "n_cov_act": int(cov["act"][:, 0].sum()),
             "delivered_critical": T[0, 2] / T[0, 0], "delivered_telemetry": T[1, 2] / T[1, 0],
             "delivered_best_effort": T[2, 2] / T[2, 0], "be_metered_share": 1 - T[2, 1] / T[2, 0],
             "lost_in_network_share": float((T[:, 1] - T[:, 2]).sum() / T[:, 0].sum()),
             "decision_ms_mean": 1000 * float(np.mean(log.decide_s)) if log.decide_s else np.nan,
             "decision_ms_p95": 1000 * float(np.percentile(log.decide_s, 95)) if log.decide_s else np.nan,
             **stability(log.cfgs[warm:], hours)}
    return {**extra, "violation_rate": float(np.nansum(V) / meas.sum()),
            "violating_link_epochs_per_h": float(np.nansum(V) / hours),
            "actions_per_h": log.enforced / hours, "rollbacks_per_h": log.rollbacks / hours,
            "abstention_rate": float(np.mean(ver == "abstain")), "fallback_epochs": float(np.mean(ver == "fallback")),
            "false_approval": float(av[:, 1].sum() / av[:, 0].sum()) if av[:, 0].sum() else np.nan,
            "cost_per_h": log.cost / hours, "be_withheld_mbps": log.withheld / (E - 0)}
