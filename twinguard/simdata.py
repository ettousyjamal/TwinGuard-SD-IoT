"""SDN action space, controller-side features, and datasets of the simulator for the action-conditioned twin.

Features (per link and epoch) are what an SDN controller observes: probe RTT, jitter, and loss, the offered load
per class from flow counters, and its own configuration (priority queue, best-effort meter). The twin's input
window ends with the *post-action* step G_t^a: the loads of the last epoch are re-mapped onto the routes,
gateways, and meters of the candidate configuration, and the configuration flags are those of the candidate.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import sim

W, HORIZONS = 20, (1, 2, 4)
CAP_LEVELS = (0.3, 0.5)
FEAT = ["rtt", "jitter", "loss", "load", "load_c0", "load_c2", "prio", "cap", "dload"]
LOGF = 6                                        # the first six features are log1p-transformed and standardized


# ----------------------------------------------------------------------------------------------- actions
@dataclass(frozen=True)
class Action:
    kind: str                                   # noop, reroute, gateway, cap, prio, revert
    target: int = -1                            # flow, group, or link index
    value: float = 0.0                          # path index, gateway index, cap fraction, priority flag

    def label(self):
        if self.kind == "noop":
            return "no-op"
        if self.kind == "revert":
            return "revert to default"
        if self.kind == "reroute":
            return f"reroute {sim.FLOWS[self.target].name} to path {int(self.value)}"
        if self.kind == "gateway":
            return f"move {sim.GROUPS[self.target][0]} to gateway {sim.GROUPS[self.target][1 + int(self.value)]}"
        if self.kind == "cap":
            return f"meter best effort on {sim.LINKS[self.target]} at {self.value:.0%}" if self.value else \
                f"remove meter on {sim.LINKS[self.target]}"
        return f"{'enable' if self.value else 'disable'} priority on {sim.LINKS[self.target]}"


NOOP = Action("noop")


def apply(cfg, a):
    c = cfg.copy()
    if a.kind == "reroute":
        c.route[a.target] = int(a.value)
    elif a.kind == "gateway":
        g = a.target; c.gw[g] = int(a.value)
        for f, fl in enumerate(sim.FLOWS):                          # routes restart from the new gateway
            if fl.group == g:
                c.route[f] = 0
    elif a.kind == "cap":
        c.cap[a.target] = a.value
    elif a.kind == "prio":
        c.prio[a.target] = bool(a.value)
    elif a.kind == "revert":
        c = sim.Config()
    return c


def cost(cfg, a, loads_c2=None):
    """Reconfiguration cost (Table 2): flow rules installed, devices remapped, best-effort throughput withheld."""
    if a.kind == "noop":
        return 0.0
    if a.kind == "reroute":
        return 0.25 * len(sim.path_of(a.target, apply(cfg, a), None))
    if a.kind == "gateway":
        return 2.0
    if a.kind == "cap":
        if not a.value:
            return 0.25
        lc2 = 0.0 if loads_c2 is None else loads_c2[a.target]
        return 0.5 + max(0.0, lc2 - a.value * sim.C_NOM) / 20.0
    if a.kind == "prio":
        return 0.5
    return 0.5 * (int(cfg.route.astype(bool).sum()) + int(cfg.gw.sum()) + int((cfg.cap > 0).sum()) + int(cfg.prio.sum()))


def all_actions(cfg):
    acts = []
    for f, fl in enumerate(sim.FLOWS):
        n = len(sim.PATHS[(sim.src_of(f, cfg), fl.dst)])
        acts += [Action("reroute", f, i) for i in range(n) if i != cfg.route[f]]
    acts += [Action("gateway", g, 1 - cfg.gw[g]) for g in range(len(sim.GROUPS))]
    for e in range(sim.L):
        acts += [Action("cap", e, v) for v in (0.0, *CAP_LEVELS) if v != cfg.cap[e]]
        acts.append(Action("prio", e, float(not cfg.prio[e])))
    if not cfg.is_default():
        acts.append(Action("revert"))
    return acts


# ----------------------------------------------------------------------------------------------- features
def controller_loads(cfg, rates_epoch, failed):
    """Offered load per class and link [3, L] for the mean flow rates of an epoch [NF] under configuration cfg."""
    lc, _ = sim.class_loads(cfg, rates_epoch[:, None], failed)
    return lc[..., 0]


def raw_features(out, cfgs, exo, e0=0):
    """Per-epoch raw features [E, L, 9] of a simulated run (measurements NaN when missing). Loads come from the
    epoch-mean flow counters mapped onto the configuration of the epoch, as an SDN controller computes them."""
    E = len(out["rtt"]); m = out["measured"]
    F = np.full((E, sim.L, len(FEAT)), np.nan, np.float32)
    F[..., 0] = np.where(m, out["rtt"], np.nan); F[..., 1] = np.where(m, out["jitter"], np.nan)
    F[..., 2] = np.where(m, out["loss"], np.nan)
    for k in range(E):
        sl = slice((e0 + k) * sim.STEP, (e0 + k + 1) * sim.STEP)
        lc = controller_loads(cfgs[k], exo.rates[:, sl].mean(1), exo.capf[:, sl].min(1) <= 0)
        F[k, :, 3], F[k, :, 4], F[k, :, 5] = lc.sum(0), lc[0], lc[2]
    F[..., 6] = np.array([c.prio for c in cfgs]); F[..., 7] = np.array([c.cap for c in cfgs]); F[..., 8] = 0.0
    return F


class Normalizer:
    def __init__(self, F):                                        # F [N, L, 9] raw training features
        Z = np.log1p(np.clip(F[..., :LOGF], 0, None)).reshape(-1, LOGF)
        self.mu = np.nanmean(Z, 0); self.sd = np.nanstd(Z, 0) + 1e-6

    def __call__(self, F):
        """Raw [T, L, 9] -> model input [T, L, 11]: normalized features with forward-filled measurements,
        observation mask, and age of the last measurement (as in the real-data pipeline)."""
        T = len(F); obs = ~np.isnan(F[..., 0])
        Z = (np.log1p(np.clip(F[..., :LOGF], 0, None)) - self.mu) / self.sd
        Zf = pd.DataFrame(Z.reshape(T, -1)).ffill().to_numpy().reshape(Z.shape)
        Zf = np.nan_to_num(Zf, nan=0.0)
        age = np.zeros((T, sim.L), np.float32); run = np.zeros(sim.L, np.float32)
        for t in range(T):
            run = np.where(obs[t], 0, np.minimum(run + 1, 20)); age[t] = run
        return np.concatenate([Zf, F[..., 6:8], F[..., 8:9] / 100.0, obs[..., None], (age / 20.0)[..., None]],
                              -1).astype(np.float32)


def post_step(F_t, cfg_post, rates_epoch, failed):
    """Raw features of the last step under a candidate configuration: loads re-mapped, flags of the candidate."""
    G = F_t.copy()
    lc = controller_loads(cfg_post, rates_epoch, failed)
    new = lc.sum(0).astype(np.float32)
    G[:, 8] = new - F_t[:, 3]
    G[:, 3], G[:, 4], G[:, 5] = new, lc[0], lc[2]
    G[:, 6], G[:, 7] = cfg_post.prio, cfg_post.cap
    return G


# ----------------------------------------------------------------------------------------------- data generation
def explore(exo, seed, p_act=0.2, p_reset=0.03, dwell=4, net=None):
    """Run a scenario under a random exploration policy: when no action was taken in the last `dwell` epochs,
    a random action (or a reset to the default configuration) is applied with probability p_act (p_reset).
    Returns per-epoch raw features (under the configuration of the epoch), the post-action last-step features,
    the configurations, the actions, and the simulated outcomes."""
    rng = np.random.default_rng(seed); net = sim.Network(exo) if net is None else net; E = net.E
    cfgs = [sim.Config()]; state = None; outs = []; posts = np.zeros((E, sim.L, len(FEAT)), np.float32)
    acts = []; last = -dwell
    for k in range(E):
        o, state = net.run(lambda _k: cfgs[k], k, k + 1, state); outs.append(o)
        Fk = raw_features(o, [cfgs[k]], exo, k)[0]
        a = NOOP
        if k - last >= dwell:
            u = rng.uniform()
            if u < p_reset and not cfgs[k].is_default():
                a = Action("revert")
            elif u < p_reset + p_act:
                cand = all_actions(cfgs[k]); a = cand[int(rng.integers(len(cand)))]
            if a != NOOP:
                last = k
        nxt = apply(cfgs[k], a)
        sl = slice(k * sim.STEP, (k + 1) * sim.STEP); failed = exo.capf[:, sl].min(1) <= 0
        posts[k] = post_step(Fk, nxt, exo.rates[:, sl].mean(1), failed)
        cfgs.append(nxt); acts.append(a)
    out = {key: np.concatenate([o[key] for o in outs]) for key in outs[0]}
    return {"F": raw_features(out, cfgs[:E], exo), "post": posts, "cfgs": cfgs[:E], "acts": acts, "out": out}


def windows(run, norm):
    """Windows X [N, W, L, 11] whose last step is the post-action step, with targets at t + h for h in HORIZONS.
    A window is kept only if the configuration chosen at t stays in place up to t + max(h), so that every target
    is the outcome of that configuration."""
    F, P, cfgs, out = run["F"], run["post"], run["cfgs"], run["out"]
    E = len(F); hmax = max(HORIZONS); v = sim.violation(out)
    keys = [c.key() for c in cfgs]
    idx = np.array([t for t in range(W - 1, E - hmax) if all(keys[t + h] == keys[t + 1] for h in range(2, hmax + 1))])
    X = np.stack([_window(F, P, norm, t) for t in idx]).astype(np.float32)
    Y = np.stack([np.stack([np.stack([out["rtt"][t + h], out["jitter"][t + h], out["loss"][t + h]], -1)
                            for h in HORIZONS], 1) for t in idx])                             # [N, L, H, 3]
    Ym = np.stack([np.stack([out["measured"][t + h] for h in HORIZONS], 1) for t in idx])
    V = np.stack([np.stack([v[t + h] for h in HORIZONS], 1) for t in idx]).astype(np.float32)
    Y = np.where(Ym[..., None], Y, np.nan).astype(np.float32)
    acted = np.array([run["acts"][t] != NOOP for t in idx])
    return {"X": X, "Y": Y, "Ym": Ym, "V": V, "t": idx, "acted": acted}


def _window(F, P, norm, t):
    """Normalized window ending at t with the post-action last step. The forward fill and the age features use
    only the window itself plus a warm-up of W epochs, so that they match decision-time computation."""
    lo = max(0, t - 2 * W + 1)
    raw = np.concatenate([F[lo:t], P[t:t + 1]])
    return norm(raw)[-W:]
