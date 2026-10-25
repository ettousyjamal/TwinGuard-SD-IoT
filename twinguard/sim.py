"""Flow-level simulator of a oneM2M-based SD-IoT network with SDN actions and exact counterfactuals.

Topology: the 9-router, 13-link topology of the public dataset. oneM2M deployment: device groups (ADN/ASN nodes)
attach to MN-CSE gateways hosted on edge routers and report to the IN-CSE at R5 over two traffic classes,
critical MQTT messages (alarms, actuation acknowledgements) and periodic CoAP/MQTT telemetry; HTTP best-effort
transfers (firmware distribution, video analytics, cloud synchronization) share the links.

Time runs in seconds; measurements are aggregated per 15-s epoch. Each link is one queue. Per second, the probe
delay is the propagation and processing delay plus the fluid backlog delay plus a load-dependent queueing term
s * rho / (1 - rho) (bursty arrivals), and the loss is the fluid overflow plus the M/M/1/K blocking probability.
With strict priority on a link, the probes (critical class) only queue behind critical traffic.

Every random quantity is exogenous and drawn in advance (traffic, impairments, noise, missing measurements), so
the outcome of any configuration is a deterministic function of the same draws: counterfactual outcomes of
alternative actions are exact (common random numbers).
"""
import itertools
from dataclasses import dataclass, field

import networkx as nx
import numpy as np
from scipy.signal import lfilter

LINKS = ["R1-R2", "R1-R9", "R2-R3", "R2-R9", "R3-R4", "R3-R8", "R4-R5", "R4-R6", "R6-R5", "R7-R6", "R8-R6",
         "R8-R7", "R9-R8"]
L = len(LINKS)
STEP = 15                                  # seconds per epoch
CLASSES = ("critical", "telemetry", "best_effort")
IN_CSE = "R5"
RTT_MAX, PLR_MAX = 10.0, 1.0               # SLA of the critical class (probe) on every link
C_NOM = 100.0                              # Mbps, nominal capacity of every link
BUF_MS = 50.0                              # buffer, in ms at nominal capacity
S_BURST = 0.9                              # ms, scale of the load-dependent queueing delay
K_BLOCK = 40                               # M/M/1/K buffer (packets) for the burst loss term
N_PROBE = 1000                             # probe packets per link and epoch (loss resolution 0.1%)
MISS_A, MISS_C = 0.58, 0.40                # dataset-like missing measurements (Network(missing="dataset"))
LOAD_SCALE = 1.45                          # traffic volume: busiest links reach the 80-90% utilization knee at peaks

# device groups: (name, primary MN-CSE gateway, alternative gateway, critical Mbps, telemetry Mbps)
GROUPS = [("G1", "R1", "R2", 1.6, 9.0), ("G2", "R9", "R1", 1.4, 8.0), ("G3", "R7", "R8", 1.8, 10.0),
          ("G4", "R2", "R3", 1.2, 7.0), ("G5", "R8", "R9", 1.5, 9.0)]
# best-effort HTTP transfers: (name, source, destination, Mbps when active)
BE_FLOWS = [("B1", "R3", "R5", 26.0), ("B2", "R2", "R6", 22.0), ("B3", "R7", "R5", 24.0), ("B4", "R9", "R4", 20.0)]


def _graph():
    g = nx.Graph()
    for i, l in enumerate(LINKS):
        a, b = l.split("-"); g.add_edge(a, b, idx=i)
    return g


G = _graph()


def k_paths(src, dst, k=3):
    """k shortest simple paths (hop count, ties broken deterministically), as lists of link indices."""
    out = []
    for p in itertools.islice(nx.shortest_simple_paths(G, src, dst), 12):
        out.append(tuple(G.edges[a, b]["idx"] for a, b in zip(p[:-1], p[1:])))
    out.sort(key=lambda q: (len(q), q))
    return out[:k]


@dataclass(frozen=True)
class Flow:
    name: str
    cls: int                 # 0 critical, 1 telemetry, 2 best effort
    group: int               # device group index, -1 for best effort
    src_options: tuple       # possible source routers (gateways) ; best effort: (src,)
    dst: str


def make_flows():
    flows = []
    for gi, (name, gw, alt, _, _) in enumerate(GROUPS):
        flows.append(Flow(f"{name}-crit", 0, gi, (gw, alt), IN_CSE))
        flows.append(Flow(f"{name}-tele", 1, gi, (gw, alt), IN_CSE))
    for name, src, dst, _ in BE_FLOWS:
        flows.append(Flow(name, 2, -1, (src,), dst))
    return flows


FLOWS = make_flows()
NF = len(FLOWS)
PATHS = {(s, f.dst): k_paths(s, f.dst) for f in FLOWS for s in f.src_options}


@dataclass
class Config:
    """SDN configuration. route[f]: path index; gw[g]: gateway index of device group g;
    cap[e]: best-effort meter on link e as a fraction of nominal capacity (0 = none); prio[e]: strict priority."""
    route: np.ndarray = field(default_factory=lambda: np.zeros(NF, int))
    gw: np.ndarray = field(default_factory=lambda: np.zeros(len(GROUPS), int))
    cap: np.ndarray = field(default_factory=lambda: np.zeros(L))
    prio: np.ndarray = field(default_factory=lambda: np.zeros(L, bool))

    def copy(self):
        return Config(self.route.copy(), self.gw.copy(), self.cap.copy(), self.prio.copy())

    def key(self):
        return (tuple(self.route), tuple(self.gw), tuple(np.round(self.cap, 3)), tuple(self.prio))

    def is_default(self):
        return not (self.route.any() or self.gw.any() or self.cap.any() or self.prio.any())


def src_of(f, cfg):
    fl = FLOWS[f]
    return fl.src_options[cfg.gw[fl.group]] if fl.group >= 0 else fl.src_options[0]


def path_of(f, cfg, failed):
    """Path of flow f; a path through a failed link is replaced by the first surviving alternative (fast failover)."""
    opts = PATHS[(src_of(f, cfg), FLOWS[f].dst)]
    p = opts[min(cfg.route[f], len(opts) - 1)]
    if failed is not None and any(failed[e] for e in p):
        for q in opts:
            if not any(failed[e] for e in q):
                return q
        return ()                                                 # disconnected
    return p


def incidence(cfg, failed=None):
    R = np.zeros((L, NF))
    for f in range(NF):
        for e in path_of(f, cfg, failed):
            R[e, f] = 1.0
    return R


CLS = np.array([f.cls for f in FLOWS])


def class_loads(cfg, rates, failed=None):
    """Offered load per link and class [3, L, T] for flow rates [NF, T], after best-effort meters."""
    R = incidence(cfg, failed)
    lam = rates.copy()
    if cfg.cap.any():
        be = CLS == 2
        lam3 = R[:, be] @ rates[be]                                # best-effort load per link [L, T]
        lim = np.where(cfg.cap[:, None] > 0, cfg.cap[:, None] * C_NOM, np.inf)
        fac = np.minimum(1.0, lim / np.maximum(lam3, 1e-9))         # meter factor per link [L, T]
        for f in np.where(be)[0]:
            on = R[:, f] > 0
            if on.any():
                lam[f] = rates[f] * fac[on].min(0)
    return np.stack([R[:, CLS == c] @ lam[CLS == c] for c in range(3)]), lam


@dataclass
class Exo:
    """All exogenous randomness of one scenario (seconds unless stated)."""
    rates: np.ndarray        # [NF, S] offered rate of every flow, Mbps
    capf: np.ndarray         # [L, S] capacity factor (1 nominal, 0 failed)
    dnoise: np.ndarray       # [L, S] multiplicative noise of the queueing delay
    base: np.ndarray         # [L] propagation + processing delay, ms
    mnoise: np.ndarray       # [L, E, 3] measurement noise (rtt, jitter, loss), standard normal
    umiss: np.ndarray        # [L, E] uniform draws for missing measurements
    events: list             # (kind, start_s, end_s, target, magnitude)


def make_exo(seed, hours, regime="train"):
    """Traffic and impairments. regime: 'train' / 'test' (same law), 'shift' (severe capacity degradations and
    link failures, stronger alarm storms: never seen in training)."""
    rng = np.random.default_rng(seed)
    S = int(hours * 3600); S -= S % STEP; E = S // STEP; t = np.arange(S)
    phase = rng.uniform(0, 2 * np.pi)
    diurnal = 1.0 + 0.35 * np.sin(2 * np.pi * t / 86400.0 * 6 + phase)    # compressed day: 4-h period

    def ar1(tau, sigma):                                          # stationary AR(1), time constant tau seconds
        a = np.exp(-1.0 / tau); z = rng.standard_normal(S) * sigma * np.sqrt(1 - a * a)
        return lfilter([1.0], [1.0, -a], z)

    rates = np.zeros((NF, S)); events = []
    for f, fl in enumerate(FLOWS):
        if fl.cls < 2:
            base = GROUPS[fl.group][3 + fl.cls]
            rates[f] = base * diurnal * np.exp(ar1(60.0, 0.15) + 0.05 * rng.standard_normal(S))
        else:
            peak = BE_FLOWS[f - 2 * len(GROUPS)][3]; on = np.zeros(S, bool); i = int(rng.integers(0, 600))
            while i < S:                                          # ON/OFF transfers
                d_on = int(rng.exponential(300)); on[i:i + d_on] = True; i += d_on + int(rng.exponential(600))
            rates[f] = np.where(on, peak * rng.uniform(0.7, 1.3), 0.0) * np.exp(0.08 * rng.standard_normal(S))
    # alarm storms / flash crowds on device groups
    storm_rate, storm_mag = (1 / 1800, (3, 6)) if regime != "shift" else (1 / 1200, (5, 9))
    i = int(rng.exponential(1 / storm_rate))
    while i < S:
        g = int(rng.integers(len(GROUPS))); dur = int(rng.uniform(120, 900)); m = rng.uniform(*storm_mag)
        for f, fl in enumerate(FLOWS):
            if fl.group == g:
                rates[f, i:i + dur] *= m if fl.cls == 0 else 1 + (m - 1) / 3
        events.append(("storm", i, i + dur, g, m)); i += dur + int(rng.exponential(1 / storm_rate))
    capf = np.ones((L, S))
    # capacity degradations (e.g., interference on wireless backhaul); failures only in the shift regime
    deg_rate, deg_rng = (1 / 5400, (0.65, 0.9)) if regime != "shift" else (1 / 2400, (0.3, 0.55))
    i = int(rng.exponential(1 / deg_rate))
    while i < S:
        e = int(rng.integers(L)); dur = int(rng.uniform(300, 2400)); m = rng.uniform(*deg_rng)
        capf[e, i:i + dur] = np.minimum(capf[e, i:i + dur], m); events.append(("degradation", i, i + dur, e, m))
        i += dur + int(rng.exponential(1 / deg_rate))
    if regime == "shift":
        i = int(rng.exponential(5400))
        while i < S:
            e = int(rng.integers(L)); dur = int(rng.uniform(300, 1200))
            capf[e, i:i + dur] = 0.0; events.append(("failure", i, i + dur, e, 0.0)); i += dur + int(rng.exponential(5400))
    return Exo(rates=rates * LOAD_SCALE, capf=capf, dnoise=np.exp(0.5 * rng.standard_normal((L, S)) - 0.125),
               base=rng.uniform(0.8, 1.6, L), mnoise=rng.standard_normal((L, E, 3)), umiss=rng.uniform(size=(L, E)),
               events=events)


def p_block(rho, K=K_BLOCK):
    rho = np.clip(rho, 1e-9, 0.999999)
    return (1 - rho) * rho ** K / (1 - rho ** (K + 1))


class Network:
    """Simulates epochs under a configuration; state = backlogs (total and critical) per link, in Mb."""

    def __init__(self, exo, missing="nominal"):
        """missing: 'nominal' (about 2% of the probes time out, mostly under overload) or 'dataset' (missing
        measurements as frequent and as load-dependent as in the public dataset: about a fifth of the records, rare
        on lightly loaded links and most frequent on congested ones)."""
        self.exo = exo; self.E = exo.rates.shape[1] // STEP; self.missing = missing

    def run(self, cfg_of_epoch, e0, e1, state=None):
        """Simulate epochs [e0, e1); cfg_of_epoch(k) -> Config. Returns per-epoch measurements and the final state."""
        ex = self.exo
        bt, bh = (np.zeros(L), np.zeros(L)) if state is None else (state[0].copy(), state[1].copy())
        n = e1 - e0
        out = {k: np.zeros((n, L)) for k in ("rtt", "jitter", "loss", "load", "load_c0", "load_c2", "rho", "true_rtt",
                                              "true_loss")}
        out["measured"] = np.zeros((n, L), bool)
        for k in range(n):
            ep = e0 + k; cfg = cfg_of_epoch(ep); sl = slice(ep * STEP, (ep + 1) * STEP)
            capf = ex.capf[:, sl]; failed = capf.min(1) <= 0.0
            lc, _ = class_loads(cfg, ex.rates[:, sl], failed)             # [3, L, 15]
            tot = lc.sum(0); c = C_NOM * capf; buf = BUF_MS * C_NOM / 1000.0
            dl = np.zeros((L, STEP)); ls = np.zeros((L, STEP)); rho_p = np.zeros((L, STEP))
            for s in range(STEP):
                cs = np.maximum(c[:, s], 1e-6)
                prev_t, prev_h = bt, bh
                bt = np.clip(prev_t + tot[:, s] - cs, 0.0, buf)
                over_t = np.maximum(0.0, prev_t + tot[:, s] - cs - buf)
                bh = np.clip(prev_h + lc[0][:, s] - cs, 0.0, buf)
                over_h = np.maximum(0.0, prev_h + lc[0][:, s] - cs - buf)
                pr = cfg.prio
                rho = np.where(pr, lc[0][:, s], tot[:, s]) / cs
                back = np.where(pr, bh, bt)
                over = np.where(pr, over_h / np.maximum(lc[0][:, s], 1e-9), over_t / np.maximum(tot[:, s], 1e-9))
                rc = np.minimum(rho, 0.99)
                dl[:, s] = ex.base + 1000.0 * back / cs + S_BURST * rc / (1 - rc) * ex.dnoise[:, ep * STEP + s]
                ls[:, s] = np.minimum(1.0, over + p_block(rho))
                rho_p[:, s] = rho
            z = ex.mnoise[:, ep]
            p = ls.mean(1)
            rtt = dl.mean(1) * (1 + 0.03 * z[:, 0])
            jit = dl.std(1) + 0.05 * np.abs(z[:, 1])
            loss = np.clip(p + z[:, 2] * np.sqrt(p * (1 - p) / N_PROBE), 0, 1) * 100
            rho_m = rho_p.mean(1)
            if self.missing == "dataset":
                p_miss = 0.02 + MISS_A / (1 + np.exp(-(rho_m - MISS_C) / 0.12))
            else:
                p_miss = 0.02 + 0.28 / (1 + np.exp(-(rho_m - 0.95) / 0.04))  # probes time out under overload
            meas = (ex.umiss[:, ep] >= p_miss) & ~failed
            out["rtt"][k], out["jitter"][k], out["loss"][k] = rtt, jit, loss
            out["true_rtt"][k], out["true_loss"][k] = dl.mean(1), p * 100
            out["load"][k], out["load_c0"][k], out["load_c2"][k] = tot.mean(1), lc[0].mean(1), lc[2].mean(1)
            out["rho"][k] = rho_m; out["measured"][k] = meas
        return out, (bt, bh)


# ----------------------------------------------------------------------------------------------- other topologies
NSFNET = [(1, 2), (1, 3), (1, 8), (2, 3), (2, 4), (3, 6), (4, 5), (4, 11), (5, 6), (5, 7), (6, 10), (6, 14), (7, 8),
          (8, 9), (9, 10), (9, 12), (9, 13), (11, 12), (11, 13), (12, 14), (13, 14)]


def _two_tier():
    """Two-tier IoT access network: a meshed core of 4 routers and 8 dual-homed access routers (12 routers, 22 links)."""
    core = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
    acc = [(a, c) for i, a in enumerate(range(5, 13)) for c in ((1 + i) % 4 + 1, (2 + i) % 4 + 1)]
    return core + acc


TOPOLOGIES = {
    # name: (links, IN-CSE router, device groups (name, gateway, alternative, critical, telemetry Mbps),
    #        best-effort transfers (name, src, dst, Mbps), load scale)
    "dataset": (LINKS, IN_CSE, GROUPS, BE_FLOWS, LOAD_SCALE),
    "nsfnet": ([f"R{a}-R{b}" for a, b in NSFNET], "R9",
               [("G1", "R1", "R2", 1.6, 9.0), ("G2", "R3", "R1", 1.4, 8.0), ("G3", "R14", "R12", 1.8, 10.0),
                ("G4", "R5", "R4", 1.2, 7.0), ("G5", "R11", "R13", 1.5, 9.0)],
               [("B1", "R2", "R10", 26.0), ("B2", "R7", "R12", 22.0), ("B3", "R6", "R9", 24.0),
                ("B4", "R4", "R13", 20.0)], None),
    "two_tier": ([f"R{a}-R{b}" for a, b in _two_tier()], "R1",
                 [("G1", "R5", "R6", 1.6, 9.0), ("G2", "R7", "R8", 1.4, 8.0), ("G3", "R9", "R10", 1.8, 10.0),
                  ("G4", "R11", "R12", 1.2, 7.0), ("G5", "R6", "R5", 1.5, 9.0)],
                 [("B1", "R8", "R1", 26.0), ("B2", "R10", "R2", 22.0), ("B3", "R12", "R1", 24.0),
                  ("B4", "R3", "R1", 20.0)], None),
}
TOPOLOGY = "dataset"


def set_topology(name, load_scale=None):
    """Rebind the module-level topology, deployment, and derived tables (all other modules use sim.<name>)."""
    global LINKS, L, IN_CSE, GROUPS, BE_FLOWS, LOAD_SCALE, G, FLOWS, NF, PATHS, CLS, TOPOLOGY
    links, in_cse, groups, be, scale = TOPOLOGIES[name]
    LINKS, L, IN_CSE, GROUPS, BE_FLOWS = list(links), len(links), in_cse, groups, be
    LOAD_SCALE = load_scale if load_scale is not None else (scale if scale is not None else 1.45)
    G = _graph(); FLOWS = make_flows(); NF = len(FLOWS)
    PATHS = {(s, f.dst): k_paths(s, f.dst) for f in FLOWS for s in f.src_options}
    CLS = np.array([f.cls for f in FLOWS]); TOPOLOGY = name


def violation(out):
    v = ((out["rtt"] > RTT_MAX) | (out["loss"] > PLR_MAX)).astype(np.float32)
    return np.where(out["measured"], v, np.nan)
