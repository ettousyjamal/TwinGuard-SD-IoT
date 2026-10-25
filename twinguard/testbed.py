"""Controller side of the Mininet / Open vSwitch testbed: OpenFlow compilation of configurations and the environment
that lets the unchanged agent loop (agents.run_loop) and exploration policy (simdata.explore) drive the testbed.

The data plane (testbed/dataplane.py) runs in a privileged container; this module launches it, sends it the topology,
the scaled traffic trace, and the capacity trace of a scenario, and then, epoch by epoch, installs the flow tables,
meters, and queue assignments of the configuration chosen by the controller and collects the probe statistics, the
ingress flow counters, and the port states. Rates and capacities are scaled by SCALE (10-Mbps links for the
100-Mbps links of the simulator); measured rates are scaled back, so the twin sees the units it was trained on.
"""
import json
import socket
import subprocess
import time
from pathlib import Path

import numpy as np

from . import sim

SCALE = 0.1
IMAGE = "twinguard-testbed:latest"           # docker build -t twinguard-testbed:latest testbed/
ROOT = Path(__file__).resolve().parents[1]


def ip_of(r):
    return f"10.0.0.{int(r[1:])}"


def walk(src, path):
    seq = [src]
    for e in path:
        a, b = sim.LINKS[e].split("-"); seq.append(b if seq[-1] == a else a)
    return seq


def compile_config(cfg, failed, ports):
    """OpenFlow 1.3 tables of a configuration: per flow, one rule per hop (cookie = flow + 1; the first hop matches
    the host port), best-effort meters and critical-queue assignment on the links that have them, and the probe
    rules of every link (critical class). ports[(a, b)]: port of switch a towards b."""
    flows = {"s" + r[1:]: [] for r in sorted({x for l in sim.LINKS for x in l.split("-")})}
    meters = {sw: {} for sw in flows}
    for f, fl in enumerate(sim.FLOWS):
        src, dst = sim.src_of(f, cfg), fl.dst; path = sim.path_of(f, cfg, failed)
        match = f"udp,nw_dst={ip_of(dst)},tp_dst={5000 + f}"; ck = f"cookie={f + 1:#x}"
        if not path:
            flows["s" + src[1:]].append(f"{ck},priority=100,in_port=1,{match},actions=drop"); continue
        seq = walk(src, path)
        for i, (r, e) in enumerate(zip(seq[:-1], path)):
            acts = []
            if fl.cls == 2 and cfg.cap[e] > 0:
                acts.append(f"meter:{e + 1}")
            if fl.cls == 0 and cfg.prio[e]:
                acts.append("set_queue:1")
            acts.append(f"output:{ports[(r, seq[i + 1])]}")
            flows["s" + r[1:]].append(f"{ck},priority=100,{'in_port=1,' if i == 0 else ''}{match},actions={','.join(acts)}")
        flows["s" + dst[1:]].append(f"{ck},priority=100,{match},actions=drop")      # delivered to the IN-CSE side
    for e, l in enumerate(sim.LINKS):
        a, b = l.split("-"); q = "set_queue:1," if cfg.prio[e] else ""; ck = f"cookie={0x1000 + e:#x}"
        for (x, y, port) in ((a, b, 7000 + e), (b, a, 7500 + e)):
            m = f"udp,nw_dst={ip_of(y)},tp_dst={port}"
            flows["s" + x[1:]].append(f"{ck},priority=200,{m},actions={q}output:{ports[(x, y)]}")
            flows["s" + y[1:]].append(f"{ck},priority=200,{m},actions=output:1")
        if cfg.cap[e] > 0:
            for x in (a, b):
                meters["s" + x[1:]][str(e + 1)] = int(round(cfg.cap[e] * sim.C_NOM * SCALE * 1000))
    return {sw: sorted(r) for sw, r in flows.items()}, meters, {str(g): int(cfg.gw[g]) for g in range(len(sim.GROUPS))}


class Testbed:
    """Launches the data-plane container for one scenario and runs it epoch by epoch (Network-like interface)."""

    def __init__(self, exo, run_dir, port=9000, name="tg_testbed"):
        self.run_dir = Path(run_dir); io = self.run_dir / "io"; io.mkdir(parents=True, exist_ok=True)
        self.E = exo.rates.shape[1] // sim.STEP
        spec = {"routers": sorted({x for l in sim.LINKS for x in l.split("-")}, key=lambda r: int(r[1:])),
                "links": sim.LINKS, "capacity_mbps": sim.C_NOM * SCALE,
                "ip": {r: ip_of(r) for l in sim.LINKS for r in l.split("-")},
                "flows": [{"f": f, "cls": int(fl.cls), "group": fl.group, "src_options": list(fl.src_options),
                           "dst_ip": ip_of(fl.dst)} for f, fl in enumerate(sim.FLOWS)],
                "active0": {str(g): 0 for g in range(len(sim.GROUPS))}}
        json.dump(spec, open(io / "spec.json", "w"))
        np.save(io / "rates.npy", (exo.rates * SCALE).astype(np.float32))
        np.save(io / "capf.npy", exo.capf.astype(np.float32))
        # the controller only learns what it measures: rates and capacities are filled in epoch by epoch
        self.exo = sim.Exo(rates=np.zeros_like(exo.rates), capf=np.ones_like(exo.capf), dnoise=None, base=None,
                           mnoise=None, umiss=None, events=[])
        self.name = name
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
        subprocess.run(["docker", "run", "-d", "--rm", "--privileged", "--name", name, "-e", "HOST_UID=1000",
                        "-v", "/lib/modules:/lib/modules:ro", "-v", f"{ROOT / 'testbed'}:/work:ro",
                        "-v", f"{io}:/io", "-p", f"127.0.0.1:{port}:9000", IMAGE,
                        "python3", "/work/dataplane.py", "--io", "/io"], check=True, capture_output=True)
        for _ in range(240):
            if "data plane ready" in subprocess.run(["docker", "logs", name], capture_output=True, text=True).stdout:
                break
            time.sleep(0.5)
        else:
            raise RuntimeError(subprocess.run(["docker", "logs", name], capture_output=True, text=True).stdout)
        self.sock = socket.create_connection(("127.0.0.1", port)); self.f = self.sock.makefile("rw")
        self.ports = {tuple(k.split("-")): v for k, v in json.load(open(io / "ports.json"))["ports"].items()}
        self.down = set(); self.timing = []
        self.call({"cmd": "start"})

    def call(self, msg):
        self.f.write(json.dumps(msg) + "\n"); self.f.flush()
        out = json.loads(self.f.readline())
        if "error" in out:
            raise RuntimeError(out["error"])
        return out

    def run(self, cfg_of_epoch, e0, e1, state=None):
        assert e1 == e0 + 1
        cfg = cfg_of_epoch(e0); failed = np.array([e in self.down for e in range(sim.L)])
        t = time.time()
        flows, meters, active = compile_config(cfg, failed, self.ports)
        ap = self.call({"cmd": "apply", "flows": flows, "meters": meters, "active": active})
        compile_ms = 1000 * (time.time() - t) - ap["ms"]
        m = self.call({"cmd": "measure", "k": e0})
        self.timing.append({"k": e0, "compile_ms": compile_ms, "install_ms": ap["ms"], "switches": ap["switches"],
                            "measure_ms": m["measure_ms"]})
        rtt = np.full(sim.L, np.nan); jit = np.full(sim.L, np.nan); loss = np.full(sim.L, np.nan)
        for e, l in enumerate(sim.LINKS):
            p = m["probes"].get(l)
            if p and p[0] > 0 and p[1] > 0:
                rtt[e], jit[e], loss[e] = p[2], p[3], 100.0 * (p[0] - p[1]) / p[0]
        self.down = {sim.LINKS.index(l) for l in m["down"]}
        rates = np.zeros(sim.NF)
        for f, b in m["flow_bytes"].items():
            rates[int(f)] = b * 8 / 1e6 / sim.STEP / SCALE
        sl = slice(e0 * sim.STEP, e1 * sim.STEP)
        self.exo.rates[:, sl] = rates[:, None]
        self.exo.capf[:, sl] = np.where([e in self.down for e in range(sim.L)], 0.0, 1.0)[:, None]
        fl_now = np.array([e in self.down for e in range(sim.L)])
        lc, _ = sim.class_loads(cfg, rates[:, None], fl_now)
        meas = ~np.isnan(rtt) & ~fl_now
        out = {"rtt": rtt[None], "jitter": jit[None], "loss": loss[None], "measured": meas[None],
               "load": lc.sum(0)[:, 0][None], "load_c0": lc[0][:, 0][None], "load_c2": lc[2][:, 0][None],
               "rho": (lc.sum(0)[:, 0] / sim.C_NOM)[None],
               "true_rtt": np.nan_to_num(rtt, nan=0.0)[None], "true_loss": np.nan_to_num(loss, nan=0.0)[None]}
        return out, None

    def close(self):
        try:
            self.call({"cmd": "stop"})
        except Exception:                                   # noqa: BLE001 - the container may already be gone
            pass
        subprocess.run(["docker", "rm", "-f", self.name], capture_output=True)
