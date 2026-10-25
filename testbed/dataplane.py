"""Mininet / Open vSwitch data plane of the TwinGuard-SD-IoT testbed (runs as root inside the container).

One OVS switch (OpenFlow 1.3, fail-mode secure: no forwarding without controller rules) per router of the topology,
one host per router (MN-CSE gateway or IN-CSE side), and one shaped link per topology link: linux-htb at the scaled
capacity with two queues (best effort / critical, strict priority via HTB priorities) and a 60-packet buffer, i.e.
about 50 ms at the scaled capacity as in the simulator. Hosts add 0.25 ms of delay on their access interface.

Traffic: every host runs a generator that sends the UDP flows of the device groups attached to it (critical MQTT
messages, telemetry) and its best-effort transfers, with Poisson packet arrivals at the per-second rates of the
scenario trace (rates.npy, already scaled), DSCP-marked by class. A group sends from its active MN-CSE gateway only
(ctrl.json, written when the controller moves a group). Every link is probed by its first endpoint: 50 critical-class
probes per second to an echo server on the other endpoint; per 15-s epoch the prober logs probes sent and returned,
mean RTT, and RTT standard deviation (jitter).

Impairments: capacity degradations change the HTB rates of both directions of a link; failures set both ends down.
The controller talks to this process over a JSON-lines TCP socket (port 9000):
  {"cmd": "start"}                                   -> {"t0": epoch-0 start time}
  {"cmd": "apply", "flows": {sw: [rule, ...]}, "meters": {sw: {id: kbps}}, "active": {group: host}}
                                                     -> {"ms": install time, "switches": n changed}
  {"cmd": "measure", "k": k}                         -> probe statistics, flow byte counters, links down
  {"cmd": "stop"}
"""
import argparse
import json
import os
import socket
import subprocess
import threading
import time

import numpy as np
from mininet.net import Mininet
from mininet.node import OVSSwitch

HERE = os.path.dirname(os.path.abspath(__file__))
STEP = 15


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True)


class DataPlane:
    def __init__(self, io):
        self.io = io
        self.spec = json.load(open(os.path.join(io, "spec.json")))
        self.capf = np.load(os.path.join(io, "capf.npy"))            # [L, S] capacity factor per second
        self.cap_bps = int(self.spec["capacity_mbps"] * 1e6)
        self.routers = self.spec["routers"]; self.links = self.spec["links"]
        self.net = Mininet(switch=OVSSwitch, controller=None, autoStaticArp=True)
        self.h, self.s = {}, {}
        for i, r in enumerate(self.routers):
            self.h[r] = self.net.addHost("h" + r[1:], ip=self.spec["ip"][r] + "/16")
            self.s[r] = self.net.addSwitch("s" + r[1:], protocols="OpenFlow13", failMode="secure",
                                           dpid="%016x" % (i + 1))
            self.net.addLink(self.h[r], self.s[r])                       # host on switch port 1
        self.port, self.intf = {}, {}
        for l in self.links:
            a, b = l.split("-")
            lk = self.net.addLink(self.s[a], self.s[b])
            self.port[(a, b)] = self.s[a].ports[lk.intf1]; self.port[(b, a)] = self.s[b].ports[lk.intf2]
            self.intf[(a, b)] = lk.intf1.name; self.intf[(b, a)] = lk.intf2.name
        self.net.start()
        self.qos = {}
        for (a, b), ifn in self.intf.items():
            sh(f"ip link set {ifn} txqueuelen 60")
            self.qos[(a, b)] = self._set_qos(ifn, self.cap_bps)
        for r, h in self.h.items():
            h.cmd(f"tc qdisc add dev {h.defaultIntf().name} root netem delay 0.25ms limit 10000")
        self.cur_rate = {l: 1.0 for l in self.links}
        self.installed = {}; self.meters = {}; self.last_bytes = {}; self.t0 = None; self.procs = []
        json.dump({"ports": {f"{a}-{b}": p for (a, b), p in self.port.items()}}, open(os.path.join(io, "ports.json"), "w"))

    def _set_qos(self, ifn, rate):
        out = sh(f"ovs-vsctl -- set port {ifn} qos=@q -- --id=@q create qos type=linux-htb "
                 f"other-config:max-rate={rate} queues:0=@q0 queues:1=@q1 "
                 f"-- --id=@q0 create queue other-config:min-rate={rate // 100} other-config:max-rate={rate} "
                 f"other-config:priority=1 "
                 f"-- --id=@q1 create queue other-config:min-rate={rate} other-config:max-rate={rate} "
                 f"other-config:priority=0").stdout.split()
        return out                                                    # uuids of qos, q0, q1

    def _rate(self, link, factor):
        a, b = link.split("-")
        for d in ((a, b), (b, a)):
            ifn = self.intf[d]
            if factor <= 0:
                sh(f"ip link set {ifn} down")
                continue
            if self.cur_rate[link] <= 0:
                sh(f"ip link set {ifn} up")
            rate = int(self.cap_bps * factor)
            qos, q0, q1 = self.qos[d]
            sh(f"ovs-vsctl set qos {qos} other-config:max-rate={rate} -- set queue {q0} other-config:max-rate={rate} "
               f"other-config:min-rate={rate // 100} -- set queue {q1} other-config:max-rate={rate} "
               f"other-config:min-rate={rate}")
        self.cur_rate[link] = factor

    def impairments(self):
        """Follow the capacity-factor trace of the scenario (degradations and failures), second by second."""
        S = self.capf.shape[1]
        while self.running:
            s = int(time.time() - self.t0)
            if 0 <= s < S:
                for e, l in enumerate(self.links):
                    f = round(float(self.capf[e, s]), 2)
                    if f != self.cur_rate[l]:
                        self._rate(l, f)
            time.sleep(0.25)

    def start(self):
        self.t0 = time.time() + 3.0
        json.dump({"t0": self.t0, "active": self.spec["active0"]}, open(os.path.join(self.io, "ctrl.json"), "w"))
        for r, h in self.h.items():
            self.procs.append(h.popen(["python3", os.path.join(HERE, "agents_host.py"), self.io, r],
                                      stdout=open(os.path.join(self.io, f"host_{r}.log"), "w"),
                                      stderr=subprocess.STDOUT))
        self.running = True
        threading.Thread(target=self.impairments, daemon=True).start()
        return {"t0": self.t0}

    def apply(self, msg):
        t = time.time(); n = 0
        for sw, ms in msg["meters"].items():                         # meters before the rules that use them
            have = self.meters.setdefault(sw, {})
            for mid, kbps in ms.items():
                if have.get(mid) != kbps:
                    verb = "mod-meter" if mid in have else "add-meter"
                    sh(f"ovs-ofctl -O OpenFlow13 {verb} {sw} meter={mid},kbps,band=type=drop,rate={kbps}")
                    have[mid] = kbps
        for sw, rules in msg["flows"].items():
            if self.installed.get(sw) == rules:
                continue
            fn = os.path.join(self.io, f"flows_{sw}.txt")
            open(fn, "w").write("\n".join(rules) + "\n")
            r = sh(f"ovs-ofctl -O OpenFlow13 replace-flows {sw} {fn}")
            if r.returncode:
                return {"error": r.stderr}
            self.installed[sw] = rules; n += 1
        ctrl = json.load(open(os.path.join(self.io, "ctrl.json")))
        if msg.get("active") and ctrl["active"] != msg["active"]:
            ctrl["active"] = msg["active"]
            tmp = os.path.join(self.io, "ctrl.tmp"); json.dump(ctrl, open(tmp, "w")); os.replace(tmp, os.path.join(self.io, "ctrl.json"))
        return {"ms": 1000 * (time.time() - t), "switches": n}

    def measure(self, k):
        end = self.t0 + STEP * (k + 1) + 0.8
        while time.time() < end:
            time.sleep(min(0.05, end - time.time()))
        t = time.time()
        probes = {}
        for r in self.routers:
            fn = os.path.join(self.io, f"probe_{r}.log")
            if not os.path.exists(fn):
                continue
            for line in open(fn):
                p = line.split()
                if int(p[1]) == k:
                    probes[p[0]] = [int(p[2]), int(p[3]), float(p[4]), float(p[5])]
        flows = {}
        for r in self.routers:                                        # ingress counters (cookie = flow + 1)
            sw = "s" + r[1:]
            for line in sh(f"ovs-ofctl -O OpenFlow13 dump-flows {sw}").stdout.splitlines():
                if "cookie=" not in line or "n_bytes=" not in line:
                    continue
                ck = int(line.split("cookie=")[1].split(",")[0], 16)
                if ck == 0 or ck >= 0x1000:
                    continue
                if "in_port=1," not in line and "in_port=1 " not in line:
                    continue                                          # counted once, at the ingress switch
                nb = int(line.split("n_bytes=")[1].split(",")[0])
                last = self.last_bytes.get((sw, ck), 0)
                d = nb - last if nb >= last else nb                   # a re-installed rule restarts from zero
                self.last_bytes[(sw, ck)] = nb
                flows[ck - 1] = flows.get(ck - 1, 0) + d
        down = [l for l in self.links if self.cur_rate[l] <= 0]
        return {"k": k, "probes": probes, "flow_bytes": {str(f): b for f, b in flows.items()}, "down": down,
                "measure_ms": 1000 * (time.time() - t)}

    def stop(self):
        self.running = False
        for p in self.procs:
            p.terminate()
        self.net.stop()
        uid = os.environ.get("HOST_UID", "1000")
        sh(f"chown -R {uid}:{uid} {self.io}")                         # leave the outputs to the host user


def serve(dp, port):
    srv = socket.socket(); srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", port)); srv.listen(1)
    print("data plane ready", flush=True)
    conn, _ = srv.accept(); f = conn.makefile("rw")
    for line in f:
        msg = json.loads(line); cmd = msg["cmd"]
        if cmd == "start":
            out = dp.start()
        elif cmd == "apply":
            out = dp.apply(msg)
        elif cmd == "measure":
            out = dp.measure(msg["k"])
        elif cmd == "stop":
            f.write(json.dumps({"ok": True}) + "\n"); f.flush(); dp.stop(); break
        else:
            out = {"error": "unknown command"}
        f.write(json.dumps(out) + "\n"); f.flush()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--io", required=True); ap.add_argument("--port", type=int, default=9000)
    a = ap.parse_args()
    sh("service openvswitch-switch start")
    serve(DataPlane(a.io), a.port)
