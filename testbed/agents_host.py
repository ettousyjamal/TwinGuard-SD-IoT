"""Per-host traffic generator, link prober, and probe echo server of the testbed (one process per Mininet host).

Flows: Poisson packet arrivals (1000-byte IP packets) at the per-second rate of the trace, DSCP-marked by class
(critical EF, telemetry AF11, best effort default); a device group sends only from its active gateway (ctrl.json).
Probes: for every link whose first endpoint is this router, 50 critical-class probes per second to the echo server
of the other endpoint; per epoch, the probes sent in the epoch, those returned within 0.5 s after its end, and the
mean and standard deviation of their RTT are appended to probe_<router>.log.
"""
import heapq
import json
import os
import random
import select
import socket
import struct
import sys
import time

import numpy as np

STEP, PROBE_GAP, PKT = 15, 0.02, 972             # 972 B of UDP payload = 1000-byte IP packets
TOS = {0: 0xb8, 1: 0x28, 2: 0x00}


def main(io, me):
    spec = json.load(open(os.path.join(io, "spec.json")))
    rates = np.load(os.path.join(io, "rates.npy"), mmap_mode="r")         # [NF, S] Mbps (testbed scale)
    ctrl = json.load(open(os.path.join(io, "ctrl.json"))); t0 = ctrl["t0"]; active = ctrl["active"]
    ip = spec["ip"]; rnd = random.Random(hash(me) & 0xffff)
    flows = [fl for fl in spec["flows"] if me in fl["src_options"]]
    socks = {}
    for fl in flows:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.setsockopt(socket.IPPROTO_IP, socket.IP_TOS, TOS[fl["cls"]])
        s.setblocking(False); socks[fl["f"]] = s
    probe, echo, peer = {}, {}, {}
    for e, l in enumerate(spec["links"]):
        a, b = l.split("-")
        if a == me:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.setsockopt(socket.IPPROTO_IP, socket.IP_TOS, 0xb8)
            s.bind((ip[a], 7500 + e)); s.setblocking(False); probe[e] = s; peer[e] = (ip[b], 7000 + e)
        if b == me:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.setsockopt(socket.IPPROTO_IP, socket.IP_TOS, 0xb8)
            s.bind((ip[b], 7000 + e)); s.setblocking(False); echo[e] = s
    payload = b"x" * PKT
    heap = [(t0 + rnd.random() * 0.01, "flow", fl["f"]) for fl in flows]
    heap += [(t0 + i * PROBE_GAP / max(1, len(probe)), "probe", e) for i, e in enumerate(probe)]
    heap.append((t0, "ctrl", -1)); heapq.heapify(heap)
    seq = {e: 0 for e in probe}; stats = {}; done = -1
    out = open(os.path.join(io, f"probe_{me}.log"), "a")
    by_f = {fl["f"]: fl for fl in flows}
    rsocks = list(probe.values()) + list(echo.values())
    while True:
        now = time.time()
        while heap and heap[0][0] <= now:
            t, kind, x = heapq.heappop(heap)
            if kind == "ctrl":
                try:
                    active = json.load(open(os.path.join(io, "ctrl.json")))["active"]
                except (OSError, ValueError):
                    pass
                heapq.heappush(heap, (t + 0.5, "ctrl", -1))
            elif kind == "probe":
                k = int((t - t0) // STEP)
                if k >= 0:
                    seq[x] += 1; st = stats.setdefault(k, {}).setdefault(x, [0, []]); st[0] += 1
                    try:
                        probe[x].sendto(struct.pack("!IIId", x, seq[x], k, time.time()) + b"p" * 44, peer[x])
                    except OSError:
                        pass
                heapq.heappush(heap, (t + PROBE_GAP, "probe", x))
            else:
                fl = by_f[x]; s_ = int(t - t0)
                on = fl["group"] < 0 or fl["src_options"][int(active[str(fl["group"])])] == me
                r = float(rates[x, s_]) if 0 <= s_ < rates.shape[1] else 0.0
                if on and r > 0:
                    try:
                        socks[x].sendto(payload, (fl["dst_ip"], 5000 + x))
                    except OSError:
                        pass
                    heapq.heappush(heap, (t + rnd.expovariate(r * 1e6 / 8000.0), "flow", x))
                else:
                    heapq.heappush(heap, (t0 + s_ + 1 + rnd.random() * 0.01, "flow", x))
        # replies and echoes
        timeout = max(0.0, min(heap[0][0] - time.time(), 0.005)) if heap else 0.005
        rd, _, _ = select.select(rsocks, [], [], timeout)
        for s in rd:
            try:
                data, addr = s.recvfrom(2048)
            except OSError:
                continue
            e = struct.unpack("!I", data[:4])[0]
            if e in echo and s is echo[e]:
                try:
                    s.sendto(data, addr)
                except OSError:
                    pass
            else:
                _, _, k, ts = struct.unpack("!IIId", data[:20])
                if k in stats and e in stats[k]:
                    stats[k][e][1].append(1000.0 * (time.time() - ts))
        # close epochs 0.5 s after their end
        kc = int((time.time() - t0 - 0.5) // STEP) - 1
        while done < kc:
            done += 1
            for e, (sent, rtts) in sorted(stats.pop(done, {}).items()):
                m = float(np.mean(rtts)) if rtts else float("nan"); sd = float(np.std(rtts)) if rtts else float("nan")
                out.write(f"{spec['links'][e]} {done} {sent} {len(rtts)} {m:.4f} {sd:.4f}\n")
            out.flush()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
