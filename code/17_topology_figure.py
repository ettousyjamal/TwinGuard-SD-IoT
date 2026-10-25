"""Figure of the two additional simulator topologies (NSFNET and the two-tier IoT access network), in the style of the
dataset topology: thick links carry at least 30% mean utilization under static shortest-path routing (nominal law,
three 3-h scenarios); the IN-CSE router is drawn with a double circle and the MN-CSE gateway routers are shaded.
Run from the repository root:  python code/17_topology_figure.py   (writes results/figs/fig_topologies.png)
"""
import json
import subprocess
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
mpl.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 9, "savefig.dpi": 600,
                     "savefig.bbox": "tight"})

POS = {"nsfnet": {1: (0.5, 3.4), 2: (0.0, 2.0), 3: (0.8, 0.6), 4: (2.2, 2.6), 5: (3.4, 1.6), 6: (3.8, 0.2),
                  7: (4.7, 1.9), 8: (5.3, 3.0), 9: (6.9, 2.3), 10: (6.2, 1.0), 11: (6.2, 3.9), 12: (8.1, 3.5),
                  13: (8.6, 2.0), 14: (7.4, 0.4)},
       # core square R1-R4; each pair of dual-homed access routers sits outside the side of its two core routers
       "two_tier": {1: (0.0, 2.4), 2: (2.4, 2.4), 3: (2.4, 0.0), 4: (0.0, 0.0),
                    8: (0.7, 3.6), 12: (1.7, 3.6), 5: (3.6, 1.7), 9: (3.6, 0.7),
                    6: (1.7, -1.2), 10: (0.7, -1.2), 7: (-1.2, 0.7), 11: (-1.2, 1.7)}}
TITLE = {"nsfnet": "(a) NSFNET (14 routers, 21 links)", "two_tier": "(b) Two-tier IoT (12 / 22)"}


def info(topo):
    """Links, IN-CSE, gateway routers, and mean static-routing utilization per link (computed by the simulator)."""
    code = ("import sys, json; sys.path.insert(0, 'code'); import numpy as np\n"
            "from code_sim_common import sim\n"
            "U = []\n"
            "for s in range(800, 803):\n"
            "    ex = sim.make_exo(s, 3.0, 'test')\n"
            "    o, _ = sim.Network(ex).run(lambda k: sim.Config(), 0, ex.rates.shape[1] // sim.STEP)\n"
            "    U.append(o['load'] / sim.C_NOM)\n"
            "u = np.concatenate(U).mean(0)\n"
            "gw = sorted({g[1] for g in sim.GROUPS} | {g[2] for g in sim.GROUPS})\n"
            "print(json.dumps({'links': sim.LINKS, 'in_cse': sim.IN_CSE, 'gw': gw, 'util': u.tolist()}))\n")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True,
                         env={**__import__("os").environ, "TG_TOPOLOGY": topo})
    return json.loads(out.stdout.strip().splitlines()[-1])


def draw(ax, topo):
    d = info(topo); pos = {f"R{k}": v for k, v in POS[topo].items()}
    for l, u in zip(d["links"], d["util"]):
        a, b = l.split("-"); (x0, y0), (x1, y1) = pos[a], pos[b]
        ax.plot([x0, x1], [y0, y1], color="black", lw=2.6 if u >= 0.3 else 0.8, zorder=1, solid_capstyle="butt")
    for r, (x, y) in pos.items():
        fc = "0.82" if r in d["gw"] else "white"
        ax.add_patch(plt.Circle((x, y), 0.34, facecolor=fc, edgecolor="black", lw=0.9, zorder=2))
        if r == d["in_cse"]:
            ax.add_patch(plt.Circle((x, y), 0.41, facecolor="none", edgecolor="black", lw=0.9, zorder=2))
        ax.text(x, y, r, ha="center", va="center", fontsize=6.3 if len(r) > 2 else 8, zorder=3)
    ax.set_aspect("equal"); ax.axis("off")
    xs, ys = zip(*pos.values()); ax.set_xlim(min(xs) - 0.6, max(xs) + 0.6); ax.set_ylim(min(ys) - 0.6, max(ys) + 0.6)


def main(out):
    fig, axes = plt.subplots(1, 2, figsize=(4.75, 2.55), gridspec_kw={"width_ratios": [1.75, 1]})
    draw(axes[0], "nsfnet"); draw(axes[1], "two_tier")
    h = [plt.Line2D([], [], color="black", lw=2.6), plt.Line2D([], [], color="black", lw=0.8),
         (plt.Line2D([], [], marker="o", ls="", markersize=11, markerfacecolor="none", markeredgecolor="black"),
          plt.Line2D([], [], marker="o", ls="", markersize=8, markerfacecolor="white", markeredgecolor="black")),
         plt.Line2D([], [], marker="o", ls="", markersize=10, markerfacecolor="0.82", markeredgecolor="black")]
    from matplotlib.legend_handler import HandlerTuple
    fig.legend(h, ["mean utilization $\\geq$ 30% (static routing)", "other links", "IN-CSE", "MN-CSE gateway"],
               loc="lower center", ncol=2, fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, -0.04),
               handler_map={tuple: HandlerTuple(ndivide=1)})
    fig.tight_layout(rect=(0, 0.13, 1, 0.92), w_pad=0.5)
    for ax, topo in zip(axes, ("nsfnet", "two_tier")):         # titles on one line, above both panels
        fig.text(ax.get_position().x0, 0.95, TITLE[topo], ha="left", va="top", fontsize=8)
    fig.savefig(ROOT / out); plt.close(fig)
    print("written:", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/figs/fig_topologies.png")
