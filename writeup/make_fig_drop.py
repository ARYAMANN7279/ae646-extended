"""Downstream figure for the pressure-drop study: (a) pressure error vs effective-permeability error per model,
on the test split (filled) and on maps from an independent generator (hollow); (b) effective-permeability and
late-breakthrough (t95) error by training loss. Reads results/drop/<group>.json (falls back to prelim_<group>.json)."""
import json, sys
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = Path(__file__).resolve().parents[1]
C = {"l2": "#2a78d6", "h1": "#eb6834", "flux1": "#1baf7a"}; LAB = {"l2": "$L^2$ loss", "h1": "$H^1$ loss", "flux1": "face-flux loss"}
M = {"fno": "o", "unet": "s", "deeponet": "^"}; ARCH = {"fno": "FNO", "unet": "U-Net", "deeponet": "DeepONet"}
INK, MUT, GRID = "#0b0b0b", "#52514e", "#e8e7e3"
plt.rcParams.update({"font.family": "serif", "font.serif": ["TeX Gyre Termes", "Times New Roman", "DejaVu Serif"], "font.size": 8,
                     "axes.edgecolor": MUT, "axes.labelcolor": INK, "xtick.color": MUT, "ytick.color": MUT, "axes.linewidth": 0.6})


def load(g):
    for f in (R / f"results/drop/{g}.json", R / f"results/drop/prelim_{g}.json"):
        if f.exists(): return json.load(open(f))
    return None


fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.3), gridspec_kw={"width_ratios": [1, 1.2]})
for arch in ("fno", "unet", "deeponet"):
    for loss in ("l2", "h1", "flux1"):
        r = load(f"d_{arch}_{loss}")
        if r is None: continue
        t, o = r["test"], r["ood"]["ID"]
        a.plot([100 * t["p"], 100 * o["p"]], [100 * t["keff"], 100 * o["keff"]], color=C[loss], lw=0.6, alpha=0.5, zorder=2)
        a.scatter(100 * t["p"], 100 * t["keff"], marker=M[arch], s=26, c=C[loss], edgecolor="white", lw=0.6, zorder=3)
        a.scatter(100 * o["p"], 100 * o["keff"], marker=M[arch], s=26, facecolor="white", edgecolor=C[loss], lw=1.0, zorder=3)
a.set_xscale("log"); a.set_yscale("log"); a.set_xlabel("pressure error (%)"); a.set_ylabel("effective-permeability error (%)")
xs = np.array([0.08, 5]); a.plot(xs, xs, color=MUT, lw=0.5, ls=":", zorder=1); a.text(3, 2.0, "equal", color=MUT, fontsize=6.5)
h = [plt.Line2D([], [], color=C[k], marker="o", ls="", ms=4.5, label=LAB[k]) for k in C]
h += [plt.Line2D([], [], color=MUT, marker="o", ls="", ms=4, mfc="white", label="other generator")]
h += [plt.Line2D([], [], color=MUT, marker=M[k], ls="", ms=4, label=ARCH[k]) for k in M]
a.legend(handles=h, frameon=False, fontsize=6.2, loc="upper left", ncol=2, handletextpad=0.2, columnspacing=0.6)
a.set_title("(a) pressure error does not predict $k_{\\mathrm{eff}}$ error", fontsize=8, loc="left", color=INK)

groups = [(arch, loss) for arch in ("fno", "unet") for loss in ("l2", "h1", "flux1")]
x = np.array([0, 1, 2, 3.6, 4.6, 5.6]); w = 0.38
for xi, (arch, loss) in zip(x, groups):
    r = load(f"d_{arch}_{loss}")
    if r is None: continue
    k, t95 = 100 * r["test"]["keff"], 100 * r["test"]["t95"]
    b.bar(xi - w / 2 - 0.01, k, w, color=C[loss], zorder=3); b.bar(xi + w / 2 + 0.01, t95, w, color=C[loss], alpha=0.55, zorder=3, hatch="////", edgecolor="white", lw=0)
    for xx, v in ((xi - w / 2, k), (xi + w / 2, t95)):
        b.text(xx, v * 1.12, f"{v:.1f}" if v < 10 else f"{v:.0f}", ha="center", fontsize=6.2, color=INK)
b.set_yscale("log"); b.set_ylim(0.3, 60); b.set_xticks(x); b.set_xticklabels([LAB[l].replace(" loss", "") for _, l in groups], fontsize=6.5)
b.text(1, -0.2, "FNO", ha="center", fontsize=7.5, color=INK, transform=b.get_xaxis_transform()); b.text(4.6, -0.2, "U-Net", ha="center", fontsize=7.5, color=INK, transform=b.get_xaxis_transform())
b.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=MUT), plt.Rectangle((0, 0), 1, 1, color=MUT, alpha=0.55, hatch="////", ec="white", lw=0)],
         labels=["$k_{\\mathrm{eff}}$", "late breakthrough $t_{95}$"], frameon=False, fontsize=6.5, loc="upper right")
b.set_ylabel("test error (%)"); b.set_title("(b) downstream error by training loss", fontsize=8, loc="left", color=INK)
for ax in (a, b):
    ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
fig.tight_layout(pad=0.3)
out = sys.argv[1] if len(sys.argv) > 1 else str(R / "writeup/fig_drop")
fig.savefig(out + ".pdf"); fig.savefig(out + ".png", dpi=200); print("wrote", out)
