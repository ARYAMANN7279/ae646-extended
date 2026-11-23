"""One-pager figure: (a) PDEBench label error per test map vs high-permeability share; (b) pressure vs flux error."""
import json
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = Path(__file__).resolve().parents[1]; B, O, INK, MUT = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.family": "serif", "font.serif": ["TeX Gyre Termes", "Times New Roman", "DejaVu Serif"], "font.size": 8,
                     "axes.edgecolor": MUT, "axes.labelcolor": INK, "xtick.color": MUT, "ytick.color": MUT, "axes.linewidth": 0.6})
fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.1), gridspec_kw={"width_ratios": [1, 1.35]})
d = json.load(open(R / "results/flux/label_decomposition.json"))["per_map"]
x = np.array(d["area_hi"]); a.scatter(x, 100 * np.array(d["label_vs_clean"]), s=9, c=B, lw=0.4, edgecolor="white", label="vs stated problem", zorder=3)
a.scatter(x, 100 * np.array(d["label_vs_ghost"]), s=9, c=O, lw=0.4, edgecolor="white", label="vs its own steady state", zorder=3)
a.set_yscale("log"); a.set_xlabel("high-permeability area share"); a.set_ylabel("PDEBench label error (%)")
a.axvspan(0, 0.2, color="#e8e7e3", zorder=0); a.text(0.01, 60, "hard maps", color=MUT, fontsize=7)
a.legend(frameon=False, fontsize=7, loc="upper right"); a.set_title("(a) label error per test map", fontsize=8, loc="left", color=INK)
G = [("FNO\nPDEBench lab.", "e1_fno_pde"), ("DeepONet", "e1_deeponet_clean"), ("FNO", "e1_fno_clean"), ("U-Net", "e1_unet_clean"),
     ("FNO\n+flux loss", "e2_fno_clean_flux1"), ("U-Net\n+flux loss", "e2_unet_clean_flux1")]
p = [100 * json.load(open(R / f"results/ext/analysis_{g}.json"))[g]["test"]["p"] for _, g in G]
q = [100 * json.load(open(R / f"results/ext/analysis_{g}.json"))[g]["test"]["q"] for _, g in G]
i = np.arange(len(G)); w = 0.36
b.bar(i - w / 2 - 0.01, p, w, color=B, label="pressure", zorder=3); b.bar(i + w / 2 + 0.01, q, w, color=O, label="flux", zorder=3)
for xi, v in zip(i, q): b.text(xi + w / 2, v * 1.12, f"{v:.0f}" if v >= 10 else f"{v:.1f}", ha="center", fontsize=6.5, color=INK)
for xi, v in zip(i, p): b.text(xi - w / 2, v * 1.12, f"{v:.1f}" if v >= 1 else f"{v:.2f}", ha="center", fontsize=6.5, color=INK)
b.set_yscale("log"); b.set_ylim(0.4, 200); b.set_xticks(i); b.set_xticklabels([g for g, _ in G], fontsize=6.5); b.set_ylabel("test error vs converged labels (%)")
b.legend(frameon=False, fontsize=7, loc="upper right", ncol=2); b.set_title("(b) pressure vs flux error, 128x128, 3 seeds", fontsize=8, loc="left", color=INK)
for ax in (a, b):
    ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", color="#e8e7e3", lw=0.5, zorder=0)
fig.tight_layout(pad=0.3); fig.savefig(R / "writeup/fig_onepager.pdf"); fig.savefig(R / "writeup/fig_onepager.png", dpi=200)
