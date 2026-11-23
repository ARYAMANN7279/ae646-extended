"""Two-panel figure for abstract v2 (data: results/stage4/summary.json). Run: python3 ercoftac_abstract/make_fig_v2.py"""
import json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parent.parent
S = json.load(open(ROOT / "results/stage4/summary.json"))
G, B, O, GR = "#1E8E5A", "#2F7FD1", "#C96F1E", "#5F6B7A"
fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.15), gridspec_kw={"width_ratios": [1, 1.15]})
ax = axs[0]; x = np.arange(3); w = 0.27
for i, (c, lab, col) in enumerate((("s4_fno_ref", "FNO", G), ("s4_fno_multires", "FNO, multi-res", B), ("s4_unet", "U-Net", O))):
    m = [S[c]["at_res"][r]["mean"] for r in ("32", "64", "128")]
    ax.bar(x + (i - 1) * w, m, w, color=col, label=lab)
ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(["32$^2$", "64$^2$ (train)", "128$^2$"], fontsize=7)
ax.set_ylabel("test rel. $L_2$ error", fontsize=7.5); ax.tick_params(labelsize=7); ax.legend(fontsize=6.5, frameon=False, loc="upper center", ncol=3)
ax.set_ylim(0.015, 1.2); ax.set_title("(a) error on other grids", fontsize=7.5)
ax = axs[1]
sets = [("test", "PDEBench"), ("contrast_lo", "contrast\n0.3"), ("contrast_hi", "contrast\n0.01"), ("three_phase", "three\nphases"), ("mostly_low", "mostly\nlow")]
x = np.arange(len(sets)); w = 0.38
for i, (c, lab, col) in enumerate((("s5_fno_pde", "trained on PDEBench", GR), ("s7_fno_pdemixwide_logk", "+ random contrast, log $\\kappa$", G))):
    m = [S[c]["test"]["mean"] if k == "test" else S[c]["ood"][k]["mean"] for k, _ in sets]
    ax.bar(x + (i - .5) * w, m, w, color=col, label=lab)
ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels([l for _, l in sets], fontsize=6.5); ax.tick_params(labelsize=7)
ax.set_ylim(0.004, 8); ax.legend(fontsize=6.3, frameon=False, loc="upper left"); ax.set_title("(b) new permeability values (FNO)", fontsize=7.5)
for a in axs:
    a.grid(alpha=.25, axis="y")
    for s in ("top", "right"): a.spines[s].set_visible(False)
fig.tight_layout(pad=0.4); fig.savefig(Path(__file__).parent / "fig_abstract_v2.pdf"); print("ok")
