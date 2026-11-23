"""Compact two-panel figure for the one-page ERCOFTAC abstract (data: results/stage3 and results/stage3_extra).
Run from the repo root: python3 ercoftac_abstract/make_fig.py"""
import json, re, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from stage3_aggregate import load
rows = load()
G, O, GR = "#1E8E5A", "#C96F1E", "#5F6B7A"
g = defaultdict(list)
for n, r in rows.items():
    m = re.match(r"e1_(\w+?)_(leg|cos)_(mse|rel_l2)_(none|d4)_s\d", n)
    if m: g[m.groups()].append(r)
order = [("leg", "mse", "none", "Stage-2\nrecipe"), ("cos", "mse", "none", "+cosine"), ("leg", "rel_l2", "none", "+rel-L2\nloss"),
         ("leg", "mse", "d4", "+D4\naug."), ("cos", "rel_l2", "d4", "all\nthree")]
fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.1), gridspec_kw={"width_ratios": [1.25, 1]})
ax = axs[0]; x = np.arange(len(order)); w = 0.38
for j, (model, c, lab) in enumerate((("fno", G, "FNO"), ("mlp", O, "MLP"))):
    mm = [np.mean([r["test"] for r in g[(model,) + o[:3]]]) for o in order]
    sd = [np.std([r["test"] for r in g[(model,) + o[:3]]], ddof=1) for o in order]
    ax.bar(x + (j - .5) * w, mm, w, yerr=sd, color=c, label=lab, capsize=1.5, error_kw={"lw": .8})
    for xi, v, e in zip(x + (j - .5) * w, mm, sd): ax.text(xi, v + e + .002, f"{v:.3f}", ha="center", fontsize=5.8, rotation=90, va="bottom")
ax.set_xticks(x); ax.set_xticklabels([o[3] for o in order], fontsize=6.5); ax.set_ylim(0, .125)
ax.set_ylabel("test rel. $L_2$ error", fontsize=7.5); ax.tick_params(labelsize=7); ax.legend(fontsize=7, frameon=False, ncol=2, loc="upper right")
ax.set_title("(a) training recipe, same model and budget", fontsize=7.5)
S = json.load(open(ROOT / "results/stage3_extra/summary.json"))["noise"]
ax = axs[1]; xs = [0, 1, 5, 10]; keys = ["0", "0.01", "0.05", "0.1"]
for m, c, lab in (("fno", G, "FNO"), ("mlp", O, "MLP")):
    ax.errorbar(xs, [S[m][k]["mean"] for k in keys], yerr=[S[m][k]["sd"] for k in keys], marker="o", ms=3.5, lw=1.3, capsize=1.5, color=c, label=lab)
ax.plot(xs, [v / 100 for v in xs], "--", color=GR, lw=1, label="noisy labels' own error")
ax.set_xticks(xs); ax.set_xlabel("label noise (% of map size)", fontsize=7.5); ax.set_ylabel("test error, clean maps", fontsize=7.5)
ax.tick_params(labelsize=7); ax.legend(fontsize=6.3, frameon=False, loc="upper left"); ax.set_title("(b) training on noisy labels", fontsize=7.5)
for a in axs:
    a.grid(alpha=.25)
    for s in ("top", "right"): a.spines[s].set_visible(False)
fig.tight_layout(pad=0.4); fig.savefig(Path(__file__).parent / "fig_abstract.pdf")
print("ok")
