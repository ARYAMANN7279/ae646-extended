"""Does pressure error rank models the same way as downstream errors? Kendall tau over all trained
pressure-drop models (every group x seed in results/drop/d_*.json, PDEBench-trained only), test split."""
import json, glob, itertools, sys
from pathlib import Path
import numpy as np
from scipy.stats import kendalltau, spearmanr
R = Path(__file__).resolve().parents[1]
rows = []
for f in sorted(glob.glob(str(R / "results/drop/d_*.json"))):
    g = Path(f).stem
    if "mix" in g: continue
    t = json.load(open(f))["test"]
    for i, (p, k) in enumerate(zip(t["per_seed_p"], t["per_seed_keff"])):
        rows.append((g, i, p, k, t["per_seed_q"][i]))
def study(rows, tag):
    p = np.array([r[2] for r in rows]); k = np.array([r[3] for r in rows]); q = np.array([r[4] for r in rows])
    out = {"n_models": len(rows), "tau_p_keff": kendalltau(p, k)[0], "tau_p_q": kendalltau(p, q)[0], "tau_q_keff": kendalltau(q, k)[0],
           "rho_p_keff": spearmanr(p, k)[0]}
    pairs = list(itertools.combinations(range(len(rows)), 2))
    flips = sum((p[i] - p[j]) * (k[i] - k[j]) < 0 for i, j in pairs)
    out["pair_flip_share_p_vs_keff"] = flips / len(pairs)
    best_p = int(np.argmin(p)); out["best_by_pressure"] = rows[best_p][0]; out["its_keff_rank"] = int((k < k[best_p]).sum()) + 1
    out["best_by_keff"] = rows[int(np.argmin(k))][0]
    print(tag, json.dumps(out, default=float)); return out
res = {"all": study(rows, "all"), "no_deeponet": study([r for r in rows if "deeponet" not in r[0]], "no_deeponet")}
json.dump(res, open(R / "results/drop/rank_study.json", "w"), indent=1, default=float)
