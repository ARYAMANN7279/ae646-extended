"""Split the PDEBench-label error into (a) boundary convention: steady solve with u=0 one full cell outside
the array (the ghost-cell wall; bc=1) vs the stated problem (u=0 on the domain edge; bc=0.5, clean labels), and
(b) non-convergence: PDEBench label vs the bc=1 steady solve. Relates both to the high-permeability area share."""
import json, os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, str(Path(__file__).resolve().parent)); import solvers as S
ROOT = Path(__file__).resolve().parents[1]
z = np.load(ROOT / "data/clean/test.npz"); U = np.load(ROOT / "data/cache/u128.npy", mmap_mode="r")
rel = lambda a, b: float(np.linalg.norm(a - b) / np.linalg.norm(b))
out = {k: [] for k in ("label_vs_clean", "label_vs_ghost", "ghost_vs_clean", "area_hi")}
for k, c, i in zip(z["kappa128"], z["u128"], z["idx"]):
    k = k.astype(np.float64); g = np.asarray(U[i], np.float64)
    gh = np.mean([S.solve(k, method="direct", face=f, bc=1.0) for f in ("harmonic", "arithmetic")], 0)
    out["label_vs_clean"].append(rel(g, c)); out["label_vs_ghost"].append(rel(g, gh)); out["ghost_vs_clean"].append(rel(gh, c)); out["area_hi"].append(float((k > 0.5).mean()))
a = {k: np.array(v) for k, v in out.items()}; hard = a["area_hi"] < 0.2
res = {k: {"mean": float(v.mean()), "median": float(np.median(v)), "max": float(v.max()), "hard_mean": float(v[hard].mean()), "rest_mean": float(v[~hard].mean())} for k, v in a.items() if k != "area_hi"}
res["n_hard"] = int(hard.sum()); res["spearman_area_vs_label_vs_ghost"] = float(spearmanr(a["area_hi"], a["label_vs_ghost"])[0])
res["spearman_area_vs_label_vs_clean"] = float(spearmanr(a["area_hi"], a["label_vs_clean"])[0])
(ROOT / "results/flux").mkdir(parents=True, exist_ok=True); json.dump({"summary": res, "per_map": {k: v.tolist() for k, v in a.items()}}, open(ROOT / "results/flux/label_decomposition.json", "w"), indent=1)
print(json.dumps(res, indent=1))
