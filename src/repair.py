"""Cheap physics repair of surrogate predictions: k algebraic-multigrid V-cycles on the FV system at 64x64,
started from the surrogate's prediction, versus the same cycles started from zero and a direct solve.
Scores pressure, flux and mass balance against the converged labels (data/clean/test.npz, read at 64x64)
and times every step per map on one CPU thread. Usage: repair.py <preds.npz | stage3> [name] [n_maps]"""
import json, os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
from pathlib import Path
import numpy as np, pyamg, scipy.sparse.linalg as spla
sys.path.insert(0, str(Path(__file__).resolve().parent))
import solvers as S, flux_audit as F
ROOT = Path(__file__).resolve().parents[1]
CYC = [0, 1, 2, 3, 4, 6, 8, 12]

def qrel(u, g, k):
    a, b = F.faces(u, k), F.faces(g, k)
    num = sum(((x - y) ** 2).sum() for x, y in zip(a[:2] + a[2], b[:2] + b[2])); den = sum((y ** 2).sum() for y in b[:2] + b[2])
    return float(np.sqrt(num / den))

if __name__ == "__main__":
    src = sys.argv[1]; name = sys.argv[2] if len(sys.argv) > 2 else Path(src).parent.name; n = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    P = (np.load(ROOT / "old/results/stage3/diagnostics_arrays.npz")["pred"] if src == "stage3" else np.load(src)["pred"]).astype(np.float64)
    R = P.shape[-1]; s = 128 // R                     # score at the prediction's own grid (128 = native, consistent with the labels)
    z = np.load(ROOT / "data/clean/test.npz"); K = z["kappa128"][:, ::s, ::s].astype(np.float64); G = z["u128"][:, ::s, ::s].astype(np.float64)
    rows = {f"{init}_{c}": {"p": [], "q": [], "res": []} for init in ("pred", "zero") for c in CYC}; t = {"setup": [], "vcycle": [], "direct": []}
    for i in range(n):
        k = K[i]; A, b = S.assemble(k, 1.0, "harmonic", 0.5)
        t0 = time.perf_counter(); ml = pyamg.smoothed_aggregation_solver(A); t["setup"].append(time.perf_counter() - t0)
        t0 = time.perf_counter(); spla.spsolve(A.tocsc(), b); t["direct"].append(time.perf_counter() - t0)
        for init, x0 in (("pred", P[i].ravel()), ("zero", np.zeros(b.size))):
            for c in CYC:
                if c == 0:
                    x = x0.copy()
                else:
                    t0 = time.perf_counter(); x = ml.solve(b, x0=x0.copy(), maxiter=c, tol=1e-300, cycle="V")
                    if init == "pred" and c == 1: t["vcycle"].append(time.perf_counter() - t0)
                u = x.reshape(R, R); r = rows[f"{init}_{c}"]
                r["p"].append(np.linalg.norm(u - G[i]) / np.linalg.norm(G[i])); r["q"].append(qrel(u, G[i], k))
                r["res"].append(np.linalg.norm(b - A @ x) / np.linalg.norm(b))
    out = {"name": name, "n": n, "cycles": CYC, "time_ms": {k: 1e3 * float(np.median(v)) for k, v in t.items()},
           "curves": {k: {m: float(np.mean(v)) for m, v in r.items()} for k, r in rows.items()}}
    (ROOT / "results/repair").mkdir(parents=True, exist_ok=True); json.dump(out, open(ROOT / f"results/repair/{name}.json", "w"), indent=1)
    print(name, "times ms", {k: round(v, 2) for k, v in out["time_ms"].items()})
    for c in CYC:
        a, z0 = out["curves"][f"pred_{c}"], out["curves"][f"zero_{c}"]
        print(f"  cycles {c:2d} | from pred p {a['p']:.4f} q {a['q']:.4f} res {a['res']:.1e} | from zero p {z0['p']:.4f} q {z0['q']:.4f} res {z0['res']:.1e}")
