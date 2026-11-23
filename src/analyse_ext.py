"""Analysis of the results/ext runs (all at 128x128, scored against converged labels).
Per group (name without _sN): pressure error vs clean and vs PDEBench labels, flux error, interface-band flux error,
mass-balance residual, on the test split and every shifted set; ensemble-spread vs FV-residual error indicators
(Spearman with the ensemble-mean error); and AMG repair curves from the seed-0 prediction (first 50 test maps).
Writes results/ext/analysis.json and prints a table."""
import glob, json, os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, str(Path(__file__).resolve().parent))
import solvers as S, flux_audit as F
ROOT = Path(__file__).resolve().parents[1]; R = ROOT / "results/ext"

def qerr(u, g, k):
    a, b = F.faces(u, k), F.faces(g, k)
    return float(np.sqrt(sum(((x - y) ** 2).sum() for x, y in zip(a[:2] + a[2], b[:2] + b[2])) / sum((y ** 2).sum() for y in b[:2] + b[2])))

def band_q(u, g, k):
    a, b = F.faces(u, k), F.faces(g, k); B = F.band(k); bx = B[:, 1:] | B[:, :-1]; by = B[1:] | B[:-1]
    num = ((a[0] - b[0]) ** 2 * bx).sum() + ((a[1] - b[1]) ** 2 * by).sum(); den = (b[0] ** 2 * bx).sum() + (b[1] ** 2 * by).sum()
    tot = ((a[0] - b[0]) ** 2).sum() + ((a[1] - b[1]) ** 2).sum()
    return float(np.sqrt(num / max(den, 1e-30))), float(num / max(tot, 1e-30)), float(B.mean())

def fvres(u, k):
    A, b = S.assemble(k, 1.0, "harmonic", 0.5); return float(np.linalg.norm(b - A @ u.ravel()) / np.linalg.norm(b))

def score(preds, K, G):
    """preds: (seeds, n, 128, 128)."""
    m = preds.mean(0); out = defaultdict(list)
    for i in range(len(G)):
        k, g = K[i].astype(np.float64), G[i].astype(np.float64)
        for s in range(len(preds)):
            u = preds[s, i].astype(np.float64)
            out["p"].append(np.linalg.norm(u - g) / np.linalg.norm(g)); out["q"].append(qerr(u, g, k))
            bq, sh, ar = band_q(u, g, k); out["q_band"].append(bq); out["band_share"].append(sh); out["band_area"].append(ar)
            out["res"].append(fvres(u, k))
        e = np.linalg.norm(m[i] - g) / np.linalg.norm(g); out["ens_p"].append(e); out["ens_q"].append(qerr(m[i].astype(np.float64), g, k))
        out["spread"].append(float(np.linalg.norm(preds[:, i].std(0)) / np.linalg.norm(m[i]))); out["res_ens"].append(fvres(m[i].astype(np.float64), k))
    r = {k: float(np.mean(v)) for k, v in out.items() if k not in ("spread", "res_ens")}
    if len(preds) > 1:
        r["rho_spread"] = float(spearmanr(out["spread"], out["ens_p"])[0]); r["rho_res"] = float(spearmanr(out["res_ens"], out["ens_p"])[0])
    r["n_seeds"] = len(preds)
    return r

if __name__ == "__main__":
    groups = defaultdict(list)
    for d in sorted(glob.glob(str(R / "runs/*/metrics.json"))):
        groups[Path(d).parent.name.rsplit("_s", 1)[0]].append(Path(d).parent)
    zt = np.load(ROOT / "data/clean/test.npz"); Kt, Gt = zt["kappa128"], zt["u128"]
    Upde = np.load(ROOT / "data/cache/u128.npy", mmap_mode="r"); Gpde = np.stack([Upde[i] for i in zt["idx"]])
    sets = {Path(f).stem: np.load(f) for f in sorted(glob.glob(str(ROOT / "data/ood_clean/*.npz")))}
    only = sys.argv[1:]; res = json.load(open(R / "analysis.json")) if (R / "analysis.json").exists() else {}
    for g, dirs in sorted(groups.items()):
        if only and g not in only: continue
        P = np.stack([np.load(d / "preds.npz")["pred"] for d in dirs]).astype(np.float32)
        if P.shape[-1] != 128: continue
        r = {"test": score(P, Kt, Gt)}
        r["test"]["p_vs_pdebench"] = float(np.mean([np.linalg.norm(P[s, i] - Gpde[i]) / np.linalg.norm(Gpde[i]) for s in range(len(P)) for i in range(len(Gpde))]))
        oo = [np.load(d / "preds_ood.npz") for d in dirs if (d / "preds_ood.npz").exists()]
        if oo:
            r["ood"] = {s: score(np.stack([o[s] for o in oo]), z["kappa128"], z["u128"]) for s, z in sets.items() if all(s in o for o in oo)}
        res[g] = r; t = r["test"]
        print(f"{g:28s} p {t['p']:.4f} (vs PDEBench {t['p_vs_pdebench']:.4f}) q {t['q']:.4f} band q {t['q_band']:.3f} share {100*t['band_share']:.0f}%/{100*t['band_area']:.0f}% res {t['res']:.2e}"
              + (f" rho spread {t['rho_spread']:.2f} res {t['rho_res']:.2f}" if "rho_spread" in t else ""), flush=True)
        for s, o in r.get("ood", {}).items():
            print(f"    {s:12s} p {o['p']:.3f} q {o['q']:.3f} res {o['res']:.2e}" + (f" rho spread {o['rho_spread']:.2f} res {o['rho_res']:.2f}" if "rho_spread" in o else ""))
        json.dump(res, open(R / "analysis.json", "w"), indent=1)
