"""Pressure-drop Darcy problem and its downstream quantities.

Problem: -div(k grad u) = 0 on the unit square, u = 1 on the left edge, u = 0 on the right edge, no flow through
top and bottom. Same permeability maps as the PDEBench split and the OOD sets, so models are compared on exactly
the maps used elsewhere. Labels: strict cell-centred FV with harmonic faces, solved on a 2x refined grid and
cell-averaged to 128 (as for the source-driven clean labels).

Downstream quantities computed from a pressure field through two-point face fluxes (so a surrogate is judged by
what an engineer would compute from it):
  k_eff     outlet flux (= effective permeability, since the drop and the length are 1); also the inlet flux
  t5/t50/t95 flux-weighted breakthrough times from semi-analytical (Pollock) particle tracking, particles
            released at the inlet in proportion to the inlet flux, porosity 1

Usage:
  python src/drop.py labels [nproc]            -> data/drop/{train,val,test,mixtrain}.npz, data/drop_ood/*.npz
  python src/drop.py check                     -> self-convergence of the labels (128 / 256 / 512)
  python src/drop.py score <run_dir> [...]     -> downstream errors of runs' preds.npz / preds_ood.npz vs labels
"""
import json, os, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl

ROOT = Path(__file__).resolve().parents[1]


def solve(k):
    """Cell-centred FV solve of the drop problem on the grid of k (N x N, rows = y, cols = x)."""
    k = np.asarray(k, np.float64); n = k.shape[0]; idx = np.arange(n * n).reshape(n, n)
    tx = 2 * k[:, 1:] * k[:, :-1] / (k[:, 1:] + k[:, :-1]); ty = 2 * k[1:] * k[:-1] / (k[1:] + k[:-1])
    r, c, v = [], [], []
    for a, b, t in ((idx[:, :-1], idx[:, 1:], tx), (idx[:-1], idx[1:], ty)):
        a, b, t = a.ravel(), b.ravel(), t.ravel()
        r += [a, b, a, b]; c += [a, b, b, a]; v += [t, t, -t, -t]
    d = np.zeros((n, n)); d[:, 0] += 2 * k[:, 0]; d[:, -1] += 2 * k[:, -1]
    r.append(idx.ravel()); c.append(idx.ravel()); v.append(d.ravel())
    A = sp.csr_matrix((np.concatenate(v), (np.concatenate(r), np.concatenate(c))), shape=(n * n, n * n))
    rhs = np.zeros((n, n)); rhs[:, 0] = 2 * k[:, 0]
    return spl.spsolve(A.tocsc(), rhs.ravel()).reshape(n, n)


def label(k):
    """Converged label at 128: solve on the 2x refined map, cell-average back."""
    n = k.shape[0]
    return solve(np.kron(np.asarray(k, np.float64), np.ones((2, 2)))).reshape(n, 2, n, 2).mean((1, 3)).astype(np.float32)


def faces(u, k):
    """Two-point face fluxes (positive in +x / +y), shapes (..., N, N+1) and (..., N+1, N); no flow top/bottom.
    Units: flux per face = Darcy velocity * face length; grid spacing cancels."""
    u = np.asarray(u, np.float64); k = np.asarray(k, np.float64)
    hx = 2 * k[..., :, 1:] * k[..., :, :-1] / (k[..., :, 1:] + k[..., :, :-1])
    hy = 2 * k[..., 1:, :] * k[..., :-1, :] / (k[..., 1:, :] + k[..., :-1, :])
    qx = np.concatenate([2 * k[..., :, :1] * (1 - u[..., :, :1]), hx * (u[..., :, :-1] - u[..., :, 1:]),
                         2 * k[..., :, -1:] * u[..., :, -1:]], -1)
    z = np.zeros(u.shape[:-2] + (1, u.shape[-1]))
    qy = np.concatenate([z, hy * (u[..., :-1, :] - u[..., 1:, :]), z], -2)
    return qx, qy


def keff(u, k):
    qx, _ = faces(u, k)
    return qx[..., :, -1].sum(-1), qx[..., :, 0].sum(-1)        # outlet, inlet


def track(u, k, P=400, max_steps=40000):
    """Pollock tracking for a batch of maps (M, N, N). Returns arrival times (M, P) (inf = lost: stuck or back out
    through the inlet). Time unit: homogeneous k = 1 gives t = 1."""
    qx, qy = faces(u, k); M, n = u.shape[0], u.shape[-1]
    vx, vy = qx * n * n, qy * n * n                               # cells per unit time (velocity / h, velocity = q / h)
    qin = np.clip(qx[:, :, 0], 0, None); cum = np.cumsum(qin, 1); cum /= cum[:, -1:]
    s = (np.arange(P) + 0.5) / P
    row = np.stack([np.searchsorted(cum[m], s) for m in range(M)]).clip(0, n - 1)
    lo = np.where(row > 0, np.take_along_axis(cum, np.maximum(row - 1, 0), 1), 0.0)
    hi = np.take_along_axis(cum, row, 1)
    fy = ((s[None] - lo) / np.maximum(hi - lo, 1e-30)).clip(0, 1)
    m = np.repeat(np.arange(M), P); ci = row.ravel(); cj = np.zeros_like(ci); fx = np.zeros(M * P); fy = fy.ravel()
    t = np.zeros(M * P); out = np.full(M * P, np.inf); act = np.arange(M * P)

    def tex(f, a, b):                                             # time to leave [0,1] with v(f) = a + (b - a) f
        A = b - a; vp = a + A * f; tiny = np.abs(A) < 1e-12
        with np.errstate(divide="ignore", invalid="ignore"):
            te = np.where(tiny, (1 - f) / vp, np.log(b / vp) / A); tw = np.where(tiny, -f / vp, np.log(a / vp) / A)
        te = np.where((vp > 0) & (b > 0), te, np.inf); tw = np.where((vp < 0) & (a < 0), tw, np.inf)
        return np.minimum(te, tw), te <= tw, vp, A

    def adv(f, vp, A, a, dt):
        with np.errstate(over="ignore", invalid="ignore"):
            g = np.where(np.abs(A) < 1e-12, f + vp * dt, (vp * np.exp(A * dt) - a) / np.where(np.abs(A) < 1e-12, 1, A))
        return np.clip(g, 0, 1)

    for _ in range(max_steps):
        if not len(act): break
        mm, i, j = m[act], ci[act], cj[act]
        tx_, east, vpx, Ax = tex(fx[act], vx[mm, i, j], vx[mm, i, j + 1])
        ty_, north, vpy, Ay = tex(fy[act], vy[mm, i, j], vy[mm, i + 1, j])
        dt = np.minimum(tx_, ty_); stuck = ~np.isfinite(dt)
        dtf = np.where(stuck, 0, dt)
        nfx = adv(fx[act], vpx, Ax, vx[mm, i, j], dtf); nfy = adv(fy[act], vpy, Ay, vy[mm, i, j], dtf)
        xs = tx_ <= ty_
        nfx = np.where(xs, np.where(east, 1.0, 0.0), nfx); nfy = np.where(~xs, np.where(north, 1.0, 0.0), nfy)
        t[act] += dtf
        nj = j + np.where(xs, np.where(east, 1, -1), 0); ni = i + np.where(~xs, np.where(north, 1, -1), 0)
        nfx = np.where(xs, np.where(east, 0.0, 1.0), nfx); nfy = np.where(~xs, np.where(north, 0.0, 1.0), nfy)
        arrived = (nj >= n) & ~stuck; lost = stuck | (nj < 0) | (ni < 0) | (ni >= n)
        out[act[arrived]] = t[act[arrived]]
        keep = ~(arrived | lost)
        ci[act] = np.clip(ni, 0, n - 1); cj[act] = np.clip(nj, 0, n - 1); fx[act] = nfx; fy[act] = nfy
        act = act[keep]
    return out.reshape(M, P)


def downstream(u, k, P=400):
    ko, ki = keff(u, k); tt = track(u, k, P)
    q = np.quantile(tt, [0.05, 0.5, 0.95], axis=1, method="inverted_cdf").T
    return {"keff": ko, "keff_in": ki, "t": q, "lost": np.isinf(tt).mean(1)}


# ── labels ───────────────────────────────────────────────────────────────────
def _lab(k):
    return label(k)


def make_labels(nproc):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    jobs = {f"drop/{s}": ROOT / f"data/clean/{s}.npz" for s in ("test", "val", "train")}
    jobs["drop/mixtrain"] = ROOT / "data/cleanmix/train.npz"
    for f in sorted((ROOT / "data/ood_clean").glob("*.npz")):
        jobs[f"drop_ood/{f.stem}"] = f
    for name, src in jobs.items():
        out = ROOT / "data" / f"{name}.npz"
        if out.exists(): continue
        out.parent.mkdir(parents=True, exist_ok=True); K = np.load(src)["kappa128"].astype(np.float32); t0 = time.time()
        with Pool(nproc) as p:
            U = np.stack(p.map(_lab, list(K), chunksize=4))
        np.savez_compressed(out, kappa128=K, u128=U)
        ko, ki = keff(U, K)
        print(name, len(K), f"{time.time()-t0:.0f}s keff median {np.median(ko):.4f} in/out mismatch max {np.abs(ko/ki-1).max():.1e}", flush=True)


def check():
    K = np.load(ROOT / "data/clean/test.npz")["kappa128"][:8].astype(np.float64)
    r = {}
    for i, k in enumerate(K):
        u1 = solve(k); u2 = label(k)
        u4 = solve(np.kron(k, np.ones((4, 4)))).reshape(128, 4, 128, 4).mean((1, 3))
        e = lambda a, b: float(np.linalg.norm(a - b) / np.linalg.norm(b))
        k1, k2, k4 = keff(u1, k)[0], keff(solve(np.kron(k, np.ones((2, 2)))), np.kron(k, np.ones((2, 2))))[0], \
            keff(solve(np.kron(k, np.ones((4, 4)))), np.kron(k, np.ones((4, 4))))[0]
        r[i] = {"p128_vs_512": e(u1, u4), "p256_vs_512": e(u2, u4), "keff128": float(k1), "keff256": float(k2), "keff512": float(k4)}
        print(i, r[i], flush=True)
    t = downstream(np.stack([solve(k) for k in K[:4]]), K[:4]); print("track test", t, flush=True)
    homog = downstream(np.stack([solve(np.ones((128, 128)))]), np.ones((1, 128, 128)))
    print("homogeneous (expect keff 1, t 1):", homog, flush=True)


# ── scoring ──────────────────────────────────────────────────────────────────
def rel(a, b):
    return np.abs(a - b) / np.abs(b)


def score_set(P, K, G, P_ref=400):
    """P (S, M, N, N) predictions of S seeds; returns mean errors over seeds and maps."""
    ref = downstream(G, K, P_ref); qx_t, qy_t = faces(G, K); out = []
    for p in P:
        d = downstream(p, K, P_ref); qx, qy = faces(p, K)
        num = np.sqrt(((qx - qx_t) ** 2).sum((1, 2)) + ((qy - qy_t) ** 2).sum((1, 2)))
        den = np.sqrt((qx_t ** 2).sum((1, 2)) + (qy_t ** 2).sum((1, 2)))
        lt = np.abs(np.log(d["t"] / ref["t"]))
        out.append({"p": np.linalg.norm((p - G).reshape(len(G), -1), axis=1) / np.linalg.norm(G.reshape(len(G), -1), axis=1),
                    "q": num / den, "keff": rel(d["keff"], ref["keff"]), "keff_inout": np.abs(d["keff"] / d["keff_in"] - 1),
                    "t5": lt[:, 0], "t50": lt[:, 1], "t95": lt[:, 2], "lost": d["lost"]})
    agg = {key: float(np.mean([np.nanmean(np.where(np.isfinite(o[key]), o[key], np.nan)) for o in out])) for key in out[0]}
    agg["t95_inf_share"] = float(np.mean([np.mean(~np.isfinite(o["t95"])) for o in out]))
    agg["per_seed_p"] = [float(o["p"].mean()) for o in out]; agg["per_seed_q"] = [float(o["q"].mean()) for o in out]
    agg["per_seed_keff"] = [float(o["keff"].mean()) for o in out]; agg["per_seed_t5"] = [float(np.nanmean(np.where(np.isfinite(o["t5"]), o["t5"], np.nan))) for o in out]
    return agg


def score(dirs, outjson):
    z = np.load(ROOT / "data/drop/test.npz"); K, G = z["kappa128"].astype(np.float64), z["u128"].astype(np.float64)
    P = np.stack([np.load(Path(d) / "preds.npz")["pred"] for d in dirs]).astype(np.float64)
    res = {"runs": [Path(d).name for d in dirs], "test": score_set(P, K, G)}
    oo = [np.load(Path(d) / "preds_ood.npz") for d in dirs if (Path(d) / "preds_ood.npz").exists()]
    if len(oo) == len(dirs):
        res["ood"] = {}
        for f in sorted((ROOT / "data/drop_ood").glob("*.npz")):
            if all(f.stem in o for o in oo):
                zz = np.load(f); res["ood"][f.stem] = score_set(np.stack([o[f.stem] for o in oo]).astype(np.float64),
                                                                 zz["kappa128"].astype(np.float64), zz["u128"].astype(np.float64))
    json.dump(res, open(outjson, "w"), indent=1)
    t = res["test"]
    print(f"{Path(outjson).stem:28s} p {t['p']:.4f} q {t['q']:.4f} keff {t['keff']:.4f} t5 {t['t5']:.3f} t50 {t['t50']:.3f} t95 {t['t95']:.3f} lost {t['lost']:.3f}", flush=True)
    for s, o in res.get("ood", {}).items():
        print(f"    {s:12s} p {o['p']:.3f} q {o['q']:.3f} keff {o['keff']:.3f} t5 {o['t5']:.3f} t50 {o['t50']:.3f}", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "labels":
        make_labels(int(sys.argv[2]) if len(sys.argv) > 2 else 8)
    elif cmd == "check":
        check()
    elif cmd == "score":
        out = Path(sys.argv[2]); score(sys.argv[3:], out)
