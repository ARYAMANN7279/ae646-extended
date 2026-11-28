"""External test on real geology: SPE10 model 2 permeability (Christie and Blunt 2001; spe.org, por_perm_case2a.zip).

The model is 60 x 220 x 85 cells; layers 1-35 are Tarbert (smooth, log-normal), 36-85 Upper Ness (fluvial channels).
From every layer we cut four 60 x 60 windows (y offsets 0, 53, 107, 160) and map them to 128 x 128 by nearest-cell
lookup, so each map is still piecewise constant on the original SPE10 cells. Models were trained on two-valued maps
(0.1 and 1), so two variants keep the input range:
  bin:  1 where k is above the map median, 0.1 elsewhere (real geometry, training contrast)
  cont: log k rescaled per map onto [log 0.1, log 1] (real geometry and continuous texture, training range)
Labels and downstream metrics are exactly those of src/drop.py (pressure-drop problem, converged at 2x).

  python src/spe10.py build [nproc]          -> data/spe10/{tarbert,upperness}_{bin,cont}.npz
  python src/spe10.py predict <run_dir> ...  -> <run_dir>/preds_spe10.npz (CPU inference from model.pt)
  python src/spe10.py score <out.json> <group> ...   groups are run-name prefixes; seeds _s0 _s1 _s2
"""
import json, os, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
NX, NY, NZ = 60, 220, 85
SETS = ("tarbert_bin", "tarbert_cont", "upperness_bin", "upperness_cont")


def read_kx():
    v = np.loadtxt(ROOT / "data/spe10_raw/spe_perm.dat").ravel()
    assert v.size == 3 * NX * NY * NZ, v.size
    return v[:NX * NY * NZ].reshape(NZ, NY, NX)                 # x fastest, then y, then z


def maps(kx):
    idx = np.floor(np.arange(128) * NX / 128).astype(int)
    out = {"tarbert": [], "upperness": []}
    for z in range(NZ):
        for y0 in (0, 53, 107, 160):
            w = np.maximum(kx[z, y0:y0 + NX, :], 1e-3)          # (60, 60); a few zero cells in Upper Ness
            out["tarbert" if z < 35 else "upperness"].append(w[np.ix_(idx, idx)])
    return {k: np.stack(v) for k, v in out.items()}


def variants(K):
    med = np.median(K.reshape(len(K), -1), axis=1)[:, None, None]
    kb = np.where(K > med, 1.0, 0.1)
    lk = np.log(K); lo = lk.min((1, 2), keepdims=True); hi = lk.max((1, 2), keepdims=True)
    kc = np.exp(np.log(0.1) * (1 - (lk - lo) / np.maximum(hi - lo, 1e-12)))
    return {"bin": kb.astype(np.float32), "cont": kc.astype(np.float32)}


def build(nproc):
    import drop
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    (ROOT / "data/spe10").mkdir(parents=True, exist_ok=True)
    for region, K in maps(read_kx()).items():
        for v, Kv in variants(K).items():
            out = ROOT / f"data/spe10/{region}_{v}.npz"
            if out.exists(): continue
            t0 = time.time()
            with Pool(nproc) as p:
                U = np.stack(p.map(drop.label, list(Kv), chunksize=4))
            np.savez_compressed(out, kappa128=Kv, u128=U)
            ko, ki = drop.keff(U, Kv)
            print(out.name, Kv.shape, f"{time.time()-t0:.0f}s keff median {np.median(ko):.4f} in/out {np.abs(ko/ki-1).max():.1e}", flush=True)


def predict(dirs):
    import torch, argparse
    import stage4 as S, stage3_train as T
    from stage4_models import get_stage4_model
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    data = None
    for d in map(Path, dirs):
        if (d / "preds_spe10.npz").exists() or not (d / "model.pt").exists(): continue
        ck = torch.load(d / "model.pt", map_location="cpu", weights_only=False); m = json.load(open(d / "metrics.json"))
        S.LOGK, S.EQUIV, S.DROP = m.get("log_kappa", False), m.get("scale_equiv", False), True
        a = argparse.Namespace(**ck["spec"])
        model = T.build_model(a, a.res) if ck["arch"] in ("fno", "mlp") else get_stage4_model(ck["arch"])
        model.load_state_dict(ck["state"]); model.eval()
        if data is None: data = T.Data(900, torch.device("cpu"))
        out = {}
        with torch.no_grad():
            for s in SETS:
                k = torch.tensor(np.load(ROOT / f"data/spe10/{s}.npz")["kappa128"])
                st = ck["stats"]
                out[s] = (T.predict(model, data, S.kin(k), a.res, st, bs=20)[..., 0] * st["u_std"] + st["u_mean"]).numpy().astype(np.float32)
        np.savez_compressed(d / "preds_spe10.npz", **out); print("pred", d.name, flush=True)


def score(outjson, groups):
    import drop
    runs = ROOT / "results/ext/runs"; res = json.load(open(outjson)) if Path(outjson).exists() else {}
    Z = {s: np.load(ROOT / f"data/spe10/{s}.npz") for s in SETS}
    for g in groups:
        dirs = [runs / f"{g}_s{i}" for i in range(3)]
        if not all((d / "preds_spe10.npz").exists() for d in dirs): print("skip", g); continue
        pp = [np.load(d / "preds_spe10.npz") for d in dirs]; res[g] = {}
        for s in SETS:
            K, G = Z[s]["kappa128"].astype(np.float64), Z[s]["u128"].astype(np.float64)
            res[g][s] = drop.score_set(np.stack([p[s] for p in pp]).astype(np.float64), K, G)
        print(f"{g:18s} " + " ".join(f"{s}: p {res[g][s]['p']*100:.2f} keff {res[g][s]['keff']*100:.1f} t95 {res[g][s]['t95']*100:.1f} |" for s in SETS), flush=True)
        json.dump(res, open(outjson, "w"), indent=1)


if __name__ == "__main__":
    c = sys.argv[1]
    if c == "build": build(int(sys.argv[2]) if len(sys.argv) > 2 else 8)
    elif c == "predict": predict(sys.argv[2:])
    elif c == "score": score(sys.argv[2], sys.argv[3:])
