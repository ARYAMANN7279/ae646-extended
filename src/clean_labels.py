"""Converged labels for the PDEBench 2D Darcy train/val/test maps (the fixed 900/100/200 split).
PDEBench labels differ by about 5% from the converged solution of -div(k grad u) = 1, u = 0 on the edge
(results/flux/pdebench_label_check_128.json). This re-solves every split map with stage4_ood.clean_solve
(strict cell-centred FV, 2x refined, cell-averaged to 128) and writes data/clean/{train,val,test}.npz."""
import os, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np
os.environ["DARCY_CLEAN"] = "1"; os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import stage4_ood as O
ROOT = Path(__file__).resolve().parents[1]; C = ROOT / "data" / "cache"; OUT = ROOT / "data" / "clean"

def one(i):
    return O.clean_solve(np.load(C / "kappa128.npy", mmap_mode="r")[i]).astype(np.float32)

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True); sp = np.load(C / "splits.npz"); K = np.load(C / "kappa128.npy", mmap_mode="r")
    U = np.load(C / "u128.npy", mmap_mode="r"); nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    for name in ("test", "val", "train"):
        idx = np.asarray(sp[name]); t0 = time.time()
        with Pool(nproc) as p:
            u = np.stack(p.map(one, idx.tolist(), chunksize=4))
        k = np.stack([K[i] for i in idx]).astype(np.float32); g = np.stack([U[i] for i in idx])
        d = np.linalg.norm((u - g).reshape(len(u), -1), axis=1) / np.linalg.norm(u.reshape(len(u), -1), axis=1)
        np.savez_compressed(OUT / f"{name}.npz", kappa128=k, u128=u, idx=idx, pdebench_vs_clean=d.astype(np.float32))
        print(name, len(idx), f"PDEBench vs clean rel-L2 mean {d.mean():.4f} median {np.median(d):.4f} max {d.max():.4f}", f"{time.time()-t0:.0f}s", flush=True)
