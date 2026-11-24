"""AE646-extended seed-2 pressure-drop runs (also builds data/drop*, src/drop.py) at 128x128 with converged labels. Rebuilds the PDEBench cache, checks the split,
builds converged labels (clean_labels.py), the clean shifted sets and clean wide-contrast data, then runs jobs on the T4s."""
import glob, hashlib, os, shutil, subprocess, sys, tarfile, threading, time
from pathlib import Path
import numpy as np
ROOT = Path("/kaggle/temp/AE646"); OUT = Path("/kaggle/working/ext"); (ROOT/"src").mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
t = glob.glob("/kaggle/input/**/ae646_s11_code.tar.gz", recursive=True)
if t:
    tarfile.open(t[0]).extractall(ROOT/"src")
else:
    d = Path(glob.glob("/kaggle/input/**/stage4.py", recursive=True)[0]).parent
    [shutil.copy(f, ROOT/"src"/f.name) for f in d.glob("*")]
shutil.move(str(ROOT/"src"/"s11_jobs.txt"), str(ROOT/"jobs.txt"))
LOG = open(OUT/"progress.txt", "a")
def say(*a): print(*a, flush=True); print(*a, file=LOG, flush=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "h5py", "tqdm", "requests", "pyamg"], check=False)
def run(cmd, env=None):
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env=env); say(" ".join(cmd[1:3]), r.returncode, r.stdout[-500:], r.stderr[-300:]); assert r.returncode == 0
for s in ("download_data.py", "preprocess.py", "data_cache.py"):
    run([sys.executable, f"src/{s}"])
sp = np.load(ROOT/"data/cache/splits.npz"); E = {"train": "f83c5c1e0da10feb", "val": "87208fa8e36a7f95", "test": "cc5fb1cf52a81817"}
assert all(hashlib.sha256(np.ascontiguousarray(sp[k]).tobytes()).hexdigest()[:16] == v for k, v in E.items()); say("split MATCH")
env = dict(os.environ, DARCY_CLEAN="1", OMP_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="")
run([sys.executable, "src/clean_labels.py", "4"], env)
p1 = subprocess.Popen([sys.executable, "src/stage4_ood.py", "--build"], cwd=ROOT, env=env); p2 = subprocess.Popen([sys.executable, "src/stage4_ood.py", "--build-mixed-wide"], cwd=ROOT, env=env)
p1.wait(); p2.wait(); assert p1.returncode == 0 and p2.returncode == 0
a = np.load(ROOT/"data/clean/train.npz"); b = np.load(ROOT/"data/mixed_wide_clean/train.npz"); (ROOT/"data/cleanmix").mkdir(exist_ok=True)
np.savez(ROOT/"data/cleanmix/train.npz", kappa128=np.concatenate([a["kappa128"], b["kappa128"]]), u128=np.concatenate([a["u128"], b["u128"]]))
run([sys.executable, "src/drop.py", "labels", "4"], env)
say("data hashes", {f: hashlib.sha256(np.load(f)["u128"].tobytes()).hexdigest()[:12] for f in sorted(glob.glob(str(ROOT/"data/*clean*/*.npz")))})
ngpu = max(1, len(subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.strip().splitlines()))
queue = [l.rstrip("\n").split("|", 1) for l in open(ROOT/"jobs.txt") if l.strip()]; lock = threading.Lock()
def lane(g):
    while True:
        with lock:
            if not queue: return
            name, args = queue.pop(0)
        t0 = time.time(); r = subprocess.run([sys.executable, "src/stage4.py", "--name", name] + args.split() + ["--out-root", str(OUT/"runs")],
                                            cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(g)), capture_output=True, text=True)
        tail = [l for l in r.stdout.splitlines() if l.startswith("[")]; say(f"gpu{g} {name} rc={r.returncode} {time.time()-t0:.0f}s", tail[-1] if tail else r.stderr[-400:])
ths = [threading.Thread(target=lane, args=(g,)) for g in range(ngpu)]; [x.start() for x in ths]; [x.join() for x in ths]; say("ALL-DONE")
