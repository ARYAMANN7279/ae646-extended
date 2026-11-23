"""Job runner for the shared lab VM. Runs jobs from a file ("name|args" per line) on the given cards only,
one job per card, and only while the card has no other user's process. A watchdog kills (and requeues) our job
if the card reaches KILL_C or another user's process appears on it. CUDA_VISIBLE_DEVICES is set inside a
--norc shell so ~/.bashrc cannot override it, and placement is verified from nvidia-smi."""
import os, subprocess, sys, threading, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; PY = sys.executable
KILL_C, START_C = 89, 78
cards = [int(c) for c in sys.argv[2].split(",")]; jobs_file = Path(sys.argv[1]); log = open(ROOT / "logs" / "lanes.log", "a")
def say(*a):
    print(time.strftime("%m-%d %H:%M:%S"), *a, file=log, flush=True)
def smi(q, kind="gpu"):
    return subprocess.run(["nvidia-smi", f"--query-{kind}={q}", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip().splitlines()
def uuid_of():
    return {l.split(", ")[1]: int(l.split(", ")[0]) for l in smi("index,uuid")}
def card_pids(c):
    u = uuid_of(); return {int(l.split(", ")[0]) for l in smi("pid,gpu_uuid", "compute-apps") if l and u.get(l.split(", ")[1]) == c}
def temp(c):
    return int(smi("temperature.gpu")[c])
lock = threading.Lock(); queue = [l.rstrip("\n").split("|", 1) for l in open(jobs_file) if l.strip() and not l.startswith("#")]
def descendants(pid):
    out = subprocess.run(["pgrep", "-g", str(pid)], capture_output=True, text=True).stdout.split()
    return {int(x) for x in out}
def lane(c):
    while True:
        with lock:
            if not queue: return
            name, args = queue.pop(0)
        if (ROOT / "results/ext/runs" / name / "metrics.json").exists(): continue
        while card_pids(c) or temp(c) >= START_C:
            time.sleep(20)
        cmd = f"export CUDA_VISIBLE_DEVICES={c} OMP_NUM_THREADS=2; exec {PY} src/stage4.py --name {name} {args} --throttle-gpu {c} --out-root results/ext/runs"
        p = subprocess.Popen(["env", "-u", "SSH_CLIENT", "-u", "SSH_CONNECTION", "bash", "--norc", "--noprofile", "-c", cmd], cwd=ROOT,
                             stdout=open(ROOT / "logs" / f"{name}.log", "w"), stderr=subprocess.STDOUT, start_new_session=True)
        say(f"card{c} start {name} pid {p.pid}"); t0 = time.time(); why = None; checked = False
        while p.poll() is None:
            time.sleep(5)
            ours = descendants(p.pid) | {p.pid}; on = card_pids(c)
            if not checked and time.time() - t0 > 40:
                checked = True
                if not (on & ours):
                    allc = set().union(*[card_pids(x) for x in range(4)])
                    if allc & ours: why = "WRONG CARD"
            if on - ours: why = "other user on card"
            if temp(c) >= KILL_C: why = f"temp {temp(c)}C"
            if why:
                os.killpg(p.pid, 9); p.wait(); break
        if why:
            say(f"card{c} KILLED {name}: {why}; requeue"); time.sleep(60)
            with lock: queue.append([name, args])
            if why == "WRONG CARD": return
        else:
            tail = [l for l in open(ROOT / "logs" / f"{name}.log").read().splitlines() if l.startswith("[")]
            say(f"card{c} done {name} rc={p.returncode} {time.time()-t0:.0f}s", tail[-1] if tail else "")
ths = [threading.Thread(target=lane, args=(c,)) for c in cards]; [t.start() for t in ths]; [t.join() for t in ths]; say("ALL-DONE", jobs_file.name)
