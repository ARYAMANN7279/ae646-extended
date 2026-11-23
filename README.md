# AE646 extended: flux and label fidelity of Darcy neural operators

Research extension of the AE646 course project (team PINNacles). The course work is in `old/`
(stages 1 to 4, report, poster, earlier abstracts, results). The current work lives at the top level.

**Question.** Darcy neural operators (FNO, U-Net, DeepONet) are judged on pressure error only.
How accurate are the quantities that are actually used, namely the flux q = -k grad u and local mass balance,
and how good are the PDEBench Darcy labels themselves?

**Findings so far**
- Flux error is 3.5 to 10 times the pressure error (`src/flux_audit.py`, `results/flux/flux_audit_test.json`).
- PDEBench 2D Darcy labels differ by 4.9% from the converged solution of the stated problem at native 128x128,
  while our solver self-converges to 0.06% (`kaggle/kaggle_s10.py`, `results/flux/pdebench_label_check_128.json`).

**Code (`src/`)**
- `clean_labels.py`: converged labels for the PDEBench split maps -> `data/clean/`.
- `stage4_ood.py`: shifted test sets and randomised-contrast training data; `DARCY_CLEAN=1` uses converged labels.
- `stage4.py`: training runner (`--test-npz`, `--flux-loss`, `--scale-equiv`, `--log-kappa`, `--save-pred`).
- `flux_audit.py`: pressure, flux, interface-band and mass-balance metrics.
- `vm_lanes.py`: job runner for the shared lab GPU box (idle cards only, temperature watchdog).
- `solvers.py`, `stage3_train.py`, `models.py`, `stage4_models.py`, `preprocess.py`, `data_cache.py`, `download_data.py`: shared pieces.

`data/` is git-ignored and regenerable.
