# Analysis log: downstream errors of Darcy neural operators

Status as of 28 Nov 2026. Everything here is measured in this repo; file paths point to the numbers.
All results are means over 3 seeds unless stated.

## 1. The research gap

Darcy neural operators (FNO, U-Net, DeepONet) are trained and ranked on PDEBench by the relative L2 error
of the pressure. Applications use the flux q = -k grad u: effective permeability (keff), travel and
breakthrough times, transport. No published work measures whether pressure accuracy carries over to these
quantities against converged references, or which training choices make it carry over.

Closest prior work and why it does not close the gap:

| Work | What it does | What it leaves open |
|---|---|---|
| FlowGuard (2026) | Re-solves PDEBench Darcy labels, finds them wrong | Scores pressure only |
| Boon et al. (2023, arXiv 2311.14554) | Flux and pressure errors diverge | Reduced-order models, not neural operators |
| Chakraborty and Chen (2026, arXiv 2610.00415) | Better field scores can mean worse physics | Airfoils, not porous media |
| Sobolev / H1 FNO (Zhang et al. 2026, arXiv 2610.05188) | Gradient-aware training | Not evaluated on flux or downstream quantities |
| CATO (arXiv 2605.09016), earlier conservation losses | Conservation-type losses | Not tested on keff or breakthrough times |
| Neural Green's Operators (2026) | Scaling normalisation for shifted permeability levels | Used here, not claimed |
| McGreivy and Hakim (NMI 2024) | Weak baselines and reporting bias in ML-for-PDE | General, motivates careful evaluation |

What we claim as new: the first measurement of how pressure error translates into flux, keff and
breakthrough-time error against converged references, and the first comparison of training losses on
those quantities. Not claimed: the label re-solve (FlowGuard), conservation losses (CATO and earlier),
the scaling normalisation (NGO).

Honest verdict: a solid, useful finding of the type "the field measures the wrong thing, here is how much
it matters, here is a cheap fix". Not a breakthrough method. Do not oversell it.

## 2. Setup

- Data: PDEBench 2D Darcy, beta = 1, 128x128, split 900 train / 100 val / 200 test.
- Converged labels (`src/clean_labels.py`, `data/clean/`): cell-centred finite volume, harmonic face
  permeability, solved at 256 and averaged to 128. Self-convergence 0.06% from 256 to 512.
- PDEBench labels deviate from the converged solution by 5.8% on average (4.9% at native 128;
  `results/flux/pdebench_label_check_128.json`). Cause: boundary convention plus non-convergence.
  This confirms FlowGuard. All models are trained and scored on converged labels.
- Pressure-drop problem (`src/drop.py`): same permeability maps, u = 1 left, u = 0 right, no flow top and
  bottom. Labels solved at 2x and averaged; 256 vs 512 within 0.02%; inlet vs outlet flux mismatch < 1e-4.
- Downstream metrics (`src/drop.py`):
  - keff = outlet flux.
  - Pollock semi-analytical particle tracking, flux-weighted inlet release, 2000 particles,
    arrival-time quantiles t5, t50, t95; error = |log ratio|.
  - Interface band: cells within two of a permeability jump, and its share of the flux error.
  - Metric checks (`results/drop/metric_checks.json`, 60 maps): 400 vs 2000 particles changes log t95
    by 1%; solving at 128 vs the label changes keff by 0.12%. So metric noise is far below the effects.
- Shifted test sets (`data/drop_ood/`): independent generator ("ID", our own Gaussian-random-field
  generator), high and low contrast, mostly low, rough, smooth, three phase, scaled up and scaled down.
- Training (`src/stage4.py`): 30k steps, cosine schedule, symmetry augmentation (y flip; x flip with
  u -> 1 - u for the drop problem), FNO 8 layers (9.5M params), U-Net (7.8M), DeepONet.
- Losses: relative L2; H1 (unweighted finite-difference gradient, `--h1-loss w`); permeability-weighted
  face-flux loss (`--flux-loss w`); PINO-style finite-volume residual (`--pde-loss w`).

## 3. Results

### 3.1 Source problem (PDEBench forcing, converged labels; `results/ext/analysis_[fh]_*.json`)

| Model | Pressure | Flux | Flux error in interface band |
|---|---|---|---|
| FNO, L2 | 0.33% | 8.9% | 43% of flux error in 4.8% of area |
| U-Net, L2 | 0.23% | 5.4% | 65% |
| FNO, flux loss | 0.24% | 2.0% | |
| U-Net, flux loss | 0.22% | 0.9% | |
| FNO, H1 | 0.21% | 2.8% | |
| U-Net, H1 | 0.20% | 1.4% | |

Flux error is 4 to 27 times the pressure error, concentrated at permeability jumps.
The earlier "FNO beats U-Net on pressure but loses on flux" claim for this problem was dropped:
long-trained U-Net beats FNO on both metrics here.

### 3.2 Pressure-drop problem, test set (`results/drop/*.json`)

| Model and loss | Pressure | Flux | keff | t50 | t95 | Generator keff |
|---|---|---|---|---|---|---|
| FNO, L2 | 0.26% | 15.4% | 9.0% | 4.2% | 10.3% | 894% |
| U-Net, L2 | 0.17% | 10.8% | 13.7% | 4.0% | 12.4% | 102% |
| DeepONet, L2 | 1.51% | 65.7% | 78.5% | 30.6% | 27.9% | 252% |
| FNO, H1 w=0.3 | 0.17% | 4.2% | 3.3% | 1.2% | 3.0% | 130% |
| FNO, H1 w=1 | 0.16% | 3.4% | 3.5% | 1.1% | 2.9% | 59% |
| FNO, H1 w=3 | 0.17% | 3.4% | 4.5% | 1.0% | 2.8% | 53% |
| U-Net, H1 w=0.3 | 0.13% | 2.8% | 6.4% | 1.5% | 4.3% | 66% |
| U-Net, H1 w=1 | 0.13% | 2.4% | 6.2% | 1.4% | 3.1% | 53% |
| U-Net, H1 w=3 | 0.14% | 2.4% | 8.0% | 1.1% | 2.7% | 68% |
| FNO, PDE residual w=0.1 | 0.23% | 4.2% | 1.6% | 1.0% | 3.6% | 67% |
| FNO, PDE residual w=1 | 0.32% | 4.2% | 1.7% | 1.2% | 3.8% | 24% |
| U-Net, PDE residual w=0.1 | 0.18% | 2.8% | 2.0% | 0.8% | 3.1% | 18% |
| U-Net, PDE residual w=1 | 0.24% | 2.7% | 1.4% | 0.9% | 3.0% | 13% |
| FNO, face flux w=1 | 0.21% | 2.9% | 1.1% | 0.6% | 2.4% | 23% |
| U-Net, face flux w=1 | 0.18% | 1.7% | 1.3% | 0.5% | 1.7% | 11% |

Trained on the mixed-map set (`data/cleanmix`, includes generator-like maps; log-permeability input,
scale normalisation):

| Model and loss | Pressure | keff | t95 | Generator keff |
|---|---|---|---|---|
| FNO-mix, L2 | 0.30% | 6.9% | 13.9% | 20% |
| FNO-mix, flux w=0.3 | 1.03% | 6.1% | 11.6% | 8% |
| FNO-mix, flux w=1 | 2.81% | 6.4% | 30.0% | 7% |

### 3.3 What the numbers say

1. Pressure error is a poor proxy for the quantities that matter. With L2, pressure errors of 0.17 to
   0.26% come with 9 to 14% keff error and 10 to 12% late-breakthrough error.
2. Ranking flips. FNO-L2 has worse pressure than U-Net-L2 (0.26 vs 0.17%) but better keff (9.0 vs 13.7%).
   This holds on every seed. A model picked by pressure can be the worse one for keff.
3. H1 does not fix it. It gives the best pressure (0.13 to 0.17%) but keff stays at 3.3 to 8.0% for
   weights 0.3 to 3. Tuning the weight does not close the gap; higher weight makes U-Net keff worse.
4. Losses that enforce permeability-weighted, conservative fluxes do fix it. PDE residual: keff 1.4 to
   2.0%. Face flux: keff 1.1 to 1.3%, t95 1.7 to 2.4%, with pressure unchanged. The face-flux loss is
   the best on every downstream metric, about 30% lower keff and 40% lower t95 than the residual loss,
   and the residual loss at w=1 worsens pressure (0.24 to 0.32%).
   Mechanism: H1 matches gradients without weighting them by permeability, so errors in high-k
   channels (which carry the flow) count the same as errors in low-k zones. Both successful losses
   weight by permeability.
5. Generalisation. On independent-generator maps, flux-conserving losses give 11 to 24% keff error
   (best weights), L2 and H1 give 52 to 894%. Training on mixed maps brings generator keff to 7 to 20%
   but costs in-distribution accuracy (keff 6 to 7%), and the flux loss at w=1 there hurts pressure.
6. Scaled permeability sets (scaled up / down) break every model without input normalisation (keff
   errors 100% to 30000%), although the drop solution is invariant to scaling k. Scale normalisation
   (FNO-mix) fixes this (7 to 20%). Expected and credited to NGO; not a headline.
7. DeepONet is far behind (1.5% pressure, 78% keff); trained with L2 only.
8. Flux error concentrates at permeability interfaces (band share 30 to 70% of flux error in under 5%
   of the area).

## 4. Deliverables

- Abstract: `abstract/srivastava_aryamann_main.tex` / `.pdf` (1 page, ERCOFTAC ML4Fluids, Florence,
  April 2027, due 4 Dec 2026). Figure `abstract/fig_drop.pdf` from `writeup/make_fig_drop.py`.
- One-pager for Prof. Sathesh: `writeup/onepager.tex` / `.pdf`.
- References: `abstract/refs.bib` (all entries verified).

The abstract and one-pager currently say "53/52 to 894% for the other losses" on generator maps. With the
PDE-residual baseline this is no longer true (13 to 67%). Pending rewording (not yet applied, awaiting
OK):

> An H1 loss gives the best pressure but still 3.3 to 8.0% keff error for weights 0.3 to 3. Losses that
> enforce conservative fluxes remove most of the gap: a PDE-residual loss gives 1.4 to 2.0% keff error,
> and a permeability-weighted face-flux loss gives 1.3% or less, with 2.4% late-breakthrough error at
> unchanged pressure error. On maps from an independent generator, flux-conserving losses give 11 to 24%
> keff error, against 52 to 894% for L2 and H1.

## 5. Known weaknesses (what a reviewer can say)

1. The independent generator is our own GRF generator, not an outside dataset.
2. One PDE (2D steady Darcy), one resolution (128), one data source (PDEBench beta = 1).
3. DeepONet only with L2.
4. Literature search careful but not exhaustive; re-check arXiv just before submission.
5. The residual loss is evaluated on the 128 grid, where the converged labels themselves have 1.4%
   residual; a residual on a finer grid might do better.
6. Hyperparameters (steps, learning rate) were not tuned per loss.

## 6. What to work on next

Before 4 Dec (abstract):
- [ ] Approve and apply the rewording in section 4; rebuild both PDFs on one page; push.
- [ ] Decide whether Prof. Sathesh is a co-author.
- [ ] Team review of the abstract.
- [ ] Submit through the ERCOFTAC portal from your own account.

For the full paper (after the abstract):
- [ ] Real permeability: SPE10 layers (and/or a second public dataset) as a truly external test set.
- [ ] Other benchmarks: same pressure-vs-flux audit on another elliptic or flow problem.
- [ ] DeepONet with H1, residual and flux losses; a transformer operator (e.g. Transolver) for breadth.
- [ ] Combine residual and face-flux losses; residual on a finer grid.
- [ ] Resolution transfer (train 64, test 128/256) for each loss.
- [ ] A ranking study: Kendall tau between pressure ranking and keff/t95 ranking over many trained models.
- [ ] Release converged labels, drop-problem data and the scoring script as a benchmark.

## 7. Reproduce

- Labels: `python src/drop.py labels 4` -> `data/drop/`, `data/drop_ood/`.
- Train: job lists `jobs_d.txt`, `jobs_h1w.txt`, `jobs_pde.txt`; run with
  `python src/vm_lanes.py <jobs> 1,2` on the lab VM (idle cards only, gentle thermal throttle).
- Score: `python src/drop.py score results/drop/<group>.json results/ext/runs/<group>_s{0,1,2}`.
- Figure: `python writeup/make_fig_drop.py`.
- Kaggle replicas: `kaggle/kaggle_s12.py`, `kaggle_s13.py`, `kaggle_s14.py`.
