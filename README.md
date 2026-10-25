# Bayesian variable selection via collapsed importance-weighted variational inference over discrete model spaces

Code for the paper by Mai Mai, Minh-Ngoc Tran and Matias Quiroz.

Five self-contained folders, one per setting. Every notebook is run from its own folder
(Jupyter "Run All"); paths are relative to that folder.

| Folder | Paper section |
|---|---|
| `p30/` | Experiment 1 (p = 30) and Experiment 3 (comparison with `varbvs`) |
| `p1000/` | Experiment 1 (p = 1000) |
| `p5000/` | Experiment 2 (p = 5000) |
| `bodyfat/` | Application: body fat data |
| `riboflavin/` | Application: riboflavin data |

The final section of each `*_postprocess_tables_figures` notebook also produces the learning-rate, runtime,
selected/terminal alpha, iteration and configuration-sensitivity tables for that setting.

## Methods
Each training notebook fits one method over the full tuning grid
(learning rates 0.01, 0.05, 0.1, 0.2, 0.3; fixed-alpha variants also alpha in {0.2, 0.5, 0.7, 0.9}):
`ELBO`, `VIMCO_star` (VIMCO-★), `VIMCO_star_annealed`, `VIMCO_GM`, `VIMCO_GM_annealed`.
The five training notebooks of a setting are independent and can run in parallel.
At p = 30 the VIMCO notebooks also fit an (S, N) allocation grid; the same selection rule is applied within
each allocation and the evaluated ELBO of the selected fits is compared across allocations.

The post-processing notebook selects one configuration per method and replicate (evaluated
ELBO, two-MCSE rule), applies Ordinary PSIS, IWMM and Iterative Damped PSIS with K = 5000,
computes every metric (including the Pareto k-hat and the importance-sampling ESS) and then
builds the tables (`.csv` and `.tex`) and figures in
`<results folder>/THESIS_TABLES_AND_FIGURES`.

## Software
Python 3.13 with torch 2.9.1, numpy 2.4, pandas 2.2, scipy 1.17, matplotlib, Cython, Jupyter and
**ArviZ 1.3.0** (PSIS and the Pareto k-hat; ArviZ 1.x returns an undefined k-hat when the largest
importance weights come from repeated models).
The varbvs comparison at p = 30 uses R with varbvs 2.6.10 (`p30/varbvs/`).
All computations are single-process. Reported variational runtimes are for the selected fit only, and the Gibbs
runtimes are those of the reference runs (`p30_Gibbs_reference`, `p1000_Gibbs_reference`).
The p = 1000 two-chain diagnostics (`p1000/gibbs_chain2/`) use ArviZ 0.22 if installed, otherwise an
equivalent NumPy port.

## Data
`p30/data_new/`, `p1000/data/`, `bodyfat/bodyfat.csv` and `riboflavin/riboflavin_raw_full.csv` are included.
The simulated datasets for p = 5000 (`p5000/data_5000/`, 2.4 GB) are too large for this repository; run
`p5000/p5000_data_generation.ipynb` first to create them from fixed seeds. `p30/p30_data_generation.ipynb` and
`p1000/p1000_data_generation.ipynb` recreate the included p = 30 and p = 1000 datasets in the same way. Files regenerated with a different NumPy build can differ from the included files
in the last floating-point digit. Each simulated dataset is a CSV with an `Intercept`
column, predictors `X1..Xp`, response `y` and the simulated noise; `fixed_beta_and_active_set.csv` holds the
generating coefficients (AR(1) design with correlation 0.8, noise variance 9, intercept 3).
