# Scalable Bayesian variable selection via importance-weighted variational inference over discrete spaces

Code for the Honours thesis (Mai Mai, The University of Sydney, 2026).

Five self-contained folders, one per experiment. Every notebook is run from its own folder
(Jupyter "Run All"); paths are relative to that folder.

| Folder | Thesis section | Run in this order |
|---|---|---|
| `p30/` | Section 4.3; Tables 4.3, B.1, E.1; Figures 4.1–4.3 | (`p30_data_generation`, optional: data included), 5 training notebooks, `p30_Gibbs_reference`, `p30_postprocess_tables_figures` |
| `p1000/` | Section 4.4; Tables 4.4, E.2; Figures 4.4–4.5 | (`p1000_data_generation`, optional: data included), 5 training notebooks, `p1000_Gibbs_reference`, `p1000_postprocess_tables_figures`, then `gibbs_chain2/run_p1000_chain2.sh` |
| `p5000/` | Section 4.4; Table 4.5; Figure 4.6 | `p5000_data_generation`, 5 training notebooks, `p5000_postprocess_tables_figures` |
| `bodyfat/` | Section 5.1; Table 5.1, Figure 5.1; Appendix F (Table F.1) | `bodyfat_exact_enumeration`, 5 training notebooks, `bodyfat_postprocess_tables_figures`, then `run_prior_sensitivity.sh` |
| `riboflavin/` | Section 5.2; Table 5.2, Figure 5.2 | `riboflavin_make_splits`, 5 training notebooks, `riboflavin_postprocess_tables_figures` |

The learning-rate, runtime, selected/terminal alpha, iteration and configuration-sensitivity tables
(Tables C.1, 5.3, 5.4, 5.5 and G.1) are produced for each setting by the final section of its
`*_postprocess_tables_figures` notebook.

## Methods
Each training notebook fits one method over the full tuning grid
(learning rates 0.01, 0.05, 0.1, 0.2, 0.3; fixed-alpha variants also alpha in {0.2, 0.5, 0.7, 0.9}):
`ELBO`, `VIMCO_star` (VIMCO-★), `VIMCO_star_annealed`, `VIMCO_GM`, `VIMCO_GM_annealed`.
The five training notebooks of a setting are independent and can run in parallel.
At p = 30 the VIMCO notebooks also fit the (S, N) allocation grid. Table B.1 applies the same
selection rule within each allocation and compares the evaluated ELBO of the selected fits (Appendix B).

The post-processing notebook selects one configuration per method and replicate (evaluated
ELBO, two-MCSE rule), applies Ordinary PSIS, IWMM and Iterative Damped PSIS with K = 5000,
computes every metric (including the Pareto k-hat and the importance-sampling ESS) and then
builds the thesis tables (`.csv` and `.tex`) and figures in
`<results folder>/THESIS_TABLES_AND_FIGURES`.

## Software
Python 3.13 with torch 2.9.1, numpy 2.4, pandas 2.2, scipy 1.17, matplotlib, Cython, Jupyter and
**ArviZ 1.3.0** (PSIS and the Pareto k-hat; ArviZ 1.x returns an undefined k-hat when the largest
importance weights come from repeated models, as discussed in the thesis).
The varbvs comparison at p = 30 uses R with varbvs 2.6.10 (`p30/varbvs/`).
All computations are single-process. In Table 5.3, the variational runtimes are for the selected fit only, and the Gibbs
runtimes are those of the reference runs (`p30_Gibbs_reference`, `p1000_Gibbs_reference`).
The p = 1000 two-chain diagnostics (`p1000/gibbs_chain2/`) use ArviZ 0.22 if installed, otherwise an
equivalent NumPy port.

## Data
`p30/data_new/`, `p1000/data/`, `bodyfat/bodyfat.csv` and `riboflavin/riboflavin_raw_full.csv` are included.
The simulated datasets for p = 5000 (`p5000/data_5000/`, 2.4 GB) are too large for this repository; run
`p5000/p5000_data_generation.ipynb` first to create them from fixed seeds. `p30/p30_data_generation.ipynb` and
`p1000/p1000_data_generation.ipynb` recreate the included p = 30 and p = 1000 datasets in the same way. Files regenerated with a different NumPy build can differ from the thesis files
in the last floating-point digit. Each simulated dataset is a CSV with an `Intercept`
column, predictors `X1..Xp`, response `y` and the simulated noise; `fixed_beta_and_active_set.csv` holds the
generating coefficients (AR(1) design with correlation 0.8, noise variance 9, intercept 3).
