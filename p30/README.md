# p = 30 (Experiment 1)
1. `p30_{ELBO,VIMCO_star,VIMCO_star_annealed,VIMCO_GM,VIMCO_GM_annealed}_training.ipynb` – variational fits for
   datasets 0–49 (`data_new/`), including the (S, N) allocation grid for the VIMCO methods.
2. `p30_Gibbs_reference.ipynb` – collapsed Gibbs reference, four chains per dataset.
3. `p30_postprocess_tables_figures.ipynb` – selection, PSIS / IWMM / Iterative Damped PSIS, metrics with ESS,
   Gibbs diagnostics (Table E.1), and the thesis tables and figures. These include Table B.1, in which the
   evaluated-ELBO / MCSE rule selects one fit per dataset, method and allocation (S, N), and the selected fits
   are compared across allocations by repeated-measures ANOVA (Greenhouse–Geisser, Holm). They also include
   the varbvs row of Table 4.3.

`varbvs/run_varbvs_mcmc_benchmark_p30.R` (R, varbvs 2.6.10): fits varbvs to datasets 0–49 at the 31 prior
log-odds settings −3, −2.9, …, 0, then runs the varbvs-matched collapsed Gibbs reference (σ, sa and log-odds
fixed at the varbvs values; 4 chains per setting, 2,000 burn-in and 20,000 retained sweeps), combined with the
varbvs weights. Run it with `Rscript p30/varbvs/run_varbvs_mcmc_benchmark_p30.R`; it writes
`varbvs/varbvs_vs_gibbs_pips_by_variable.csv`, from which the post-processing notebook computes the varbvs PIP
RMSE and excess marginal FKL (Table 4.3). Intermediate fits and diagnostics go to `varbvs/output_p30/`.
