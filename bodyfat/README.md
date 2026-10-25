# Body Fat (Section 5.1)
`bodyfat.csv`: 251 men after removing observation 42; response percentage body fat, 13 predictors.
1. `bodyfat_exact_enumeration.ipynb` – exact posterior over all 2^13 models (m0 = 3, Beta(0.3, 1), tau^2 = 1).
2. Five training notebooks – 50 optimisation seeds (12345–12394) per method.
3. `bodyfat_postprocess_tables_figures.ipynb` – selection, the three corrections scored against the exact posterior
   (PIP RMSE, excess marginal FKL, k-hat, ESS, exact coverage and expected interval score), thesis tables and Figure 5.1.
4. `run_prior_sensitivity.sh` – Appendix F. It re-runs the five training notebooks and the post-processing
   notebook with m0 = 1.5 and m0 = 6. The prior is set by the environment variable `BODYFAT_M0`; the default
   is 3, the main analysis. Each fit is scored against the exact posterior under its own prior, and
   `prior_sensitivity_table.py` then builds Table F.1 and the supporting checks in
   `prior_sensitivity_results/`.
