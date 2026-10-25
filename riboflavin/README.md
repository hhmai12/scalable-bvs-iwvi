# Riboflavin (Section 5.2)
`riboflavin_raw_full.csv`: n = 71, p = 4088 (from the R package hdi), response log riboflavin production rate `y`.
1. `riboflavin_make_splits.ipynb` – 50 random 50/21 train–test splits (seeds 1000–1049); predictors standardised
   with training means and SDs.
2. Five training notebooks – one fit per configuration and split, prior expected model size 6.
3. `riboflavin_postprocess_tables_figures.ipynb` – selection, the three corrections, held-out RMSE, MLPD, coverage
   and interval score (with k-hat and ESS), thesis tables and Figure 5.2.
