# p = 1000 (Experiment 2)
1. `data/` – the 50 simulated datasets used in the thesis (included). `p1000_data_generation.ipynb` recreates them
   (random seeds 0–49; coefficients from seed 2026).
2. Five training notebooks (datasets 0–49).
3. `p1000_Gibbs_reference.ipynb` – one chain per dataset from the empty model: 20,000 burn-in sweeps and
   100,000 retained sweeps (finite-budget comparator).
4. `p1000_postprocess_tables_figures.ipynb` – selection, the three importance-sampling corrections, metrics
   against the Gibbs comparator and the truth (with ESS), thesis tables and figures.
5. `gibbs_chain2/run_p1000_chain2.sh` – convergence check for the Gibbs comparator (Appendix E, Table E.2).
   It runs a second chain from 20 random predictors and computes the two-chain R-hat, ESS, MCSE and
   between-chain PIP diagnostics, plus the split-half check of the reference chain. See `gibbs_chain2/README.md`.
