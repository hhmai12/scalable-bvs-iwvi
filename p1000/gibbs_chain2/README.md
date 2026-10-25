# p = 1000 Gibbs convergence check (Appendix E, Table E.2)

The reference chain (`../p1000_Gibbs_reference.ipynb`, empty start, seed 12345) supplies every Gibbs PIP
used in the thesis. This folder runs one extra chain per dataset, used only for the convergence
diagnostics.
- Same sampler, prior and budget: 20,000 burn-in + 100,000 retained sweeps.
- Start: 20 predictors drawn at random (the prior expected model size), seed 880001 + dataset seed.
- Sampler seed: 12445 = 12345 + 100 x (chain - 1), the chain-seed rule of the reference notebook.

`bash run_p1000_chain2.sh` does everything:
1. `setup_gibbs.py` compiles `gibbs_p1000_core.pyx`, which is the `%%cython` cell of the reference
   notebook with folders taken from the environment.
2. `run_chain2.py` re-runs sweeps 1-40 of the saved reference chain for dataset 0 and stops unless
   they match exactly. It then runs chain 2 for all 50 datasets in parallel and writes the results to
   `output_gibbs_tau2_1_chain2_random20/`.
3. `chain2_diagnostics.py` computes:
   - rank-normalised split R-hat, bulk/tail ESS and MCSE for every varying inclusion indicator, and
     for the log-posterior and model-size traces;
   - between-chain PIP differences and RMSE;
   - the split-half check of the reference chain.

   It writes these to `diagnostics/`, including `p1000_two_chain_diagnostics.tex`. It uses
   ArviZ 0.22 when installed, otherwise the NumPy port of the same functions in `arviz022_port.py`.
