#!/bin/bash
# Body Fat prior sensitivity (Appendix F). The main analysis (m0 = 3) is run first with the notebooks
# as they are. This script re-runs the same notebooks with m0 = 1.5 and m0 = 6 (environment variable
# BODYFAT_M0), each evaluated against the exact posterior under its own prior, then builds Table F.1.
set -eu
cd "$(dirname "$0")"
mkdir -p logs
for M0 in 1.5 6; do
  export BODYFAT_M0=$M0
  for m in ELBO VIMCO_star VIMCO_star_annealed VIMCO_GM VIMCO_GM_annealed; do
    jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 \
      --output-dir logs --output "bodyfat_${m}_training_m0_${M0}.ipynb" "bodyfat_${m}_training.ipynb" &
  done
  wait
  jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 \
    --output-dir logs --output "bodyfat_postprocess_m0_${M0}.ipynb" bodyfat_postprocess_tables_figures.ipynb
done
unset BODYFAT_M0
python prior_sensitivity_table.py | tee logs/prior_sensitivity_table.log
