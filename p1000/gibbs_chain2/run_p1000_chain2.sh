#!/bin/bash
# p = 1000 Gibbs convergence check (Appendix E, Table E.2): second chain + two-chain diagnostics.
# Run after ../p1000_Gibbs_reference.ipynb (which writes ../output_gibbs_tau2_1).
# Usage (from anywhere):  bash p1000/gibbs_chain2/run_p1000_chain2.sh      [N_WORKERS=4 to use fewer cores]
# Each worker needs about 1.3 GB of RAM; about 2-3 hours with 6 workers. Finished datasets are skipped on re-run.
set -eu
cd "$(dirname "$0")"
N_WORKERS="${N_WORKERS:-6}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
mkdir -p logs
python setup_gibbs.py > logs/build.log 2>&1
python -u run_chain2.py --workers "$N_WORKERS" | tee logs/chain2.log
python -u chain2_diagnostics.py --workers "$N_WORKERS" | tee logs/diagnostics.log
echo "Done: results in $(pwd)/diagnostics"
