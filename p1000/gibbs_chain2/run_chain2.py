"""Second Gibbs chain for the p = 1000 simulation (convergence check only).

The reference chain (output_gibbs_tau2_1, chain 1) starts from the empty model with
method seed 12345. This script runs ONE extra chain per dataset with exactly the same
sampler, prior, burn-in (20,000) and retained sweeps (100,000), changing only
  * the start: 20 predictors drawn uniformly without replacement (the prior expected
    model size, p * a_pi / (a_pi + b_pi) = 19.6), seed 880001 + dataset seed;
  * the method seed: 12345 + 100 * (chain - 1) = 12445, the chain-seed rule already
    used in the reference notebook.
The reference results are not modified. Output: output_gibbs_tau2_1_chain2_random20/.

Before the main run the script re-runs the first 40 sweeps of the reference chain for
dataset 0 and checks they reproduce the saved trace exactly (same compiled sampler).
Finished datasets are skipped on restart.
"""
import argparse
import hashlib
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
P1000 = HERE.parent
REF_DIR = P1000 / "output_gibbs_tau2_1"
OUT_DIR = HERE / "output_gibbs_tau2_1_chain2_random20"
os.environ.setdefault("P1000_DATA_DIR", str(P1000 / "data"))
os.environ.setdefault("P1000_GIBBS_OUT", str(OUT_DIR))

CHAIN = 2
REF_METHOD_SEED = 12345
CHAIN_SEED = REF_METHOD_SEED + 100 * (CHAIN - 1)   # 12445
INIT_SIZE = 20
INIT_SEED_BASE = 880001
SEEDS = list(range(50))

_g = None


def core():
    global _g
    if _g is None:
        import gibbs_p1000_core as g
        _g = g
    return _g


def initial_gamma(seed, p):
    rng = np.random.default_rng(INIT_SEED_BASE + seed)
    idx = np.sort(rng.choice(p, size=INIT_SIZE, replace=False))
    gamma = np.zeros(p, dtype=np.uint8)
    gamma[idx] = 1
    return gamma


def atomic_csv(frame, path):
    tmp = path.with_suffix(".tmp")
    frame.to_csv(tmp, index=False, float_format="%.17g")
    tmp.replace(path)


def paths(seed):
    return (OUT_DIR / f"gibbs_draws_seed_{seed:02d}_chain_{CHAIN}.npz",
            OUT_DIR / f"runtime_gibbs_seed_{seed:02d}_chain_{CHAIN}.csv")


def sanity_check():
    """Re-run sweeps 1-40 of the saved reference chain (dataset 0) and compare."""
    g = core()
    meta = pd.read_csv(REF_DIR / "runtime_and_metrics_gibbs_seed_00.csv").iloc[0]
    assert int(meta.method_seed) == REF_METHOD_SEED and int(meta.n_chains) == 1
    assert int(meta.burn_in_per_chain) == g.GIBBS_BURN_IN == 20000
    assert int(meta.retained_sweeps_per_chain) == g.GIBBS_KEEP_SWEEPS == 100000
    with np.load(REF_DIR / "gibbs_draws_seed_00_chain_1.npz") as z:
        ref_lp, ref_m, ref_init = z["logpost"][:40], z["model_size"][:40], z["initial_gamma"]
    assert ref_init.sum() == 0, "reference chain was expected to start from the empty model"
    g.load_dataset(0)
    out = g.run_gibbs(n_iter=40, burn_in=0, thin=1, random_scan=False, verbose_every=None,
                      method_seed=REF_METHOD_SEED, gamma_init=np.zeros(g.p_py, dtype=np.uint8))
    if not (np.array_equal(out["model_size"], ref_m)
            and np.allclose(out["logpost"], ref_lp, rtol=0, atol=1e-8)):
        raise RuntimeError("Compiled sampler does NOT reproduce the saved reference chain "
                           f"(max |dlogpost| = {np.max(np.abs(out['logpost'] - ref_lp)):.3g}).")
    print("Sanity check passed: first 40 sweeps of the reference chain for dataset 0 "
          "reproduced exactly.", flush=True)


def run_one(seed):
    npz_path, meta_path = paths(seed)
    if npz_path.exists() and meta_path.exists():
        return seed, None
    g = core()
    data_file = g.load_dataset(seed)
    init = initial_gamma(seed, g.p_py)
    out = g.run_gibbs(n_iter=g.GIBBS_N_ITER, burn_in=g.GIBBS_BURN_IN, thin=g.GIBBS_THIN,
                      random_scan=g.RANDOM_SCAN, verbose_every=None,
                      method_seed=CHAIN_SEED, gamma_init=init)
    tmp = npz_path.with_name(npz_path.stem + ".part.npz")
    np.savez_compressed(tmp, samples=out["samples"], logpost=out["logpost"],
                        model_size=out["model_size"], flip_count=out["flip_count"],
                        initial_gamma=out["initial_gamma"], final_gamma=out["final_gamma"])
    tmp.replace(npz_path)
    atomic_csv(pd.DataFrame({"variable_name": g.feature_cols, "is_true_active": g.truth_np.astype(int),
                             f"pip_chain_{CHAIN}": out["pip"]}),
               OUT_DIR / f"pips_gibbs_seed_{seed:02d}_chain_{CHAIN}.csv")
    row = dict(seed=seed, chain=CHAIN, start=f"{INIT_SIZE} random predictors",
               init_seed=INIT_SEED_BASE + seed,
               init_variables=" ".join(f"X{j + 1}" for j in np.flatnonzero(init)),
               init_true_active=int((init.astype(bool) & (g.truth_np > 0.5)).sum()),
               method_seed=CHAIN_SEED, total_sweeps=g.GIBBS_N_ITER, burn_in=g.GIBBS_BURN_IN,
               retained_sweeps=len(out["samples"]), thin=g.GIBBS_THIN,
               tau2=g.tau2_py, a_pi=g.a_pi_py, b_pi=g.b_pi_py,
               data_sha256=hashlib.sha256(Path(data_file).read_bytes()).hexdigest())
    row["time (s)"] = float(out["time"])
    atomic_csv(pd.DataFrame([row]), meta_path)
    return seed, row["time (s)"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=int(os.environ.get("N_WORKERS", 6)))
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sanity_check()
    todo = [s for s in SEEDS if not all(x.exists() for x in paths(s))]
    print(f"{len(SEEDS) - len(todo)} datasets already done, {len(todo)} to run "
          f"with {args.workers} workers.", flush=True)
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers, maxtasksperchild=1) as pool:
        for k, (seed, secs) in enumerate(pool.imap_unordered(run_one, todo), 1):
            print(f"[{time.strftime('%H:%M:%S')}] dataset {seed:02d} done "
                  f"({secs:.0f} s) | {k}/{len(todo)} | elapsed {(time.time() - t0) / 60:.1f} min",
                  flush=True)
    rows = [pd.read_csv(paths(s)[1]) for s in SEEDS]
    atomic_csv(pd.concat(rows, ignore_index=True), OUT_DIR / f"runtime_all_seeds_chain_{CHAIN}.csv")
    print("Chain 2 complete for all 50 datasets.", flush=True)


if __name__ == "__main__":
    main()
