# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# Collapsed Gibbs sampler for p = 1000, taken verbatim from p1000_Gibbs_reference.ipynb
# (the %%cython cell). Only changes: data/output folders come from environment variables,
# and the final run_training() call is removed (run_chain2.py drives the sampler).
                          

import time
from pathlib import Path
import numpy as np
import pandas as pd
from libc.math cimport log, exp, sqrt, lgamma

start_all = time.time()

DATA_SEEDS = range(50)
METHOD_SEED = 12345

tau2_py = 1.0
prior_prec_py = 1.0 / tau2_py
jitter_py = 1e-8

a_pi_py = 0.02
b_pi_py = 1.0

GIBBS_N_CHAINS = 1
GIBBS_BURN_IN = 20000
GIBBS_KEEP_SWEEPS = 100000
GIBBS_THIN = 1
GIBBS_N_ITER = GIBBS_BURN_IN + GIBBS_KEEP_SWEEPS * GIBBS_THIN
RANDOM_SCAN = False
VERBOSE_EVERY = 5000

import os
data_dir = Path(os.environ["P1000_DATA_DIR"])
output_dir = Path(os.environ["P1000_GIBBS_OUT"])
output_dir.mkdir(parents=True, exist_ok=True)


MAX_CACHE_SIZE = 200000

true_model_file = data_dir / "fixed_beta_and_active_set.csv"

if not true_model_file.exists():
    raise FileNotFoundError(
        f"Cannot find true-model file: {true_model_file.resolve()}"
    )

true_model_df = pd.read_csv(true_model_file)

required_true_cols = {
    "variable_name",
    "is_intercept",
    "is_true_active",
    "beta",
}

missing_true_cols = required_true_cols.difference(true_model_df.columns)

if missing_true_cols:
    raise ValueError(
        "The true-model file is missing columns: "
        f"{sorted(missing_true_cols)}"
    )

if (len(true_model_df) != 1001
        or true_model_df['variable_name'].duplicated().any()
        or not true_model_df['is_intercept'].isin([0, 1]).all()):
    raise ValueError('The truth file must have 1001 distinct variables and binary is_intercept flags.')

slope_model_df = (
    true_model_df.loc[
        true_model_df["is_intercept"].astype(int) == 0
    ]
    .copy()
    .reset_index(drop=True)
)

p_py = int(len(slope_model_df))
d_py = p_py + 1

feature_cols = [f"X{j}" for j in range(1, p_py + 1)]

if p_py != 1000 or slope_model_df['variable_name'].duplicated().any():
    raise ValueError('This notebook requires exactly 1000 distinct slope variables.')
if set(slope_model_df['variable_name']) != set(feature_cols):
    raise ValueError('Slope variables must be named X1 through X1000.')
slope_model_df = slope_model_df.set_index('variable_name').loc[feature_cols].reset_index()
if true_model_df['is_intercept'].astype(int).eq(1).sum() != 1:
    raise ValueError('The truth file must contain exactly one intercept.')
if not np.isfinite(true_model_df['beta'].to_numpy(dtype=np.float64)).all():
    raise ValueError('True coefficients must be finite.')
if (not slope_model_df['is_true_active'].isin([0, 1]).all()
        or not np.array_equal(slope_model_df['is_true_active'].to_numpy(dtype=int),
                              (slope_model_df['beta'].to_numpy(dtype=float) != 0).astype(int))):
    raise ValueError('True inclusion indicators must match the nonzero slope coefficients.')

truth_np = slope_model_df["is_true_active"].to_numpy(dtype=np.float64)
beta_slopes_np = slope_model_df["beta"].to_numpy(dtype=np.float64)

true_active_np = np.where(truth_np > 0.5)[0].astype(np.int64)
true_idx = true_active_np + 1
m_true_py = int(len(true_active_np))

print("p =", p_py)
print("m_true =", m_true_py)
print("True active variables, 1-based:")
print(true_idx)

cdef int p = <int>p_py
cdef int d = <int>d_py
cdef int n = 0
cdef int m_true = <int>m_true_py

cdef double tau2 = tau2_py
cdef double prior_prec = 1.0 / tau2
cdef double jitter = jitter_py

cdef double a_pi = a_pi_py
cdef double b_pi = b_pi_py

cdef double LOG_BETA_AB_C = (
    lgamma(a_pi)
    + lgamma(b_pi)
    - lgamma(a_pi + b_pi)
)

cdef double nu_c = 0.0
cdef double log_2pi_c = log(2.0 * np.pi)
cdef double log_prior_prec_c = log(prior_prec)
cdef double yty = 0.0

cdef double[:, :] XtX_mv
cdef double[:] Xty_mv

XtX_np_global = None
Xty_np_global = None

LOG_MODEL_PRIOR_BY_SIZE_np = np.empty(p_py + 1, dtype=np.float64)

for m_size_py in range(p_py + 1):
    LOG_MODEL_PRIOR_BY_SIZE_np[m_size_py] = (
        lgamma(a_pi_py + m_size_py)
        + lgamma(b_pi_py + p_py - m_size_py)
        - lgamma(p_py + a_pi_py + b_pi_py)
        - (
            lgamma(a_pi_py)
            + lgamma(b_pi_py)
            - lgamma(a_pi_py + b_pi_py)
        )
    )

cdef double[:] LOG_MODEL_PRIOR_BY_SIZE_mv = LOG_MODEL_PRIOR_BY_SIZE_np

_MARGINAL_CACHE = {}

def load_dataset(int data_seed):
    global n
    global nu_c
    global yty
    global XtX_mv
    global Xty_mv
    global XtX_np_global
    global Xty_np_global

    dataset_file = data_dir / f"seed_{data_seed:02d}.csv"

    if not dataset_file.exists():
        raise FileNotFoundError(
            f"Cannot find dataset: {dataset_file.resolve()}"
        )

    dataset_df = pd.read_csv(dataset_file)

    required_cols = ["Intercept"] + feature_cols + ["y"]

    missing_cols = [
        col
        for col in required_cols
        if col not in dataset_df.columns
    ]

    if missing_cols:
        raise ValueError(
            f"Dataset seed {data_seed} is missing columns: {missing_cols}"
        )

    X_full_np = dataset_df[["Intercept"] + feature_cols].to_numpy(
        dtype=np.float64
    )

    y_np = dataset_df["y"].to_numpy(dtype=np.float64)

    if len(y_np) < 3 or not np.isfinite(X_full_np).all() or not np.isfinite(y_np).all():
        raise ValueError(f'{dataset_file.name}: need at least three finite observations.')
    if not np.allclose(X_full_np[:, 0], 1.0, rtol=0, atol=1e-12):
        raise ValueError(f'{dataset_file.name}: Intercept must equal one.')
    if float(np.var(y_np)) <= 0.0:
        raise ValueError(f'{dataset_file.name}: y must have positive variance.')

    if X_full_np.shape[1] != d:
        raise ValueError(
            f"Expected {d} design columns, found {X_full_np.shape[1]}."
        )

    if not np.allclose(X_full_np[:, 0], 1.0):
        raise ValueError("The first design column must be the intercept.")

    n = <int>X_full_np.shape[0]
    nu_c = n - 1.0

    XtX_np_global = np.asarray(
        X_full_np.T @ X_full_np,
        dtype=np.float64,
    )

    Xty_np_global = np.asarray(
        X_full_np.T @ y_np,
        dtype=np.float64,
    )

    yty = float(y_np @ y_np)

    XtX_mv = XtX_np_global
    Xty_mv = Xty_np_global

    _MARGINAL_CACHE.clear()

    return dataset_file

cdef int fill_active_design_indices(unsigned char[:] G, int[:] active):
    cdef int j
    cdef int q = 1

    active[0] = 0                           

    for j in range(p):
        if G[j] == 1:
            active[q] = j + 1
            q += 1

    return q

cdef int model_size(unsigned char[:] G):
    cdef int j
    cdef int m = 0

    for j in range(p):
        m += G[j]

    return m

cdef double log_model_prior_size_gibbs(int m):
    return LOG_MODEL_PRIOR_BY_SIZE_mv[m]

cdef double log_marginal_likelihood_from_active(int[:] active, int q):
    cdef int k_slopes = q - 1
    cdef object key
    cdef object cached

    cdef double[:, :] L
    cdef double[:] b
    cdef double[:] z
    cdef double[:] sol

    cdef int i, j, r
    cdef double s
    cdef double quad
    cdef double logdet_M
    cdef double val

    key = tuple(active[i] for i in range(q))

    cached = _MARGINAL_CACHE.get(key)
    if cached is not None:
        return <double>cached

    L = np.zeros((q, q), dtype=np.float64)
    b = np.empty(q, dtype=np.float64)
    z = np.empty(q, dtype=np.float64)
    sol = np.empty(q, dtype=np.float64)

    for i in range(q):
        b[i] = Xty_mv[active[i]]

        for j in range(i + 1):
            s = XtX_mv[active[i], active[j]]

            if i == j:
                                                                            
                if active[i] != 0:
                    s += prior_prec

                s += jitter

            for r in range(j):
                s -= L[i, r] * L[j, r]

            if i == j:
                if s <= 0.0:
                    raise FloatingPointError("Gibbs Cholesky factor has a nonpositive pivot.")
                L[i, j] = sqrt(s)
            else:
                L[i, j] = s / L[j, j]

    for i in range(q):
        s = b[i]
        for j in range(i):
            s -= L[i, j] * z[j]
        z[i] = s / L[i, i]

    for i in range(q - 1, -1, -1):
        s = z[i]
        for j in range(i + 1, q):
            s -= L[j, i] * sol[j]
        sol[i] = s / L[i, i]

    quad = yty
    for i in range(q):
        quad -= b[i] * sol[i]

    if quad <= 0.0:
        quad = jitter

    logdet_M = 0.0
    for i in range(q):
        logdet_M += 2.0 * log(L[i, i])

    val = (
        lgamma(0.5 * nu_c)
        - 0.5 * nu_c * log_2pi_c
        + 0.5 * k_slopes * log_prior_prec_c
        - 0.5 * logdet_M
        - 0.5 * nu_c * log(0.5 * quad)
    )

    if len(_MARGINAL_CACHE) > MAX_CACHE_SIZE:
        _MARGINAL_CACHE.clear()

    _MARGINAL_CACHE[key] = val

    return val

cdef double log_joint_gibbs(unsigned char[:] G):
    cdef int[:] active = np.empty(d, dtype=np.int32)
    cdef int q = fill_active_design_indices(G, active)
    cdef int m = q - 1

    return (
        log_marginal_likelihood_from_active(active, q)
        + log_model_prior_size_gibbs(m)
    )

cdef double prob_from_log_scores(double logp0, double logp1):
    cdef double mx
    cdef double w0
    cdef double w1

    if logp0 > logp1:
        mx = logp0
    else:
        mx = logp1

    w0 = exp(logp0 - mx)
    w1 = exp(logp1 - mx)

    return w1 / (w0 + w1)

def make_chain_initialization(int chain, int seed):
    if chain == 1:
        return np.zeros(p, dtype=np.uint8)

    if chain == 2:
        return np.ones(p, dtype=np.uint8)

    if chain == 3:
        g = np.zeros(p, dtype=np.uint8)
        g[::2] = 1
        return g

    rng = np.random.default_rng(seed)
    return rng.binomial(1, 0.5, size=p).astype(np.uint8)

def run_gibbs(
    int n_iter=GIBBS_N_ITER,
    int burn_in=GIBBS_BURN_IN,
    int thin=GIBBS_THIN,
    bint random_scan=RANDOM_SCAN,
    object verbose_every=VERBOSE_EVERY,
    int method_seed=METHOD_SEED,
    object gamma_init=None,
):
    gibbs_start = time.time()

    cdef int it
    cdef int j
    cdef int jj
    cdef int saved_pos
    cdef int flip_count

    cdef double logp0
    cdef double logp1
    cdef double prob1
    cdef double u
    cdef double logpost
    cdef int m

    cdef unsigned char old_value
    cdef unsigned char[:] G
    cdef int[:] coords
    cdef unsigned char[:, :] samples
    cdef double[:] logpost_hist
    cdef int[:] model_size_hist
    cdef int[:] flip_count_hist
    cdef double[:, :] uniforms

    if thin < 1 or burn_in < 0 or n_iter <= burn_in:
        raise ValueError("Require thin >= 1, burn_in >= 0, and n_iter > burn_in.")
    np.random.seed(method_seed)

    if gamma_init is None:
        G_np = np.zeros(p, dtype=np.uint8)
    else:
        G_np = np.asarray(gamma_init, dtype=np.uint8).copy()
        if G_np.shape != (p,):
            raise ValueError(f"gamma_init must have shape ({p},).")
        if np.any((G_np != 0) & (G_np != 1)):
            raise ValueError("gamma_init must contain only 0/1 values.")

    G = G_np

    coords_np = np.arange(p, dtype=np.int32)
    coords = coords_np

    max_saved = max(0, (n_iter - burn_in + thin - 1) // thin)

    samples_np = np.zeros((max_saved, p), dtype=np.uint8)
    logpost_hist_np = np.empty(n_iter, dtype=np.float64)
    model_size_hist_np = np.empty(n_iter, dtype=np.int32)
    flip_count_hist_np = np.empty(n_iter, dtype=np.int32)

    uniforms_np = np.random.random((n_iter, p)).astype(np.float64)
    uniforms = uniforms_np

    samples = samples_np
    logpost_hist = logpost_hist_np
    model_size_hist = model_size_hist_np
    flip_count_hist = flip_count_hist_np

    _MARGINAL_CACHE.clear()
    saved_pos = 0

    for it in range(1, n_iter + 1):

        if random_scan:
            coords_np = np.random.permutation(p).astype(np.int32)
            coords = coords_np
        else:
            for j in range(p):
                coords[j] = j

        flip_count = 0

        for jj in range(p):
            j = coords[jj]
            old_value = G[j]

            G[j] = 0
            logp0 = log_joint_gibbs(G)

            G[j] = 1
            logp1 = log_joint_gibbs(G)

            prob1 = prob_from_log_scores(logp0, logp1)
            u = uniforms[it - 1, jj]

            if u < prob1:
                G[j] = 1
            else:
                G[j] = 0

            if G[j] != old_value:
                flip_count += 1

        logpost = log_joint_gibbs(G)
        m = model_size(G)

        logpost_hist[it - 1] = logpost
        model_size_hist[it - 1] = m
        flip_count_hist[it - 1] = flip_count

        if it > burn_in and ((it - burn_in) % thin == 0):
            for j in range(p):
                samples[saved_pos, j] = G[j]
            saved_pos += 1

        if verbose_every is not None and it % verbose_every == 0:
            print(
                f"Gibbs iter {it:6d} | "
                f"logpost={logpost:.3f} | "
                f"model_size={m:3d} | "
                f"flips={flip_count:3d}/{p} | "
                f"cache_size={len(_MARGINAL_CACHE)}"
            )

    if saved_pos == 0:
        raise ValueError(
            "No saved samples. Increase n_iter or reduce burn_in/thin."
        )

    samples_np = samples_np[:saved_pos]
    pip_np = samples_np.mean(axis=0)

    return {
        "pip": pip_np,
        "samples": samples_np,
        "logpost": logpost_hist_np,
        "model_size": model_size_hist_np,
        "flip_count": flip_count_hist_np,
        "time": time.time() - gibbs_start,
        "n_iter": n_iter,
        "burn_in": burn_in,
        "thin": thin,
        "method_seed": method_seed,
        "cache_size": len(_MARGINAL_CACHE),
        "initial_gamma": np.asarray(gamma_init if gamma_init is not None else np.zeros(p), dtype=np.uint8),
        "final_gamma": G_np.copy(),
        "rng_state": np.random.get_state(),
    }

import hashlib

def atomic_csv(frame, path):
    temporary = path.with_suffix(".tmp")
    frame.to_csv(temporary, index=False, float_format="%.17g")
    temporary.replace(path)

def run_training():
    rows, variable_tables, summary_rows = [], [], []
    model_hash = hashlib.sha256(true_model_file.read_bytes()).hexdigest()
    for data_seed in DATA_SEEDS:
        loaded_file = load_dataset(data_seed)
        data_hash = hashlib.sha256(loaded_file.read_bytes()).hexdigest()
        chains = []
        for chain in range(1, GIBBS_N_CHAINS + 1):
            chain_seed = METHOD_SEED + 100 * (chain - 1)
            print(f"Sampling seed {data_seed:02d} | chain {chain}", flush=True)
            initial = make_chain_initialization(chain, chain_seed)
            out = run_gibbs(n_iter=GIBBS_N_ITER, burn_in=GIBBS_BURN_IN,
                            thin=GIBBS_THIN, random_scan=RANDOM_SCAN,
                            verbose_every=VERBOSE_EVERY, method_seed=chain_seed,
                            gamma_init=initial)
            chains.append(out)
            np.savez_compressed(
                output_dir / f"gibbs_draws_seed_{data_seed:02d}_chain_{chain}.npz",
                samples=out["samples"], logpost=out["logpost"],
                model_size=out["model_size"], flip_count=out["flip_count"],
                initial_gamma=out["initial_gamma"], final_gamma=out["final_gamma"],
            )
        chain_pips = np.column_stack([out["pip"] for out in chains])
        pip = chain_pips.mean(axis=1)
        export = {"variable_name": feature_cols, "is_true_active": truth_np.astype(int),
                  "beta": beta_slopes_np, "pip_gibbs": pip}
        for c in range(GIBBS_N_CHAINS):
            export[f"pip_chain_{c + 1}"] = chain_pips[:, c]
        atomic_csv(pd.DataFrame(export), output_dir / f"pips_gibbs_seed_{data_seed:02d}.csv")
        chain_table = pd.concat([pd.DataFrame({"seed": data_seed, "chain": c + 1,
                               "variable_name": feature_cols, "pip": out["pip"]})
                               for c, out in enumerate(chains)], ignore_index=True)
        atomic_csv(chain_table, output_dir / f"chain_pips_gibbs_seed_{data_seed:02d}.csv")
        runtime = float(sum(out["time"] for out in chains))
        row = dict(seed=int(data_seed), method="Gibbs", n_chains=GIBBS_N_CHAINS,
                   total_sweeps_per_chain=GIBBS_N_ITER, burn_in_per_chain=GIBBS_BURN_IN,
                   retained_sweeps_per_chain=GIBBS_KEEP_SWEEPS, thin=GIBBS_THIN,
                   total_sweeps_all_chains=GIBBS_N_CHAINS * GIBBS_N_ITER,
                   total_retained_draws=sum(len(out["samples"]) for out in chains),
                   method_seed=METHOD_SEED, tau2=tau2_py, a_pi=a_pi_py, b_pi=b_pi_py,
                   n=int(n), p=p_py,
                   data_sha256=data_hash, true_model_sha256=model_hash)
        row["time (s)"] = runtime
        rows.append(row)
        atomic_csv(pd.DataFrame([row]), output_dir / f"runtime_and_metrics_gibbs_seed_{data_seed:02d}.csv")
        variable_tables.append(pd.DataFrame({"replicate": data_seed, "variable": feature_cols,
                               "index": np.arange(1, p_py + 1), "true_active": truth_np > 0.5,
                               "true_beta": beta_slopes_np, "gibbs_pip": pip}))
        summary_rows.append(dict(replicate=data_seed, n=int(n), p=p_py, gibbs_runtime_seconds=runtime))
    atomic_csv(pd.DataFrame(rows), output_dir / "runtime_and_metrics_all_seeds_gibbs.csv")
    atomic_csv(pd.concat(variable_tables, ignore_index=True), output_dir / "gibbs_pips_by_variable.csv")
    atomic_csv(pd.DataFrame(summary_rows), output_dir / "gibbs_summary_by_replicate.csv")
    print("Gibbs complete. Total wall time:", time.time() - start_all)
