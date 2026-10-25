"""Two-chain convergence diagnostics for the p = 1000 Gibbs comparator.

Chain 1: the thesis reference chain (empty start, output_gibbs_tau2_1).
Chain 2: 20 random predictors start (output_gibbs_tau2_1_chain2_random20).
Both chains: 20,000 burn-in + 100,000 retained sweeps; only retained sweeps are used.

Per dataset, using the same definitions as the p = 30 Gibbs diagnostics (Appendix E,
ArviZ 0.22, rank-normalised split R-hat, bulk/tail ESS, MCSE of the mean):
  * every inclusion indicator that varies across the retained draws of the two chains
    (constant indicators are recorded as undefined, as at p = 30);
  * the log-posterior and model-size traces;
  * between-chain PIP differences and RMSE, and the split-half check of chain 1
    (first vs second 50,000 retained draws) that Table E.2 reports.
Writes diagnostics/: by-variable, scalar-trace and per-dataset CSVs, a summary CSV and
p1000_two_chain_diagnostics.tex. Uses ArviZ 0.22 when installed (arviz_legacy env),
otherwise the NumPy port of the same functions in arviz022_port.py.
"""
import argparse
import sys
import multiprocessing as mp
import os
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd

HERE = Path(__file__).resolve().parent
REF_DIR = HERE.parent / "output_gibbs_tau2_1"
CH2_DIR = HERE / "output_gibbs_tau2_1_chain2_random20"
OUT = HERE / "diagnostics"
SEEDS = list(range(50))
BURN = 20000
KEEP = 100000
P = 1000
PIP_FLOOR = 0.05     # secondary summaries restricted to indicators with pooled PIP >= 0.05


def _backend():
    try:
        import arviz as az
        if az.__version__.startswith("0.22"):
            def f(fun):
                return lambda *a, **k: float(np.asarray(fun(*a, **k)).item())
            return dict(name="arviz " + az.__version__,
                        rhat=f(lambda x: az.rhat(x, method="rank")),
                        ess_bulk=f(lambda x: az.ess(x, method="bulk")),
                        ess_tail=f(lambda x: az.ess(x, method="tail", prob=(0.05, 0.95))),
                        ess_mean=f(lambda x: az.ess(x, method="mean")),
                        mcse_mean=f(lambda x: az.mcse(x, method="mean")))
    except ImportError:
        pass
    import arviz022_port as A
    return dict(name="numpy port of arviz 0.22", rhat=A.rhat_rank, ess_bulk=A.ess_bulk,
                ess_tail=A.ess_tail, ess_mean=A.ess_mean, mcse_mean=A.mcse_mean)


_B = None


def _trace_diagnostics(values):
    """Same definition as the p = 30 Gibbs diagnostics (rank R-hat, bulk/tail/mean ESS, MCSE)."""
    global _B
    if _B is None:
        _B = _backend()
    x = np.asarray(values, dtype=float)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 4 or not np.isfinite(x).all():
        raise ValueError('Gibbs diagnostics require at least two finite chains with four retained draws.')
    if not bool(np.ptp(x) > 0):
        return dict(r_hat=np.nan, ess_bulk=np.nan, ess_tail=np.nan, ess_mean=np.nan,
                    mcse_mean=np.nan, status='constant_across_all_retained_draws')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        ans = dict(r_hat=_B["rhat"](x), ess_bulk=_B["ess_bulk"](x), ess_tail=_B["ess_tail"](x),
                   ess_mean=_B["ess_mean"](x), mcse_mean=_B["mcse_mean"](x))
    ans['status'] = ('estimated' if np.isfinite(list(ans.values())).all()
                     else 'nonfinite_diagnostic_inspect_chain_traces')
    return ans


def _load(path):
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k] for k in ["samples", "logpost", "model_size", "initial_gamma"]}
    if d["samples"].shape != (KEEP, P) or d["logpost"].shape != (BURN + KEEP,):
        raise ValueError(f"unexpected shapes in {path}")
    return d


def one_seed(seed):
    import pickle
    cache = OUT / "_per_seed" / f"seed_{seed:02d}.pkl"
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    res = _one_seed(seed)
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(".tmp")
    tmp.write_bytes(pickle.dumps(res))
    tmp.replace(cache)
    return res


def _one_seed(seed):
    c1 = _load(REF_DIR / f"gibbs_draws_seed_{seed:02d}_chain_1.npz")
    c2 = _load(CH2_DIR / f"gibbs_draws_seed_{seed:02d}_chain_2.npz")
    S1, S2 = c1["samples"], c2["samples"]
    pip1, pip2 = S1.mean(axis=0), S2.mean(axis=0)
    pooled = 0.5 * (pip1 + pip2)
    names = [f"X{j + 1}" for j in range(P)]

    var_rows = []
    for j in np.flatnonzero((S1.max(axis=0) != S1.min(axis=0)) | (S2.max(axis=0) != S2.min(axis=0))
                            | (S1[0] != S2[0])):
        x = np.stack([S1[:, j], S2[:, j]]).astype(float)
        var_rows.append(dict(seed=seed, variable_name=names[j], pip_chain_1=pip1[j],
                             pip_chain_2=pip2[j], pip_pooled=pooled[j],
                             abs_pip_diff=abs(pip1[j] - pip2[j]), **_trace_diagnostics(x)))
    V = pd.DataFrame(var_rows)

    scal_rows = []
    for q in ["logpost", "model_size"]:
        x = np.stack([c1[q][BURN:], c2[q][BURN:]]).astype(float)
        scal_rows.append(dict(seed=seed, quantity={"logpost": "log_posterior"}.get(q, q),
                              mean_chain_1=x[0].mean(), mean_chain_2=x[1].mean(),
                              **_trace_diagnostics(x)))

    # sweeps until chain 2 first reaches the bulk of chain 1's retained log posterior
    lp_floor = np.quantile(c1["logpost"][BURN:], 0.05)
    hit = np.flatnonzero(c2["logpost"] >= lp_floor)
    sweeps_to_bulk = int(hit[0] + 1) if hit.size else -1

    d = np.abs(pip1 - pip2)
    h = np.abs(S1[: KEEP // 2].mean(axis=0) - S1[KEEP // 2:].mean(axis=0))
    est = V[V.status.eq("estimated")] if len(V) else V
    big = est[est.pip_pooled >= PIP_FLOOR] if len(est) else est

    def mx(s):
        return float(s.max()) if len(s) else np.nan

    def mn(s):
        return float(s.min()) if len(s) else np.nan

    summary = dict(
        seed=seed,
        n_varying_indicators=int(len(V)),
        n_estimated=int(len(est)),
        n_nonfinite=int(V.status.eq("nonfinite_diagnostic_inspect_chain_traces").sum()) if len(V) else 0,
        max_inclusion_r_hat=mx(est.r_hat) if len(est) else np.nan,
        n_inclusion_r_hat_above_1_01=int((est.r_hat > 1.01).sum()) if len(est) else 0,
        min_inclusion_ess_bulk=mn(est.ess_bulk) if len(est) else np.nan,
        max_pip_mcse=mx(est.mcse_mean) if len(est) else np.nan,
        n_indicators_pip_ge_floor=int(len(big)),
        max_inclusion_r_hat_pip_ge_floor=mx(big.r_hat) if len(big) else np.nan,
        min_inclusion_ess_bulk_pip_ge_floor=mn(big.ess_bulk) if len(big) else np.nan,
        logpost_r_hat=scal_rows[0]["r_hat"], logpost_ess_bulk=scal_rows[0]["ess_bulk"],
        model_size_r_hat=scal_rows[1]["r_hat"], model_size_ess_bulk=scal_rows[1]["ess_bulk"],
        max_between_chain_pip_diff=float(d.max()),
        n_between_chain_pip_diff_gt_0_01=int((d > 0.01).sum()),
        between_chain_pip_rmse=float(np.sqrt(np.mean((pip1 - pip2) ** 2))),
        max_split_half_pip_diff_chain_1=float(h.max()),
        n_split_half_pip_diff_gt_0_01_chain_1=int((h > 0.01).sum()),
        chain2_sweeps_to_bulk=sweeps_to_bulk,
    )
    return summary, V, pd.DataFrame(scal_rows)


def tex_table(S):
    def f(x, k=4):
        return f"{x:.{k}f}"
    rows = [
        ("Mean dataset-wise maximum indicator \\(\\widehat R\\)", f(S.max_inclusion_r_hat.mean(), 5)),
        ("Datasets with maximum indicator \\(\\widehat R>1.01\\)",
         f"\\({int((S.max_inclusion_r_hat > 1.01).sum())}/{len(S)}\\)"),
        ("Largest \\(\\widehat R\\), log posterior / model size",
         f"{S.logpost_r_hat.max():.4f} / {S.model_size_r_hat.max():.4f}"),
        ("Mean dataset-wise maximum between-chain PIP difference", f(S.max_between_chain_pip_diff.mean())),
        ("Mean between-chain PIP RMSE", f(S.between_chain_pip_rmse.mean(), 5)),
        ("Smallest dataset-wise minimum bulk ESS (indicators with PIP \\(\\ge 0.05\\))",
         f"{S.min_inclusion_ess_bulk_pip_ge_floor.min():.1f}"),
        ("Largest dataset-wise maximum PIP MCSE", f(S.max_pip_mcse.max(), 5)),
    ]
    body = "\n".join(f"{a}\n& {b} \\\\\n" for a, b in rows)
    return ("\\begin{table}[H]\n\\centering\n"
            "\\caption{Two-chain convergence diagnostics for the \\(p=1000\\) Gibbs comparator.}\n"
            "\\label{tab:gibbs-p1000-two-chain}\n\\thesistablestyle\n"
            "\\begin{tabular*}{\\linewidth}{@{\\extracolsep{\\fill}}p{0.70\\linewidth}r@{}}\n"
            "\\toprule\nDiagnostic & Value \\\\\n\\midrule\n" + body +
            "\\bottomrule\n\\end{tabular*}\n\\end{table}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=int(os.environ.get("N_WORKERS", 6)))
    ap.add_argument("--max-new", type=int, default=None,
                    help="process at most this many uncached datasets, then stop (for short sessions)")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    print("Diagnostics backend:", _backend()["name"], flush=True)
    todo = [s for s in SEEDS if not (OUT / "_per_seed" / f"seed_{s:02d}.pkl").exists()]
    if args.max_new is not None and len(todo) > args.max_new:
        with mp.get_context("spawn").Pool(args.workers) as pool:
            pool.map(one_seed, todo[: args.max_new])
        print(f"cached {args.max_new}; {len(todo) - args.max_new} datasets remain", flush=True)
        return
    truth = pd.read_csv(HERE.parent / "data" / "fixed_beta_and_active_set.csv")
    truth = truth[truth.is_intercept.astype(int) == 0]
    true_names = set(truth.loc[truth.is_true_active.astype(int) == 1, "variable_name"])
    with mp.get_context("spawn").Pool(args.workers) as pool:
        res = pool.map(one_seed, SEEDS)
    (OUT / "BACKEND.txt").write_text(_backend()["name"] + "\n")
    S = pd.DataFrame([r[0] for r in res])
    meta = pd.concat([pd.read_csv(CH2_DIR / f"runtime_gibbs_seed_{s:02d}_chain_2.csv") for s in SEEDS])
    S["init_true_active"] = meta.init_variables.map(
        lambda v: len(set(v.split()) & true_names)).to_numpy()
    S["chain2_time_s"] = meta["time (s)"].to_numpy()
    V = pd.concat([r[1] for r in res], ignore_index=True)
    T = pd.concat([r[2] for r in res], ignore_index=True)
    S.to_csv(OUT / "p1000_two_chain_by_dataset.csv", index=False)
    V.to_csv(OUT / "p1000_two_chain_by_variable.csv", index=False)
    T.to_csv(OUT / "p1000_two_chain_scalar_traces.csv", index=False)
    agg = S.drop(columns="seed").agg(["mean", "median", "min", "max"]).T
    agg.to_csv(OUT / "p1000_two_chain_summary.csv")
    (OUT / "p1000_two_chain_diagnostics.tex").write_text(tex_table(S))
    with pd.option_context("display.width", 200, "display.max_rows", 100):
        print(agg)
    print("\nWorst datasets by max indicator R-hat:")
    print(S.sort_values("max_inclusion_r_hat", ascending=False)
          [["seed", "max_inclusion_r_hat", "max_between_chain_pip_diff", "logpost_r_hat"]].head(5))
    print("\nIndicators with R-hat > 1.01:")
    print(V[V.r_hat > 1.01].sort_values("r_hat", ascending=False).head(20).to_string())
    print("\nWrote", OUT)


if __name__ == "__main__":
    main()
