"""Body Fat prior sensitivity (Appendix F): m0 in {1.5, 3, 6}. Reads the three post-processing runs and writes
prior_sensitivity_results/ with a long CSV, a LaTeX table, the exact PIPs per prior and an
ordering check of the main conclusions."""
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
RUNS = {1.5: HERE / 'bodyfat_POSTPROCESSING_tau2_1_final_S1_V20_m0_1p5',
        3.0: HERE / 'bodyfat_POSTPROCESSING_tau2_1_final_S1_V20',
        6.0: HERE / 'bodyfat_POSTPROCESSING_tau2_1_final_S1_V20_m0_6'}
METHODS = ['ELBO', 'VIMCO*', 'VIMCO* annealed', 'VIMCO-GM', 'VIMCO-GM annealed']
TEX = {'ELBO': 'ELBO', 'VIMCO*': r'\vstar', 'VIMCO* annealed': r'\avstar', 'VIMCO-GM': r'\vgm', 'VIMCO-GM annealed': r'\avgm'}
STAGES = ['Original VI', 'Original PSIS', 'IWMM PSIS', 'Damped PSIS']
STAGE_NAME = {'Original VI': 'Original VI', 'Original PSIS': 'Ordinary PSIS', 'IWMM PSIS': 'IWMM', 'Damped PSIS': 'Iterative Damped PSIS'}
METRICS = {'PIP RMSE vs exact': 'PIP RMSE', 'Mean excess marginal FKL vs exact': 'DeltaFKL', 'Pareto k-hat': 'k-hat',
           'importance_ESS': 'ESS', 'Exact-posterior expected 95% interval score': 'Expected interval score'}
OUT = HERE / 'prior_sensitivity_results'; OUT.mkdir(exist_ok=True)

rows, exact = [], {}
for m0, base in RUNS.items():
    run = Path((base / 'LATEST_RESULTS.txt').read_text().strip())
    d = pd.read_csv(run / 'selected_configuration_metrics.csv')
    for stage in STAGES:
        for method in METHODS:
            g = d[(d.stage == stage) & (d.method == method)]
            row = dict(m0=m0, run=str(run), stage=stage, method=method, n=len(g))
            for col, name in METRICS.items():
                x = pd.to_numeric(g[col], errors='coerce').to_numpy(float)
                if stage == 'Original VI' and name in ('ESS',):
                    x = np.full_like(x, np.nan)
                ok = np.isfinite(x)
                row[name] = x[ok].mean() if ok.any() else np.nan
                row[name + ' SD'] = x[ok].std(ddof=1) if ok.sum() > 1 else np.nan
                row[name + ' n_defined'] = int(ok.sum())
            rows.append(row)
    ref = pd.read_csv(run / 'exact_reference_pips.csv')
    exact[m0] = ref.set_index('variable_name')['exact_pip']
res = pd.DataFrame(rows)
res.to_csv(OUT / 'bodyfat_prior_sensitivity_long.csv', index=False)
pd.DataFrame({f'm0 = {m}': v for m, v in exact.items()}).to_csv(OUT / 'bodyfat_exact_pips_by_m0.csv')

# LaTeX (Table F.1): all four processing stages; columns m0 x (PIP RMSE, DeltaFKL)
m0s = list(RUNS)
lines = [r'\thesistablestyle', r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}ll' + 'r' * (2 * len(m0s)) + '@{}}', r'\toprule',
         r' & & ' + ' & '.join(r'\multicolumn{2}{c}{$m_0=' + f'{m:g}' + '$}' for m in m0s) + r'\\',
         ''.join(r'\cmidrule(lr){%d-%d}' % (3 + 2 * i, 4 + 2 * i) for i in range(len(m0s))),
         r'Procedure & Method & ' + ' & '.join([r'PIP RMSE & $\Delta\mathrm{FKL}$'] * len(m0s)) + r'\\', r'\midrule']
for k, stage in enumerate(STAGES):
    for j, method in enumerate(METHODS):
        cells = []
        for m in m0s:
            r = res[(res.m0 == m) & (res.stage == stage) & (res.method == method)].iloc[0]
            cells += [f'{r["PIP RMSE"]:.4f}', f'{r["DeltaFKL"]:.4f}']
        label = r'\shortstack[l]{Iterative Damped\\PSIS}' if stage == 'Damped PSIS' else STAGE_NAME[stage]
        first = (r'\multirow{5}{*}{' + label + '}') if j == 0 else ''
        lines.append(f'{first} & {TEX[method]} & ' + ' & '.join(cells) + r'\\')
    if k < len(STAGES) - 1:
        lines.append(r'\midrule')
lines += [r'\bottomrule', r'\end{tabular*}']
(OUT / 'bodyfat_prior_sensitivity_table.tex').write_text('\n'.join(lines) + '\n')

# Ordering checks behind the Section 5.1 conclusions
checks = []
for m in m0s:
    for stage in STAGES:
        s = res[(res.m0 == m) & (res.stage == stage)].set_index('method')
        for metric in ['PIP RMSE', 'DeltaFKL']:
            v = s[metric]
            checks.append(dict(m0=m, stage=stage, metric=metric,
                both_annealed_below_ELBO=bool((v[['VIMCO* annealed', 'VIMCO-GM annealed']] < v['ELBO']).all()),
                both_fixed_below_ELBO=bool((v[['VIMCO*', 'VIMCO-GM']] < v['ELBO']).all()),
                best_method=v.idxmin(), best_value=float(v.min()),
                annealed_reduction_vs_ELBO_pct=f"{100*(1-v['VIMCO* annealed']/v['ELBO']):.1f}, {100*(1-v['VIMCO-GM annealed']/v['ELBO']):.1f}",
                fixed_reduction_vs_ELBO_pct=f"{100*(1-v['VIMCO*']/v['ELBO']):.1f}, {100*(1-v['VIMCO-GM']/v['ELBO']):.1f}"))
checks = pd.DataFrame(checks); checks.to_csv(OUT / 'bodyfat_prior_sensitivity_ordering_checks.csv', index=False)
pd.set_option('display.width', 250)
print(res[['m0', 'stage', 'method', 'PIP RMSE', 'DeltaFKL', 'k-hat', 'k-hat n_defined', 'ESS', 'Expected interval score']].round(4).to_string(index=False))
print(checks.to_string(index=False))

# Paired comparison with ELBO for every prior: same optimisation seed, same prior, same exact reference.
from scipy.stats import t as student_t
paired = []
for m0, base in RUNS.items():
    run = Path((base / 'LATEST_RESULTS.txt').read_text().strip())
    d = pd.read_csv(run / 'selected_configuration_metrics.csv')
    for stage in STAGES:
        e = d[(d.stage == stage) & (d.method == 'ELBO')].set_index('seed')
        for method in METHODS[1:]:
            c = d[(d.stage == stage) & (d.method == method)].set_index('seed')
            for col, name in [('PIP RMSE vs exact', 'PIP RMSE'), ('Mean excess marginal FKL vs exact', 'DeltaFKL')]:
                diff = (c[col] - e[col]).dropna().to_numpy(float)
                n = len(diff); mean = diff.mean(); se = diff.std(ddof=1) / np.sqrt(n); q = student_t.ppf(0.975, n - 1)
                paired.append(dict(m0=m0, stage=stage, method=method, metric=name, n=n, mean_difference=mean,
                                   ci_lower=mean - q * se, ci_upper=mean + q * se,
                                   relative_change_pct=100 * mean / e[col].mean(),
                                   share_of_seeds_better_than_ELBO=float((diff < 0).mean())))
paired = pd.DataFrame(paired)
paired.to_csv(OUT / 'bodyfat_prior_sensitivity_paired_vs_ELBO.csv', index=False)
print(paired.round(4).to_string(index=False))
print('Saved to', OUT)
