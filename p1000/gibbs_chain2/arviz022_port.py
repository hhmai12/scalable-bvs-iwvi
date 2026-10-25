"""NumPy-only port of the ArviZ 0.22 diagnostics used in Appendix E
(arviz.stats.diagnostics: _rhat_rank, _ess_bulk, _ess_tail, _ess_mean, _mcse_mean).
Used when ArviZ/SciPy are unavailable. Input: array of shape (chain, draw)."""
import numpy as np
from statistics import NormalDist

_ND = NormalDist()


def _rankdata_average(a):
    a = np.ravel(a)
    vals, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    upper = np.cumsum(counts)
    avg = upper - (counts - 1) / 2.0          # average 1-based rank of each tie group
    return avg[inv]


def _z_scale(ary):
    size = ary.size
    rank = _rankdata_average(ary)
    u = (rank - 0.375) / (size + 0.25)
    uu, inv = np.unique(u, return_inverse=True)
    z = np.array([_ND.inv_cdf(float(x)) for x in uu])[inv]
    return z.reshape(ary.shape)


def _split_chains(ary):
    half = ary.shape[1] // 2
    return np.vstack([ary[:, :half], ary[:, -half:]])


def _rhat(ary):
    ary = np.asarray(ary, dtype=float)
    n = ary.shape[1]
    chain_mean = ary.mean(axis=1)
    chain_var = ary.var(axis=1, ddof=1)
    between = n * np.var(chain_mean, ddof=1)
    within = np.mean(chain_var)
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.sqrt((between / within + n - 1) / n))


def rhat_rank(ary):
    split = _split_chains(np.asarray(ary, dtype=float))
    bulk = _rhat(_z_scale(split))
    folded = np.abs(split - np.median(split))
    tail = _rhat(_z_scale(folded))
    return max(bulk, tail)


def _autocov(ary):
    n = ary.shape[1]
    m = 1 << int(np.ceil(np.log2(2 * n)))
    x = ary - ary.mean(axis=1, keepdims=True)
    f = np.fft.rfft(x, n=m, axis=1)
    f *= np.conjugate(f)
    return np.fft.irfft(f, n=m, axis=1)[:, :n] / n


def _ess(ary):
    ary = np.asarray(ary, dtype=float)
    if (np.max(ary) - np.min(ary)) < np.finfo(float).resolution:
        return float(ary.size)
    n_chain, n_draw = ary.shape
    acov = _autocov(ary)
    chain_mean = ary.mean(axis=1)
    mean_var = np.mean(acov[:, 0]) * n_draw / (n_draw - 1.0)
    var_plus = mean_var * (n_draw - 1.0) / n_draw
    if n_chain > 1:
        var_plus += np.var(chain_mean, ddof=1)
    rho = np.zeros(n_draw)
    rho_even = 1.0
    rho[0] = rho_even
    rho_odd = 1.0 - (mean_var - np.mean(acov[:, 1])) / var_plus
    rho[1] = rho_odd
    t = 1
    while t < (n_draw - 3) and (rho_even + rho_odd) > 0.0:
        rho_even = 1.0 - (mean_var - np.mean(acov[:, t + 1])) / var_plus
        rho_odd = 1.0 - (mean_var - np.mean(acov[:, t + 2])) / var_plus
        if (rho_even + rho_odd) >= 0:
            rho[t + 1] = rho_even
            rho[t + 2] = rho_odd
        t += 2
    max_t = t - 2
    if rho_even > 0:
        rho[max_t + 1] = rho_even
    t = 1
    while t <= max_t - 2:
        if (rho[t + 1] + rho[t + 2]) > (rho[t - 1] + rho[t]):
            rho[t + 1] = (rho[t - 1] + rho[t]) / 2.0
            rho[t + 2] = rho[t + 1]
        t += 2
    ess = n_chain * n_draw
    tau = -1.0 + 2.0 * np.sum(rho[: max_t + 1]) + np.sum(rho[max_t + 1: max_t + 2])
    tau = max(tau, 1 / np.log10(ess))
    return float(ess / tau)


def ess_bulk(ary):
    return _ess(_z_scale(_split_chains(np.asarray(ary, dtype=float))))


def ess_mean(ary):
    return _ess(_split_chains(np.asarray(ary, dtype=float)))


def ess_tail(ary, prob=(0.05, 0.95)):
    ary = np.asarray(ary, dtype=float)
    q_lo, q_hi = np.quantile(ary, prob[0]), np.quantile(ary, prob[1])
    return min(_ess(_split_chains((ary <= q_lo).astype(float))),
               _ess(_split_chains((ary <= q_hi).astype(float))))


def mcse_mean(ary):
    ary = np.asarray(ary, dtype=float)
    return float(np.std(ary, ddof=1) / np.sqrt(ess_mean(ary)))
