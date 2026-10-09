"""Bayesian disproportionality methods for PV Signal Explorer.

Implements an open, reproducible two-component Gamma-Poisson empirical-Bayes
model (MGPS-style shrinkage) and a BCPNN-style Information Component (IC)
approximation. These are research implementations, not proprietary FDA
Empirica or UMC production algorithms.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import optimize, special, stats

EPS = 1e-12


def expected_counts_from_pairs(pairs: pd.DataFrame) -> pd.DataFrame:
    """Observed/expected counts under drug-event independence.

    Expected = N_drug * N_event / N, where N is unique reports. This is the
    natural background expectation for Bayesian reporting-ratio shrinkage.
    """
    n = pairs["CASE_ID"].nunique()
    drug = pairs.groupby("DRUG", observed=True)["CASE_ID"].nunique().rename("drug_total")
    ae = pairs.groupby("AE", observed=True)["CASE_ID"].nunique().rename("ae_total")
    x = pairs.groupby(["DRUG", "AE"], observed=True)["CASE_ID"].nunique().rename("observed").reset_index()
    x = x.merge(drug, on="DRUG").merge(ae, on="AE")
    x["N"] = n
    x["expected"] = x["drug_total"] * x["ae_total"] / float(n)
    x["RR_raw"] = x["observed"] / x["expected"].clip(lower=EPS)
    return x


def _log_nb_marginal(n, e, alpha, beta):
    # Gamma(shape=alpha, rate=beta) prior mixed with Poisson(e*lambda).
    return (special.gammaln(n + alpha) - special.gammaln(alpha) - special.gammaln(n + 1)
            + alpha*np.log(beta) + n*np.log(np.maximum(e, EPS))
            - (n + alpha)*np.log(beta + e))


def fit_gamma_poisson_mixture(observed, expected, max_pairs: int = 200_000, random_state: int = 7):
    """Fit a 2-Gamma mixture prior to reporting ratios by marginal MLE.

    Parameters are constrained positive through log transforms; mixture weight
    uses a logit transform. A deterministic sample is used for very large sets.
    """
    n = np.asarray(observed, dtype=float)
    e = np.asarray(expected, dtype=float)
    ok = np.isfinite(n) & np.isfinite(e) & (n >= 0) & (e > 0)
    n, e = n[ok], e[ok]
    if len(n) == 0:
        raise ValueError("No valid observed/expected counts for EBGM fitting.")
    if len(n) > max_pairs:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(n), max_pairs, replace=False)
        n, e = n[idx], e[idx]

    def unpack(theta):
        a1, b1, a2, b2 = np.exp(theta[:4])
        p = special.expit(theta[4])
        return a1, b1, a2, b2, p

    def objective(theta):
        a1,b1,a2,b2,p = unpack(theta)
        l1 = np.log(np.clip(p, EPS, 1)) + _log_nb_marginal(n,e,a1,b1)
        l2 = np.log(np.clip(1-p, EPS, 1)) + _log_nb_marginal(n,e,a2,b2)
        return -np.sum(np.logaddexp(l1,l2))

    starts = [
        np.array([np.log(0.5),np.log(0.5),np.log(3.0),np.log(1.5),0.0]),
        np.array([np.log(1.0),np.log(1.0),np.log(8.0),np.log(2.0),-0.5]),
    ]
    best = None
    for s in starts:
        r = optimize.minimize(objective, s, method="L-BFGS-B", options={"maxiter":500})
        if best is None or r.fun < best.fun:
            best = r
    a1,b1,a2,b2,p = unpack(best.x)
    # Identifiability: order components by prior mean.
    if a1/b1 > a2/b2:
        a1,b1,a2,b2,p = a2,b2,a1,b1,1-p
    return {"alpha1":a1,"beta1":b1,"alpha2":a2,"beta2":b2,"p":p,
            "converged":bool(best.success),"neg_loglik":float(best.fun),"n_fit":int(len(n))}


def _posterior_components(n, e, params):
    a1,b1,a2,b2,p = (params[k] for k in ("alpha1","beta1","alpha2","beta2","p"))
    l1 = np.log(np.clip(p,EPS,1)) + _log_nb_marginal(n,e,a1,b1)
    l2 = np.log(np.clip(1-p,EPS,1)) + _log_nb_marginal(n,e,a2,b2)
    z = np.logaddexp(l1,l2)
    w1 = np.exp(l1-z); w2 = 1-w1
    return w1,w2,a1+n,b1+e,a2+n,b2+e


def _mixture_quantile(q, w1, w2, a1, b1, a2, b2):
    def cdf(x):
        return w1*stats.gamma.cdf(x,a=a1,scale=1/b1) + w2*stats.gamma.cdf(x,a=a2,scale=1/b2)
    hi = max(2.0, (a1/b1)*10, (a2/b2)*10)
    while cdf(hi) < q and hi < 1e8:
        hi *= 2
    return optimize.brentq(lambda x: cdf(x)-q, EPS, hi, maxiter=100)


def compute_ebgm(counts: pd.DataFrame, params=None, fit_max_pairs=200_000) -> tuple[pd.DataFrame, dict]:
    """Add EBGM, EB05 and EB95 to a table with observed and expected counts."""
    out = counts.copy()
    if params is None:
        params = fit_gamma_poisson_mixture(out["observed"], out["expected"], max_pairs=fit_max_pairs)
    n = out["observed"].to_numpy(float); e = out["expected"].to_numpy(float)
    w1,w2,a1,b1,a2,b2 = _posterior_components(n,e,params)
    elog = w1*(special.digamma(a1)-np.log(b1)) + w2*(special.digamma(a2)-np.log(b2))
    out["EBGM"] = np.exp(elog)
    out["EB05"] = [_mixture_quantile(.05,*z) for z in zip(w1,w2,a1,b1,a2,b2)]
    out["EB95"] = [_mixture_quantile(.95,*z) for z in zip(w1,w2,a1,b1,a2,b2)]
    out["EB_signal"] = out["EB05"] >= 2.0
    return out, params


def compute_information_component(counts: pd.DataFrame, prior: float = 0.5) -> pd.DataFrame:
    """BCPNN-style shrinkage Information Component with an approximate 95% interval.

    Uses Jeffreys-style pseudo-count smoothing and a delta-method variance on
    log probabilities. IC025 > 0 is exposed as a screening flag. This is an
    open approximation and is not claimed to reproduce UMC's proprietary
    production implementation exactly.
    """
    out = counts.copy()
    a = out["observed"].astype(float)
    nd = out["drug_total"].astype(float)
    ne = out["ae_total"].astype(float)
    N = out["N"].astype(float)
    # Smoothed joint and marginal probabilities.
    p11 = (a + prior) / (N + 4*prior)
    pd_ = (nd + 2*prior) / (N + 4*prior)
    pe_ = (ne + 2*prior) / (N + 4*prior)
    out["IC"] = np.log2(p11 / (pd_*pe_))
    # Conservative delta approximation; covariance terms omitted.
    var_ln = 1/(a+prior) + 1/(nd+2*prior) + 1/(ne+2*prior)
    se_ic = np.sqrt(var_ln) / np.log(2)
    out["IC_SE"] = se_ic
    out["IC025"] = out["IC"] - 1.96*se_ic
    out["IC975"] = out["IC"] + 1.96*se_ic
    out["IC_signal"] = out["IC025"] > 0
    return out


def bayesian_signal_table(pairs: pd.DataFrame, min_reports: int = 1):
    counts = expected_counts_from_pairs(pairs)
    counts = counts[counts["observed"] >= min_reports].reset_index(drop=True)
    eb, params = compute_ebgm(counts)
    both = compute_information_component(eb)
    return both.sort_values(["EB05","IC025"], ascending=False), params
