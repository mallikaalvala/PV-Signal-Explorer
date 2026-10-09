"""Quarterly temporal pharmacovigilance surveillance."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats
from .bayesian import expected_counts_from_pairs, compute_ebgm, compute_information_component


def attach_period(pairs: pd.DataFrame, event_dt_by_case: pd.Series, freq="Q") -> pd.DataFrame:
    x = pairs[["CASE_ID","DRUG","AE"]].copy()
    x["EVENT_DT"] = x["CASE_ID"].map(event_dt_by_case)
    x = x.dropna(subset=["EVENT_DT"])
    x["PERIOD"] = pd.to_datetime(x["EVENT_DT"]).dt.to_period(freq).astype(str)
    return x


def temporal_surveillance(pairs, event_dt_by_case, min_reports=3, freq="Q", eb_params=None):
    """Compute quarter-specific RR/EBGM/IC using each period as its own universe."""
    dated = attach_period(pairs, event_dt_by_case, freq=freq)
    rows=[]
    params = eb_params
    for period, sub in dated.groupby("PERIOD", sort=True):
        c = expected_counts_from_pairs(sub)
        c = c[c["observed"] >= min_reports].copy()
        if c.empty: continue
        # Fit prior once on first sufficiently rich period unless a global prior is supplied.
        try:
            eb, fitted = compute_ebgm(c, params=params)
            if params is None: params = fitted
        except Exception:
            continue
        eb = compute_information_component(eb)
        eb["PERIOD"] = period
        rows.append(eb)
    if not rows:
        return pd.DataFrame(), params
    return pd.concat(rows, ignore_index=True), params


def pair_temporal_summary(temporal_df, drug, ae, eb05_threshold=2.0, ic025_threshold=0.0):
    x = temporal_df[(temporal_df.DRUG==drug)&(temporal_df.AE==ae)].copy().sort_values("PERIOD")
    if x.empty: return x, {}
    x["signal"] = (x["EB05"] >= eb05_threshold) | (x["IC025"] > ic025_threshold)
    first = x.loc[x.signal,"PERIOD"].iloc[0] if x.signal.any() else None
    # longest and current consecutive signal runs
    run=best=0
    for v in x.signal:
        run = run+1 if v else 0; best=max(best,run)
    current=run
    slope=p=np.nan
    if len(x)>=3:
        lr=stats.linregress(np.arange(len(x)), np.log2(x["EBGM"].clip(lower=1e-9)))
        slope,p=float(lr.slope),float(lr.pvalue)
    x["EBGM_qoq_pct"] = x["EBGM"].pct_change()*100
    max_jump = x["EBGM_qoq_pct"].abs().max() if len(x)>1 else np.nan
    summary={"first_signal_period":first,"signal_quarters":int(x.signal.sum()),
             "longest_signal_run":int(best),"current_signal_run":int(current),
             "log2_EBGM_trend_per_period":slope,"trend_p":p,"max_abs_qoq_EBGM_pct":max_jump,
             "latest_EBGM":float(x.iloc[-1].EBGM),"latest_EB05":float(x.iloc[-1].EB05),
             "latest_IC":float(x.iloc[-1].IC),"latest_IC025":float(x.iloc[-1].IC025)}
    return x, summary
