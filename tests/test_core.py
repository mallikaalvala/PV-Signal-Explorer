import pandas as pd
from pv_signal_explorer.core import stats_from_counts, _bh_fdr

def test_stats_basic():
    r=stats_from_counts(pd.Series([20]),pd.Series([80]),pd.Series([10]),pd.Series([890]),3)
    assert r.loc[0,"ROR"] > 1
    assert r.loc[0,"PRR"] > 1

def test_bh_monotonic_bounds():
    q=_bh_fdr(pd.Series([0.01,0.02,0.5]))
    assert q.dropna().between(0,1).all()
