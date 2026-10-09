import numpy as np
import pandas as pd
from pv_signal_explorer.bayesian import expected_counts_from_pairs, compute_information_component, compute_ebgm
from pv_signal_explorer.temporal import attach_period

def test_expected_counts():
    p=pd.DataFrame({"CASE_ID":["1","2","3","4"],"DRUG":["A","A","B","B"],"AE":["X","X","X","Y"]})
    c=expected_counts_from_pairs(p)
    r=c[(c.DRUG=="A")&(c.AE=="X")].iloc[0]
    assert r.observed==2 and np.isclose(r.expected,1.5)

def test_ic_finite():
    c=pd.DataFrame({"observed":[10],"drug_total":[20],"ae_total":[30],"N":[100],"expected":[6.]})
    r=compute_information_component(c)
    assert np.isfinite(r.loc[0,"IC"]) and r.loc[0,"IC"]>0

def test_ebgm_shrinks_sparse():
    c=pd.DataFrame({"observed":[1,2,5,10,20,50],"expected":[.1,.5,2,5,10,30]})
    r,p=compute_ebgm(c)
    assert (r.EBGM>0).all() and (r.EB05<=r.EBGM).all() and (r.EBGM<=r.EB95).all()

def test_attach_quarter():
    p=pd.DataFrame({"CASE_ID":["1"],"DRUG":["A"],"AE":["X"]})
    d=pd.Series([pd.Timestamp("2025-05-01")],index=["1"])
    x=attach_period(p,d)
    assert x.iloc[0].PERIOD=="2025Q2"
