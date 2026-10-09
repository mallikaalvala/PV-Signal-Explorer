"""Reusable graphical visualizations for pharmacovigilance signal review."""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def forest_plot(df: pd.DataFrame, label_col="Stratum", title="Subgroup ROR forest plot"):
    x = df.dropna(subset=["ROR", "ROR_CI_lower", "ROR_CI_upper"]).copy()
    x = x[(x["ROR"] > 0) & (x["ROR_CI_lower"] > 0) & (x["ROR_CI_upper"] > 0)]
    if x.empty:
        return None
    y = np.arange(len(x))
    fig, ax = plt.subplots(figsize=(9, max(4, 0.42 * len(x))))
    err = np.vstack([x["ROR"].to_numpy()-x["ROR_CI_lower"].to_numpy(), x["ROR_CI_upper"].to_numpy()-x["ROR"].to_numpy()])
    ax.errorbar(x["ROR"], y, xerr=err, fmt="o", capsize=3)
    ax.axvline(1.0, linestyle="--", linewidth=1)
    ax.set_yticks(y); ax.set_yticklabels(x[label_col].astype(str))
    ax.set_xscale("log"); ax.set_xlabel("Reporting Odds Ratio (95% CI; log scale)")
    ax.set_title(title); ax.invert_yaxis(); fig.tight_layout()
    return fig


def volcano_plot(df: pd.DataFrame, title="Disproportionality signal map"):
    x = df.copy()
    x = x[(x["ROR"] > 0) & x["ROR"].notna() & x["chi_sq"].notna()]
    if x.empty:
        return None
    x["log2_ROR"] = np.log2(x["ROR"])
    p = pd.Series(np.maximum(1e-300, __import__('scipy').stats.chi2.sf(x["chi_sq"], 1)), index=x.index)
    x["minus_log10_p"] = -np.log10(p)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(x["log2_ROR"], x["minus_log10_p"], s=np.clip(x["a"], 8, 120), alpha=.55)
    ax.axvline(1.0, linestyle="--", linewidth=1)
    ax.set_xlabel("log2(ROR)"); ax.set_ylabel("-log10(chi-square p-value)"); ax.set_title(title)
    fig.tight_layout(); return fig


def top_signals_plot(df: pd.DataFrame, n=15):
    x = df[df["is_signal"]].dropna(subset=["ROR"]).nlargest(n, "ROR").copy()
    if x.empty:
        return None
    labels=(x["DRUG"]+" / "+x["AE"]).str.slice(0,70)
    fig, ax=plt.subplots(figsize=(10,max(4,.38*len(x))))
    y=np.arange(len(x)); ax.barh(y,x["ROR"]); ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.set_xlabel("ROR"); ax.set_title(f"Top {len(x)} screening-positive signals by ROR"); ax.invert_yaxis(); fig.tight_layout(); return fig


def subgroup_heatmap(stratified: pd.DataFrame, subgroup_col: str, drug: str, ae: str):
    x=stratified[(stratified.DRUG==drug)&(stratified.AE==ae)].dropna(subset=["ROR"]).copy()
    if x.empty: return None
    vals=np.log2(x["ROR"].clip(lower=1e-9)).to_numpy()[None,:]
    fig,ax=plt.subplots(figsize=(max(6,.8*len(x)),2.2)); im=ax.imshow(vals,aspect="auto")
    ax.set_xticks(np.arange(len(x))); ax.set_xticklabels(x[subgroup_col].astype(str),rotation=35,ha="right")
    ax.set_yticks([0]); ax.set_yticklabels(["log2 ROR"]); ax.set_title(f"{drug} / {ae} — subgroup signal heatmap")
    fig.colorbar(im,ax=ax,label="log2(ROR)"); fig.tight_layout(); return fig


def completeness_plot(quality_df: pd.DataFrame):
    if quality_df.empty: return None
    fig,ax=plt.subplots(figsize=(8,4)); ax.bar(quality_df["Variable"],quality_df["Completeness %"])
    ax.axhline(70,linestyle="--",linewidth=1); ax.set_ylim(0,100); ax.set_ylabel("Completeness (%)"); ax.set_title("Subgroup data completeness")
    fig.tight_layout(); return fig
