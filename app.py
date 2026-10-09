"""
FAERS/AEMS Signal Detection with Subgroup Stratification and
Simpson's-Paradox Masking Detection.

Extends the combined app with a genuinely underexplored feature in most
FAERS tooling: computing PRR/ROR/chi-square not just on the pooled
(aggregate) dataset, but separately within age bands, sex, and reporter
type -- then formally testing whether those subgroup-specific estimates
are statistically consistent with each other (Cochran's Q, standard
meta-analysis heterogeneity test) and flagging two failure modes:

  - MASKED signal: aggregate looks null/weak, but a subgroup shows a
    strong, statistically consistent elevation that's being diluted
    away by pooling with subgroups that show no elevation.

  - CONFOUNDED signal: aggregate looks like a signal, but subgroup
    estimates are significantly heterogeneous / inconsistent in
    direction -- the pooled "signal" may not represent a real,
    uniform effect and could be an artifact of subgroup composition.

Usage:
    streamlit run app_stratified.py

Requires DRUG, REAC, and DEMO files (DEMO carries age/sex/reporter type).
"""

import gc
import io
import html
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="PV Signal Explorer V2", page_icon=":pill:", layout="wide")

try:
    from scipy import stats as scipy_stats
except ModuleNotFoundError:
    st.error(
        "The `scipy` package isn't installed, but this app needs it for the "
        "heterogeneity test (Cochran's Q p-value). Run:\n\n"
        "    pip install scipy\n\n"
        "then restart the app."
    )
    st.stop()

st.title("PV Signal Explorer V3.2 — FAERS/AEMS Signal Detection")
st.write(
    """
    Computes PRR / ROR / chi-square on the pooled dataset **and** separately
    within age, sex, and reporter-type subgroups, then tests whether those
    subgroup estimates are statistically consistent (Cochran's Q). Pairs
    where pooling masks a real subgroup-specific signal, or where a pooled
    signal doesn't hold up consistently across subgroups, are flagged
    explicitly -- a Simpson's-paradox check most disproportionality tools skip.
    """
)

ENCODINGS_TO_TRY = ["utf-8", "cp1252", "latin-1"]

CASE_ID_NAMES = ["primaryid", "caseid", "case_id", "id"]
DRUG_NAME_NAMES = ["drugname", "prod_ai", "drug_name", "drug"]
REACTION_NAMES = ["pt", "reaction", "reactionmeddrapt", "ae", "adverse_event"]
ROLE_COD_NAMES = ["role_cod", "role_code", "drug_role"]
AGE_NAMES = ["age"]
AGE_COD_NAMES = ["age_cod", "age_unit"]
SEX_NAMES = ["sex", "gndr_cod", "gender"]
OCCP_NAMES = ["occp_cod", "reporter_type", "occupation"]

AGE_UNIT_TO_YEARS = {
    "DEC": 10.0, "YR": 1.0, "MON": 1 / 12, "WK": 1 / 52,
    "DY": 1 / 365, "HR": 1 / (365 * 24),
}
OCCP_LABELS = {
    "MD": "Physician", "PH": "Pharmacist", "OT": "Other health professional",
    "LW": "Lawyer", "CN": "Consumer", "RN": "Registered nurse",
}
RPSR_LABELS = {
    "FGN": "Foreign", "CSM": "Consumer", "LIT": "Literature", "STUDY": "Study",
    "UF": "User facility", "HP": "Health professional",
}
SERIOUS_OUTCOME_CODES = {"DE", "LT", "HO", "DS", "CA", "RI", "OT"}
EVENT_DT_NAMES = ["event_dt"]
THER_CASE_ID_NAMES = CASE_ID_NAMES
THER_START_DT_NAMES = ["start_dt"]
OUTC_CASE_ID_NAMES = CASE_ID_NAMES
OUTC_CODE_NAMES = ["outc_cod", "outc_code"]
INDI_CASE_ID_NAMES = CASE_ID_NAMES
INDI_PT_NAMES = ["indi_pt", "indication"]
RPSR_CASE_ID_NAMES = CASE_ID_NAMES
RPSR_COD_NAMES = ["rpsr_cod", "report_source"]


def parse_faers_date(series: pd.Series) -> pd.Series:
    """FAERS dates come as 8-digit YYYYMMDD, but sometimes only YYYYMM or
    YYYY (partial precision). Falls back progressively; unparseable values
    become NaT rather than raising."""
    s = series.astype(str).str.strip()
    d8 = pd.to_datetime(s.where(s.str.len() == 8), format="%Y%m%d", errors="coerce")
    d6 = pd.to_datetime(s.where(s.str.len() == 6) + "01", format="%Y%m%d", errors="coerce")
    d4 = pd.to_datetime(s.where(s.str.len() == 4) + "0101", format="%Y%m%d", errors="coerce")
    return d8.fillna(d6).fillna(d4)


# ---------------------------------------------------------------------------
# Helpers: column detection and file reading (same pattern as before)
# ---------------------------------------------------------------------------

def find_column(df: pd.DataFrame, possible_names):
    columns_lower = {str(col).strip().lower(): col for col in df.columns}
    for name in possible_names:
        if name.lower() in columns_lower:
            return columns_lower[name.lower()]
    return None


def clean_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.upper()


def _wanted_filter(wanted):
    """usecols callable: keep only columns whose (stripped, lowercased) name is
    in `wanted`. Cuts memory dramatically -- FAERS tables have ~20 columns
    but this pipeline only ever uses 2-6 of them."""
    if wanted is None:
        return None
    wanted_lower = {w.lower() for w in wanted}
    return lambda c: str(c).strip().lower() in wanted_lower


def read_ascii_upload(uploaded_file, wanted=None) -> pd.DataFrame:
    usecols = _wanted_filter(wanted)
    last_err = None
    for enc in ENCODINGS_TO_TRY:
        for engine in ("c", "python"):  # C engine is far faster / lighter; python is the fallback
            try:
                uploaded_file.seek(0)  # re-read from the start without copying the bytes
                kwargs = dict(sep="$", encoding=enc, dtype=str, usecols=usecols,
                              on_bad_lines="skip", engine=engine)
                if engine == "c":
                    kwargs["quoting"] = 3  # QUOTE_NONE: FAERS ASCII has no quoting
                df = pd.read_csv(uploaded_file, **kwargs)
                df.columns = [c.strip() for c in df.columns]
                return df
            except UnicodeDecodeError as e:
                last_err = e
                break  # wrong encoding; no point trying the other engine, move to next encoding
            except Exception as e:
                last_err = e
                continue
    raise RuntimeError(f"Could not read {uploaded_file.name}: {last_err}")


def read_csv_upload(uploaded_file, wanted=None) -> pd.DataFrame:
    uploaded_file.seek(0)
    df = pd.read_csv(uploaded_file, dtype=str, usecols=_wanted_filter(wanted))
    df.columns = [c.strip() for c in df.columns]
    return df


def read_any_upload(uploaded_file, is_raw_ascii: bool, wanted=None) -> pd.DataFrame:
    if is_raw_ascii:
        return read_ascii_upload(uploaded_file, wanted)
    return read_csv_upload(uploaded_file, wanted)


# ---------------------------------------------------------------------------
# Helpers: build drug-AE pairs, tagged with subgroup labels from DEMO
# ---------------------------------------------------------------------------

def normalize_age_years(age_series: pd.Series, age_cod_series: pd.Series) -> pd.Series:
    factors = age_cod_series.astype(str).str.strip().str.upper().map(AGE_UNIT_TO_YEARS)
    ages = pd.to_numeric(age_series, errors="coerce")
    return ages * factors


def age_to_band(age_years: pd.Series, edges, labels) -> pd.Series:
    return pd.cut(age_years, bins=edges, labels=labels, right=False, include_lowest=True)


def build_pairs_with_subgroups(drug_df, reac_df, demo_df, selected_roles, age_edges, age_labels):
    drug_case_col = find_column(drug_df, CASE_ID_NAMES)
    reac_case_col = find_column(reac_df, CASE_ID_NAMES)
    demo_case_col = find_column(demo_df, CASE_ID_NAMES)
    drug_name_col = find_column(drug_df, DRUG_NAME_NAMES)
    reaction_col = find_column(reac_df, REACTION_NAMES)
    role_col = find_column(drug_df, ROLE_COD_NAMES)
    age_col = find_column(demo_df, AGE_NAMES)
    age_cod_col = find_column(demo_df, AGE_COD_NAMES)
    sex_col = find_column(demo_df, SEX_NAMES)
    occp_col = find_column(demo_df, OCCP_NAMES)
    event_dt_col = find_column(demo_df, EVENT_DT_NAMES)

    detected = {
        "DRUG case ID": drug_case_col, "REAC case ID": reac_case_col, "DEMO case ID": demo_case_col,
        "Drug name": drug_name_col, "Reaction/PT": reaction_col, "Role code": role_col,
        "Age": age_col, "Age unit": age_cod_col, "Sex": sex_col, "Reporter type": occp_col,
    }

    missing = [
        label for label, col in {
            "Case ID in DRUG file": drug_case_col, "Case ID in REAC file": reac_case_col,
            "Case ID in DEMO file": demo_case_col, "Drug name in DRUG file": drug_name_col,
            "Reaction/PT in REAC file": reaction_col,
        }.items() if col is None
    ]
    if missing:
        return None, detected, missing, None

    d = drug_df[[drug_case_col, drug_name_col] + ([role_col] if role_col else [])].copy()
    rename_map = {drug_case_col: "CASE_ID", drug_name_col: "DRUG"}
    if role_col:
        rename_map[role_col] = "ROLE_COD"
    d = d.rename(columns=rename_map)
    if selected_roles is not None:
        if "ROLE_COD" not in d.columns:
            raise ValueError("A drug-role filter was selected, but no ROLE_COD/role_code/drug_role column was found in the DRUG file.")
        selected_roles = {str(x).strip().upper() for x in selected_roles}
        d["ROLE_COD"] = d["ROLE_COD"].astype(str).str.strip().str.upper()
        d = d[d["ROLE_COD"].isin(selected_roles)]
    d["CASE_ID"] = clean_text(d["CASE_ID"])
    d["DRUG"] = clean_text(d["DRUG"])
    d = d[(d["CASE_ID"] != "") & (d["DRUG"] != "")].drop_duplicates(subset=["CASE_ID", "DRUG"])

    r = reac_df[[reac_case_col, reaction_col]].rename(
        columns={reac_case_col: "CASE_ID", reaction_col: "AE"}
    ).copy()
    r["CASE_ID"] = clean_text(r["CASE_ID"])
    r["AE"] = clean_text(r["AE"])
    r = r[(r["CASE_ID"] != "") & (r["AE"] != "")].drop_duplicates(subset=["CASE_ID", "AE"])

    demo_cols = [demo_case_col]
    if age_col:
        demo_cols.append(age_col)
    if age_cod_col:
        demo_cols.append(age_cod_col)
    if sex_col:
        demo_cols.append(sex_col)
    if occp_col:
        demo_cols.append(occp_col)
    if event_dt_col:
        demo_cols.append(event_dt_col)
    demo = demo_df[demo_cols].copy()
    demo = demo.rename(columns={demo_case_col: "CASE_ID"})
    demo["CASE_ID"] = clean_text(demo["CASE_ID"])
    demo = demo.drop_duplicates(subset=["CASE_ID"])

    if age_col and age_cod_col:
        age_years = normalize_age_years(demo[age_col], demo[age_cod_col])
        demo["AGE_BAND"] = age_to_band(age_years, age_edges, age_labels)
    else:
        demo["AGE_BAND"] = "UNKNOWN"
    demo["AGE_BAND"] = demo["AGE_BAND"].astype(str).replace("nan", "UNKNOWN")

    if sex_col:
        demo["SEX_GROUP"] = clean_text(demo[sex_col]).replace("", "UNKNOWN")
        demo["SEX_GROUP"] = demo["SEX_GROUP"].where(demo["SEX_GROUP"].isin(["M", "F"]), "UNKNOWN")
    else:
        demo["SEX_GROUP"] = "UNKNOWN"

    if occp_col:
        codes = clean_text(demo[occp_col])
        demo["REPORTER_GROUP"] = codes.map(OCCP_LABELS).fillna("OTHER/UNKNOWN")
        demo.loc[codes == "", "REPORTER_GROUP"] = "UNKNOWN"
    else:
        demo["REPORTER_GROUP"] = "UNKNOWN"

    if event_dt_col:
        demo["EVENT_DT"] = parse_faers_date(demo[event_dt_col])
    else:
        demo["EVENT_DT"] = pd.NaT

    # Case-level event date kept SEPARATE from the (very large) pairs table:
    # carrying an 8-byte datetime on every case-drug-event row wastes hundreds of MB.
    event_dt_by_case = demo.set_index("CASE_ID")["EVENT_DT"]

    # Compact categorical subgroup columns (1 byte/row instead of an 8-byte pointer).
    subgroup_cols = ["AGE_BAND", "SEX_GROUP", "REPORTER_GROUP"]
    for col in subgroup_cols:
        cats = sorted(set(demo[col].dropna().unique()) | {"UNKNOWN"})
        demo[col] = pd.Categorical(demo[col], categories=cats)

    # d is already unique on (CASE_ID, DRUG) and r on (CASE_ID, AE), so the join
    # is automatically unique on (CASE_ID, DRUG, AE) -- no drop_duplicates needed
    # (skipping it avoids a full extra copy of the largest table in the app).
    pairs = pd.merge(d[["CASE_ID", "DRUG"]], r, on="CASE_ID", how="inner")
    del d, r
    pairs = pairs.merge(demo[["CASE_ID"] + subgroup_cols], on="CASE_ID", how="left")
    for col in subgroup_cols:
        pairs[col] = pairs[col].fillna("UNKNOWN")

    return pairs, detected, missing, event_dt_by_case


# ---------------------------------------------------------------------------
# Helpers: optional OUTC / INDI / THER / RPSR case-level lookups
# ---------------------------------------------------------------------------

def build_outcome_lookup(outc_df: pd.DataFrame):
    case_col = find_column(outc_df, OUTC_CASE_ID_NAMES)
    code_col = find_column(outc_df, OUTC_CODE_NAMES)
    if case_col is None or code_col is None:
        return None
    df = outc_df[[case_col, code_col]].rename(columns={case_col: "CASE_ID", code_col: "CODE"})
    df["CASE_ID"] = clean_text(df["CASE_ID"])
    df["CODE"] = clean_text(df["CODE"])
    grouped = df.groupby("CASE_ID")["CODE"].apply(set)
    lookup = pd.DataFrame(index=grouped.index)
    lookup["is_serious"] = grouped.apply(lambda codes: bool(codes & SERIOUS_OUTCOME_CODES))
    lookup["is_death"] = grouped.apply(lambda codes: "DE" in codes)
    lookup["is_hospitalization"] = grouped.apply(lambda codes: "HO" in codes)
    lookup["is_life_threatening"] = grouped.apply(lambda codes: "LT" in codes)
    lookup["is_disability"] = grouped.apply(lambda codes: "DS" in codes)
    return lookup.reset_index()  # index (named CASE_ID) becomes a plain column here


def build_indication_lookup(indi_df: pd.DataFrame):
    case_col = find_column(indi_df, INDI_CASE_ID_NAMES)
    pt_col = find_column(indi_df, INDI_PT_NAMES)
    if case_col is None or pt_col is None:
        return None
    df = indi_df[[case_col, pt_col]].rename(columns={case_col: "CASE_ID", pt_col: "INDICATION"})
    df["CASE_ID"] = clean_text(df["CASE_ID"])
    df["INDICATION"] = clean_text(df["INDICATION"])
    df = df[df["INDICATION"] != ""]
    grouped = df.groupby("CASE_ID")["INDICATION"].apply(lambda s: set(s))
    return grouped  # CASE_ID -> set of indication terms


def build_time_to_onset_lookup(ther_df: pd.DataFrame, event_dt_by_case: pd.Series):
    case_col = find_column(ther_df, THER_CASE_ID_NAMES)
    start_col = find_column(ther_df, THER_START_DT_NAMES)
    if case_col is None or start_col is None:
        return None
    df = ther_df[[case_col, start_col]].rename(columns={case_col: "CASE_ID", start_col: "START_DT"})
    df["CASE_ID"] = clean_text(df["CASE_ID"])
    df["START_DT"] = parse_faers_date(df["START_DT"])
    # earliest therapy start per case -- a case-level approximation, not
    # drug-seq-specific, since we don't carry drug_seq through the pipeline
    earliest_start = df.groupby("CASE_ID")["START_DT"].min()
    merged = pd.DataFrame({"CASE_ID": earliest_start.index, "START_DT": earliest_start.values})
    merged = merged.merge(
        event_dt_by_case.rename("EVENT_DT").reset_index().rename(columns={"index": "CASE_ID"}),
        on="CASE_ID", how="left",
    )
    merged["TIME_TO_ONSET_DAYS"] = (merged["EVENT_DT"] - merged["START_DT"]).dt.days
    return merged.set_index("CASE_ID")["TIME_TO_ONSET_DAYS"]


def build_report_source_lookup(rpsr_df: pd.DataFrame):
    case_col = find_column(rpsr_df, RPSR_CASE_ID_NAMES)
    code_col = find_column(rpsr_df, RPSR_COD_NAMES)
    if case_col is None or code_col is None:
        return None
    df = rpsr_df[[case_col, code_col]].rename(columns={case_col: "CASE_ID", code_col: "CODE"})
    df["CASE_ID"] = clean_text(df["CASE_ID"])
    df["CODE"] = clean_text(df["CODE"])
    df = df.drop_duplicates(subset=["CASE_ID"])  # take first source if a case has >1
    df["REPORT_SOURCE_GROUP"] = df["CODE"].map(RPSR_LABELS).fillna("OTHER/UNKNOWN")
    return df.set_index("CASE_ID")["REPORT_SOURCE_GROUP"]


# ---------------------------------------------------------------------------
# Helpers: 2x2 statistics, shared by aggregate and per-stratum computation
# ---------------------------------------------------------------------------

def stats_from_counts(a, b, c, d, min_reports, correct_zero_cells=False):
    """Vectorized 2x2 disproportionality statistics.

    Counts are preserved for display. A Haldane-Anscombe +0.5 correction is
    applied only for estimation when requested. Pearson chi-square is used as
    the primary large-sample statistic; Yates' continuity-corrected value is
    also returned for sensitivity review. Fisher's exact test is not vectorized
    here and should be used only for selected sparse tables during case review.
    """
    a_out, b_out, c_out, d_out = (x.copy() for x in (a, b, c, d))
    a_calc, b_calc, c_calc, d_calc = (x.astype("float64") for x in (a, b, c, d))

    if correct_zero_cells:
        has_zero = (a_calc == 0) | (b_calc == 0) | (c_calc == 0) | (d_calc == 0)
        a_calc = a_calc + 0.5 * has_zero
        b_calc = b_calc + 0.5 * has_zero
        c_calc = c_calc + 0.5 * has_zero
        d_calc = d_calc + 0.5 * has_zero

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        prr = (a_calc / (a_calc + b_calc)) / (c_calc / (c_calc + d_calc))
        ror = (a_calc * d_calc) / (b_calc * c_calc)
        n = a_calc + b_calc + c_calc + d_calc
        denom = (a_calc + b_calc) * (c_calc + d_calc) * (a_calc + c_calc) * (b_calc + d_calc)
        delta = np.abs(a_calc * d_calc - b_calc * c_calc)
        chi_sq = n * delta**2 / denom
        yates_delta = np.maximum(0.0, delta - n / 2.0)
        chi_sq_yates = n * yates_delta**2 / denom
        se_log_ror = np.sqrt(1 / a_calc + 1 / b_calc + 1 / c_calc + 1 / d_calc)
        log_ror = np.log(ror)
        ror_lower = np.exp(log_ror - 1.96 * se_log_ror)
        ror_upper = np.exp(log_ror + 1.96 * se_log_ror)

    # Classic PRR screening rule (Evans-style): >=3 reports, PRR>=2, chi-square>=4.
    # The ROR lower CI is reported separately rather than silently changing that rule.
    is_signal = (a_out >= min_reports) & (prr >= 2) & (chi_sq >= 4)
    return pd.DataFrame({
        "a": a_out, "b": b_out, "c": c_out, "d": d_out, "PRR": prr, "ROR": ror,
        "ROR_CI_lower": ror_lower, "ROR_CI_upper": ror_upper,
        "chi_sq": chi_sq, "chi_sq_yates": chi_sq_yates,
        "log_ROR": log_ror, "se_log_ROR": se_log_ror,
        "ror_ci_excludes_1": ror_lower > 1,
        "is_signal": is_signal,
    })


def compute_aggregate_signals(pairs: pd.DataFrame, min_reports: int) -> pd.DataFrame:
    total = pairs["CASE_ID"].nunique()
    drug_totals = pairs.groupby("DRUG")["CASE_ID"].nunique().rename("drug_total")
    ae_totals = pairs.groupby("AE")["CASE_ID"].nunique().rename("ae_total")
    counts = pairs.groupby(["DRUG", "AE"])["CASE_ID"].nunique().rename("a").reset_index()
    counts = counts.merge(drug_totals, on="DRUG").merge(ae_totals, on="AE")
    counts = counts[counts["a"] >= min_reports]
    b = counts["drug_total"] - counts["a"]
    c = counts["ae_total"] - counts["a"]
    d = total - counts["a"] - b - c
    valid = (counts["a"] >= 0) & (b >= 0) & (c >= 0) & (d >= 0)
    counts, b, c, d = counts[valid], b[valid], c[valid], d[valid]
    stats_df = stats_from_counts(counts["a"], b, c, d, min_reports)
    stats_df["N"] = stats_df[["a", "b", "c", "d"]].sum(axis=1)
    out = pd.concat([counts[["DRUG", "AE"]].reset_index(drop=True), stats_df.reset_index(drop=True)], axis=1)
    return out.replace([np.inf, -np.inf], np.nan).sort_values("ROR", ascending=False, na_position="last")


def compute_stratified_signals(pairs: pd.DataFrame, subgroup_col: str, min_reports_per_stratum: int) -> pd.DataFrame:
    """Same math as compute_aggregate_signals, but the universe for each
    (drug, AE, stratum) row is restricted to that stratum's own cases --
    this is what makes it a genuinely separate estimate, not just a filter
    on the pooled result."""
    rows = []
    for stratum, sub in pairs.groupby(subgroup_col, observed=True):
        agg = compute_aggregate_signals(sub, min_reports_per_stratum)
        agg[subgroup_col] = stratum
        rows.append(agg)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _bh_fdr(p_values: pd.Series) -> pd.Series:
    """Benjamini-Hochberg adjusted p-values; NaNs are retained."""
    p = pd.to_numeric(p_values, errors="coerce")
    out = pd.Series(np.nan, index=p.index, dtype=float)
    valid = p.dropna().sort_values()
    m = len(valid)
    if not m:
        return out
    ranks = np.arange(1, m + 1)
    adj = (valid.values * m / ranks)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out.loc[valid.index] = np.clip(adj, 0, 1)
    return out


def compute_heterogeneity(stratified_df: pd.DataFrame, subgroup_col: str) -> pd.DataFrame:
    """Cochran Q and I² across available stratum-specific log(ROR) estimates.

    This is an exploratory heterogeneity analysis. Because many drug-event pairs
    can be tested, BH-FDR adjusted p-values are supplied alongside raw p-values.
    """
    results = []
    for (drug, ae), sub in stratified_df.groupby(["DRUG", "AE"]):
        sub = sub.dropna(subset=["a", "b", "c", "d"])
        if len(sub) < 2:
            continue
        corrected = stats_from_counts(sub["a"], sub["b"], sub["c"], sub["d"], min_reports=0, correct_zero_cells=True)
        w = 1 / corrected["se_log_ROR"] ** 2
        log_or = corrected["log_ROR"]
        finite = np.isfinite(w) & np.isfinite(log_or) & (w > 0)
        w, log_or = w[finite], log_or[finite]
        if len(w) < 2 or w.sum() <= 0:
            continue
        pooled_log_or = (w * log_or).sum() / w.sum()
        Q = float((w * (log_or - pooled_log_or) ** 2).sum())
        df_q = len(w) - 1
        p_value = float(scipy_stats.chi2.sf(Q, df_q)) if df_q > 0 else np.nan
        i2 = max(0.0, (Q - df_q) / Q) * 100 if Q > 0 else 0.0
        direction_flip = bool((log_or.min() < 0) and (log_or.max() > 0))
        results.append({
            "DRUG": drug, "AE": ae, "n_strata": len(w), "Q_statistic": Q, "df": df_q,
            "heterogeneity_p": p_value, "I2_pct": i2,
            "pooled_ROR_from_strata": float(np.exp(pooled_log_or)),
            "direction_inconsistent": direction_flip,
        })
    out = pd.DataFrame(results)
    if not out.empty:
        out["heterogeneity_p_fdr"] = _bh_fdr(out["heterogeneity_p"])
    return out


def flag_masking(aggregate_df: pd.DataFrame, stratified_df: pd.DataFrame,
                 heterogeneity_df: pd.DataFrame, subgroup_col: str,
                 p_threshold: float = 0.05, use_fdr: bool = True) -> pd.DataFrame:
    """Flag exploratory masking/confounding patterns with explicit subgroup evidence."""
    merged = aggregate_df.merge(heterogeneity_df, on=["DRUG", "AE"], how="inner")
    p_col = "heterogeneity_p_fdr" if use_fdr and "heterogeneity_p_fdr" in merged else "heterogeneity_p"
    merged["significant_heterogeneity"] = merged[p_col] < p_threshold

    subgroup_summary = stratified_df.groupby(["DRUG", "AE"]).agg(
        any_subgroup_signal=("is_signal", "any"),
        n_subgroup_signals=("is_signal", "sum"),
        max_subgroup_ROR=("ROR", "max"),
    ).reset_index()
    merged = merged.merge(subgroup_summary, on=["DRUG", "AE"], how="left")
    merged["masked_signal"] = (~merged["is_signal"]) & merged["any_subgroup_signal"].fillna(False) & merged["significant_heterogeneity"]
    merged["confounded_signal"] = merged["is_signal"] & merged["significant_heterogeneity"] & merged["direction_inconsistent"]
    return merged


# ---------------------------------------------------------------------------
# Sidebar: upload + options
# ---------------------------------------------------------------------------

st.sidebar.header("1. Upload data")
source_is_ascii = st.sidebar.radio("File format", ["CSV (already converted)", "Raw FAERS ASCII (.txt)"]) == "Raw FAERS ASCII (.txt)"
ftype = ["txt"] if source_is_ascii else ["csv"]
drug_files = st.sidebar.file_uploader("DRUG file(s)", type=ftype, accept_multiple_files=True)
reac_files = st.sidebar.file_uploader("REAC file(s)", type=ftype, accept_multiple_files=True)
demo_files = st.sidebar.file_uploader("DEMO file(s) (required: age, sex, reporter type)", type=ftype, accept_multiple_files=True)

st.sidebar.header("1b. Optional data (unlock extra features)")
outc_files = st.sidebar.file_uploader("OUTC file(s) -- enables seriousness weighting", type=ftype, accept_multiple_files=True)
indi_files = st.sidebar.file_uploader("INDI file(s) -- enables confounding-by-indication check", type=ftype, accept_multiple_files=True)
ther_files = st.sidebar.file_uploader("THER file(s) -- enables time-to-onset analysis", type=ftype, accept_multiple_files=True)
rpsr_files = st.sidebar.file_uploader("RPSR file(s) -- enables report-source stratification", type=ftype, accept_multiple_files=True)

st.sidebar.header("2. Options")
role_mode = st.sidebar.selectbox(
    "Drug role for analysis",
    ["Primary + Secondary Suspect (PS + SS)", "Primary Suspect (PS) only", "All reported drug roles", "Custom selection"],
    index=0,
    help="Controls which FAERS ROLE_COD values enter every downstream analysis. Changing this changes a, b, c, d and therefore PRR, ROR, EBGM, IC and temporal results.",
)
ROLE_MODE_MAP = {
    "Primary + Secondary Suspect (PS + SS)": ["PS", "SS"],
    "Primary Suspect (PS) only": ["PS"],
    "All reported drug roles": None,
}
if role_mode == "Custom selection":
    selected_roles = st.sidebar.multiselect(
        "Select ROLE_COD values",
        options=["PS", "SS", "C", "I"],
        default=["PS", "SS"],
        help="PS=Primary Suspect, SS=Secondary Suspect, C=Concomitant, I=Interacting.",
    )
    if not selected_roles:
        st.sidebar.error("Select at least one drug role.")
        st.stop()
else:
    selected_roles = ROLE_MODE_MAP[role_mode]

role_report_label = role_mode if selected_roles is None else f"{role_mode} ({', '.join(selected_roles)})"

st.sidebar.caption(f"Active analysis population: {role_report_label}")

def export_csv_with_context(df: pd.DataFrame) -> bytes:
    """Add reproducibility metadata to exported result tables."""
    out = df.copy()
    out.insert(0, "ANALYSIS_DRUG_ROLE", role_report_label)
    return out.to_csv(index=False).encode("utf-8")
min_reports = st.sidebar.slider("Minimum reports for aggregate signal (a)", 1, 50, 3, 1)
min_reports_stratum = st.sidebar.slider("Minimum reports per stratum", 1, 20, 3, 1)
hetero_alpha = st.sidebar.slider("Heterogeneity significance level (p <)", 0.01, 0.20, 0.05, 0.01)

st.sidebar.header("3. Age bands")
# Standard project age bands. The final interval is open-ended.
age_edges = [0.0, 18.0, 65.0, float("inf")]
age_labels = ["0–<18", "18–<65", "≥65"]
st.sidebar.caption("Age groups: 0–<18, 18–<65, ≥65, UNKNOWN")

if not drug_files or not reac_files or not demo_files:
    st.info("Upload DRUG, REAC, and DEMO files in the sidebar to begin. DEMO is required here for age/sex/reporter-type stratification.")
    st.stop()

# ---------------------------------------------------------------------------
# Load + build (cached in session state so widget clicks don't re-parse files)
# ---------------------------------------------------------------------------

WANTED_COLS = {
    "drug": CASE_ID_NAMES + DRUG_NAME_NAMES + ROLE_COD_NAMES,
    "reac": CASE_ID_NAMES + REACTION_NAMES,
    "demo": CASE_ID_NAMES + AGE_NAMES + AGE_COD_NAMES + SEX_NAMES + OCCP_NAMES + EVENT_DT_NAMES,
    "outc": OUTC_CASE_ID_NAMES + OUTC_CODE_NAMES,
    "indi": INDI_CASE_ID_NAMES + INDI_PT_NAMES,
    "ther": THER_CASE_ID_NAMES + THER_START_DT_NAMES,
    "rpsr": RPSR_CASE_ID_NAMES + RPSR_COD_NAMES,
}


def read_many(files, is_raw_ascii, wanted):
    if not files:
        return None
    frames = [read_any_upload(f, is_raw_ascii, wanted) for f in files]
    return frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True)


def load_and_build(drug_files, reac_files, demo_files, outc_files, indi_files, ther_files, rpsr_files,
                   is_raw_ascii, selected_roles, age_edges, age_labels):
    """Reads only the needed columns of each table, builds the pairs table, and
    frees each raw table as soon as it's been used. Optional tables become small
    case-level lookups rather than extra columns on the multi-million-row pairs table."""
    drug_df = read_many(drug_files, is_raw_ascii, WANTED_COLS["drug"])
    reac_df = read_many(reac_files, is_raw_ascii, WANTED_COLS["reac"])
    demo_df = read_many(demo_files, is_raw_ascii, WANTED_COLS["demo"])

    pairs, detected, missing, event_dt_by_case = build_pairs_with_subgroups(
        drug_df, reac_df, demo_df, selected_roles, age_edges, age_labels
    )
    del drug_df, reac_df, demo_df
    gc.collect()

    out = {
        "pairs": pairs, "detected": detected, "missing": missing,
        "outcome_lookup": None, "indication_lookup": None, "time_to_onset_lookup": None,
        "event_dt_by_case": event_dt_by_case,
    }
    if missing:
        return out

    outc_df = read_many(outc_files, is_raw_ascii, WANTED_COLS["outc"])
    if outc_df is not None:
        lookup = build_outcome_lookup(outc_df)
        del outc_df
        if lookup is not None:
            out["outcome_lookup"] = lookup
            by_case = lookup.set_index("CASE_ID")
            # Only the two flags Section 5 actually reports on are attached to pairs.
            # .eq(True) turns "missing" into False without dtype-downcast warnings.
            for col in ["is_serious", "is_death"]:
                pairs[col] = pairs["CASE_ID"].map(by_case[col]).eq(True)

    rpsr_df = read_many(rpsr_files, is_raw_ascii, WANTED_COLS["rpsr"])
    rs_lookup = build_report_source_lookup(rpsr_df) if rpsr_df is not None else None
    del rpsr_df
    if rs_lookup is not None:
        pairs["REPORT_SOURCE_GROUP"] = pd.Categorical(pairs["CASE_ID"].map(rs_lookup).fillna("UNKNOWN"))
    else:
        pairs["REPORT_SOURCE_GROUP"] = pd.Categorical.from_codes(
            np.zeros(len(pairs), dtype="int8"), categories=["UNKNOWN"]
        )

    indi_df = read_many(indi_files, is_raw_ascii, WANTED_COLS["indi"])
    if indi_df is not None:
        out["indication_lookup"] = build_indication_lookup(indi_df)
        del indi_df

    ther_df = read_many(ther_files, is_raw_ascii, WANTED_COLS["ther"])
    if ther_df is not None:
        # kept as a case-level Series; looked up only for the pair being inspected
        out["time_to_onset_lookup"] = build_time_to_onset_lookup(ther_df, event_dt_by_case)
        del ther_df

    gc.collect()
    return out


def session_cached(name, key, compute_fn):
    """Keeps only the latest result per `name`; recomputes only when `key` changes.
    Streamlit re-runs this whole script on every widget click -- without this,
    slow steps (and their memory allocations) repeat on every click."""
    store = st.session_state.setdefault("_results_cache", {})
    entry = store.get(name)
    if entry is None or entry[0] != key:
        store.pop(name, None)  # release the old result before computing the new one
        store[name] = (key, compute_fn())
    return store[name][1]


upload_groups = (drug_files, reac_files, demo_files, outc_files, indi_files, ther_files, rpsr_files)
build_signature = (
    tuple(tuple((f.name, f.size) for f in (grp or [])) for grp in upload_groups)
    + (source_is_ascii, tuple(selected_roles) if selected_roles is not None else ("ALL",), tuple(age_edges))
)

if st.session_state.get("build_signature") != build_signature:
    st.session_state.pop("build", None)   # free the previous dataset BEFORE loading a new one
    st.session_state.pop("_results_cache", None)
    gc.collect()
    try:
        with st.spinner("Reading files and building drug-event pairs (large files can take a while)..."):
            st.session_state["build"] = load_and_build(
                drug_files, reac_files, demo_files, outc_files, indi_files, ther_files, rpsr_files,
                source_is_ascii, selected_roles, age_edges, age_labels,
            )
        st.session_state["build_signature"] = build_signature
    except MemoryError:
        st.error(
            "Ran out of memory while building the dataset. Try: fewer quarters at once, "
            "using PS-only or PS+SS drug roles, dropping optional files (INDI/THER), "
            "closing other programs, and making sure you're on 64-bit Python."
        )
        st.stop()
    except Exception as e:
        st.error(f"Error reading files: {e}")
        st.stop()

build = st.session_state["build"]
pairs = build["pairs"]
detected_cols = build["detected"]
missing_cols = build["missing"]
outcome_lookup = build["outcome_lookup"]
indication_lookup = build["indication_lookup"]
time_to_onset_lookup = build["time_to_onset_lookup"]
event_dt_by_case = build.get("event_dt_by_case")

st.header("1. Column Detection")
cols = st.columns(5)
items = list(detected_cols.items())
for i, (label, value) in enumerate(items):
    with cols[i % 5]:
        st.write(label)
        st.code(str(value))

if missing_cols:
    st.error("Missing required columns:")
    for item in missing_cols:
        st.write(f"- {item}")
    st.stop()

st.success(f"Built {len(pairs):,} case-drug-event records with subgroup labels attached.")

st.header("2. Subgroup Coverage")
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.write("Age bands")
    st.write(pairs["AGE_BAND"].value_counts().loc[lambda s: s > 0])
with c2:
    st.write("Sex")
    st.write(pairs["SEX_GROUP"].value_counts().loc[lambda s: s > 0])
with c3:
    st.write("Reporter type")
    st.write(pairs["REPORTER_GROUP"].value_counts().loc[lambda s: s > 0])
with c4:
    st.write("Report source")
    st.write(pairs["REPORT_SOURCE_GROUP"].value_counts().loc[lambda s: s > 0])

# ---------------------------------------------------------------------------
# Aggregate signals
# ---------------------------------------------------------------------------

st.header("3. Aggregate (Pooled) Signals")
with st.spinner("Computing pooled PRR/ROR/chi-square..."):
    aggregate_df = session_cached(
        "aggregate", (build_signature, min_reports),
        lambda: compute_aggregate_signals(pairs, min_reports),
    )
c1, c2 = st.columns(2)
c1.metric("Pairs analyzed", f"{len(aggregate_df):,}")
c2.metric("Pooled signals", f"{int(aggregate_df['is_signal'].sum()):,}")
st.dataframe(
    aggregate_df[["DRUG", "AE", "a", "b", "c", "d", "N", "PRR", "ROR", "ROR_CI_lower", "ROR_CI_upper", "chi_sq", "is_signal"]],
    use_container_width=True, height=300,
)

# ---------------------------------------------------------------------------
# Stratified analysis + Simpson's paradox flags
# ---------------------------------------------------------------------------

st.header("4. Subgroup-Stratified Signal Detection")
dimension = st.selectbox("Stratify by", ["AGE_BAND", "SEX_GROUP", "REPORTER_GROUP", "REPORT_SOURCE_GROUP"], format_func=lambda x: {
    "AGE_BAND": "Age band", "SEX_GROUP": "Sex", "REPORTER_GROUP": "Reporter type",
    "REPORT_SOURCE_GROUP": "Report source",
}[x])

with st.spinner(f"Computing signals within each {dimension} stratum..."):
    stratified_df = session_cached(
        "stratified", (build_signature, dimension, min_reports_stratum),
        lambda: compute_stratified_signals(pairs, dimension, min_reports_stratum),
    )

if stratified_df.empty:
    st.info("Not enough data to compute stratified signals for this dimension.")
else:
    with st.spinner("Testing cross-stratum consistency (Cochran's Q)..."):
        heterogeneity_df = session_cached(
            "heterogeneity", (build_signature, dimension, min_reports_stratum),
            lambda: compute_heterogeneity(stratified_df, dimension),
        )

    if heterogeneity_df.empty:
        st.info("No drug-AE pairs had data in 2+ strata for a heterogeneity test.")
    else:
        flagged = flag_masking(aggregate_df, stratified_df, heterogeneity_df, dimension, p_threshold=hetero_alpha, use_fdr=True)

        st.subheader("Simpson's-paradox flags")
        st.write(
            f"""
            - **Masked signals**: pooled result shows no signal, but stratum estimates are
              significantly inconsistent ({dimension} appears to matter) -- worth checking
              individual strata below, since one of them may hide a real elevation.
            - **Confounded signals**: pooled result IS flagged as a signal, but stratum
              estimates disagree significantly *and* flip direction across strata -- the
              pooled signal may not represent a uniform effect.
            """
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("Pairs tested", f"{len(flagged):,}")
        c2.metric("Masked signals", f"{int(flagged['masked_signal'].sum()):,}")
        c3.metric("Confounded signals", f"{int(flagged['confounded_signal'].sum()):,}")

        flag_choice = st.radio("Show", ["All tested pairs", "Masked only", "Confounded only"], horizontal=True)
        if flag_choice == "Masked only":
            display = flagged[flagged["masked_signal"]]
        elif flag_choice == "Confounded only":
            display = flagged[flagged["confounded_signal"]]
        else:
            display = flagged

        st.dataframe(
            display[["DRUG", "AE", "is_signal", "PRR", "ROR", "n_strata", "Q_statistic",
                     "heterogeneity_p", "heterogeneity_p_fdr", "I2_pct", "any_subgroup_signal", "n_subgroup_signals", "direction_inconsistent", "masked_signal", "confounded_signal"]],
            use_container_width=True, height=350,
        )

        st.subheader("Drill down: age-first Drug–AE selection")

        # Always build age-stratified results for the drill-down so that the
        # Drug–AE selector can be filtered by age independently of the
        # heterogeneity dimension selected above.
        age_drill_df = session_cached(
            "age_drilldown",
            (build_signature, min_reports_stratum),
            lambda: compute_stratified_signals(pairs, "AGE_BAND", min_reports_stratum),
        )

        age_order = ["ALL / Pooled", "0–<18", "18–<65", "≥65", "UNKNOWN"]
        available_age = set(age_drill_df["AGE_BAND"].astype(str).unique()) if not age_drill_df.empty else set()
        age_options = ["ALL / Pooled"] + [x for x in age_order[1:] if x in available_age]

        selected_age = st.selectbox(
            "1. Select age group",
            age_options,
            key="drill_age_first",
            help="The Drug–AE list below is restricted to pairs available in the selected age group.",
        )

        if selected_age == "ALL / Pooled":
            source_for_pairs = aggregate_df.copy()
            count_col = "a"
        else:
            source_for_pairs = age_drill_df[age_drill_df["AGE_BAND"].astype(str) == selected_age].copy()
            count_col = "a"

        if not source_for_pairs.empty:
            source_for_pairs = source_for_pairs.sort_values([count_col, "DRUG", "AE"], ascending=[False, True, True])
            pair_options = list(zip(source_for_pairs["DRUG"], source_for_pairs["AE"], source_for_pairs[count_col]))

            chosen3 = st.selectbox(
                "2. Select Drug–AE pair",
                pair_options,
                format_func=lambda p: f"{p[0]} / {p[1]} (n={int(p[2])})",
                key="drill_pair_age_first",
            )
            chosen = (chosen3[0], chosen3[1])

            agg_row = aggregate_df[(aggregate_df["DRUG"] == chosen[0]) & (aggregate_df["AE"] == chosen[1])]
            pair_age = age_drill_df[(age_drill_df["DRUG"] == chosen[0]) & (age_drill_df["AE"] == chosen[1])].copy()

            st.write("Pooled estimate:")
            st.dataframe(
                agg_row[["DRUG", "AE", "a", "b", "c", "d", "N", "PRR", "ROR", "ROR_CI_lower", "ROR_CI_upper", "chi_sq", "is_signal"]],
                use_container_width=True,
                hide_index=True,
            )

            st.write("Age-stratified estimates:")
            if not pair_age.empty:
                st.dataframe(
                    pair_age[["AGE_BAND", "a", "b", "c", "d", "N", "PRR", "ROR", "ROR_CI_lower", "ROR_CI_upper", "chi_sq", "is_signal"]],
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("This pair does not meet the current minimum-report threshold in any age stratum.")

            # Show the 2x2 table for the age group chosen first. If pooled was
            # selected, show the pooled 2x2 table.
            if selected_age == "ALL / Pooled":
                selected_row = agg_row.iloc[0] if not agg_row.empty else None
                table_label = "ALL / Pooled"
            else:
                sr = pair_age[pair_age["AGE_BAND"].astype(str) == selected_age]
                selected_row = sr.iloc[0] if not sr.empty else None
                table_label = selected_age

            if selected_row is not None:
                a2, b2, c2, d2 = [int(selected_row[x]) for x in ["a", "b", "c", "d"]]
                contingency = pd.DataFrame(
                    {
                        f"Selected AE: {chosen[1]}": [a2, c2],
                        "Other AEs": [b2, d2],
                    },
                    index=[f"Selected drug: {chosen[0]}", "Other drugs"],
                )
                contingency["Total"] = contingency.sum(axis=1)
                contingency.loc["Total"] = contingency.sum(axis=0)
                st.subheader(f"2 × 2 Contingency Table — {table_label}")
                st.dataframe(contingency, use_container_width=True)
                st.caption(
                    f"N = {a2+b2+c2+d2:,}. a = selected drug + selected AE; "
                    "b = selected drug + other AEs; c = other drugs + selected AE; "
                    "d = other drugs + other AEs."
                )

            if not pair_age.empty:
                st.bar_chart(pair_age.set_index("AGE_BAND")["ROR"])
                st.caption(
                    "The Drug–AE selector is filtered by the age group selected above. "
                    "The table and chart still show all available age strata for the chosen pair, "
                    "allowing direct comparison with the pooled estimate."
                )
        else:
            st.info("No Drug–AE pairs meet the current minimum-report threshold for this age group.")

st.header("5. Seriousness-Weighted Signals (OUTC)")
if outcome_lookup is None:
    st.info("Upload OUTC file(s) in the sidebar to enable seriousness weighting.")
else:
    serious_by_pair = (
        pairs.groupby(["DRUG", "AE"])[["is_serious", "is_death"]]
        .sum()
        .reset_index()
    )
    serious_signals = aggregate_df.merge(serious_by_pair, on=["DRUG", "AE"], how="left")
    serious_signals["pct_serious"] = (serious_signals["is_serious"] / serious_signals["a"] * 100).round(1)
    serious_signals["pct_death"] = (serious_signals["is_death"] / serious_signals["a"] * 100).round(1)
    # simple heuristic ranking, not a validated clinical score -- surfaces
    # signals that are both statistically elevated AND skew toward serious outcomes
    serious_signals["priority_score"] = (serious_signals["PRR"].fillna(0) * (1 + serious_signals["pct_serious"] / 100)).round(2)

    st.write(
        "Ranks flagged signals by a combination of statistical elevation (PRR) and outcome "
        "severity. `priority_score` is a simple heuristic (PRR x (1 + pct_serious/100)), not a "
        "validated clinical severity index -- use it to prioritize review, not as a final verdict."
    )
    only_serious_signals = st.checkbox("Show only flagged signals (is_signal = True)", value=True, key="serious_filter")
    display_serious = serious_signals[serious_signals["is_signal"]] if only_serious_signals else serious_signals
    st.dataframe(
        display_serious.sort_values("priority_score", ascending=False)[
            ["DRUG", "AE", "a", "PRR", "ROR", "pct_serious", "pct_death", "priority_score", "is_signal"]
        ],
        use_container_width=True, height=350,
    )
    st.download_button(
        "Download seriousness-weighted table (CSV)",
        export_csv_with_context(serious_signals),
        file_name="seriousness_weighted_signals.csv", mime="text/csv",
    )

st.header("6. Confounding-by-Indication Check (INDI)")
if indication_lookup is None:
    st.info("Upload INDI file(s) in the sidebar to enable this check.")
else:
    st.write(
        "For each flagged signal, checks what fraction of its reporting cases listed an "
        "indication (reason the drug was prescribed) whose term overlaps the adverse event "
        "term. High overlap suggests the 'event' may partly reflect the underlying condition "
        "being treated, not a drug effect. This is a plain substring heuristic (case-level, "
        "not drug-seq-precise) -- treat flags as worth a closer look, not confirmed confounding."
    )
    max_indi_pairs = st.slider("Max flagged pairs to check (limits compute)", 1, 200, 30, 1)
    signals_to_check = aggregate_df[aggregate_df["is_signal"]].head(max_indi_pairs)

    indi_results = []
    for _, row in signals_to_check.iterrows():
        pair_cases = pairs[(pairs["DRUG"] == row["DRUG"]) & (pairs["AE"] == row["AE"])]["CASE_ID"].unique()
        if len(pair_cases) == 0:
            continue
        event_term = row["AE"].lower().strip()
        overlap_count = 0
        checked = 0
        for case_id in pair_cases:
            indications = indication_lookup.get(case_id)
            if indications is None:
                continue
            checked += 1
            indications_text = " ".join(indications).lower()
            if event_term in indications_text or event_term.rstrip("s") in indications_text:
                overlap_count += 1
        pct_overlap = (overlap_count / checked * 100) if checked else np.nan
        indi_results.append({
            "DRUG": row["DRUG"], "AE": row["AE"], "cases_with_indication_data": checked,
            "pct_indication_overlap": round(pct_overlap, 1) if checked else np.nan,
            "possible_indication_confound": (pct_overlap >= 30) if checked else False,
        })

    indi_df_result = pd.DataFrame(indi_results)
    if indi_df_result.empty:
        st.info("No indication data matched the currently flagged signals.")
    else:
        n_confound = int(indi_df_result["possible_indication_confound"].sum())
        st.metric("Pairs flagged as possible indication confounding", f"{n_confound:,}")
        show_confound_only = st.checkbox("Show only possible confounds", value=True)
        display_indi = indi_df_result[indi_df_result["possible_indication_confound"]] if show_confound_only else indi_df_result
        st.dataframe(display_indi.sort_values("pct_indication_overlap", ascending=False), use_container_width=True, height=300)

st.header("7. Time-to-Onset Analysis (THER)")
if time_to_onset_lookup is None:
    st.info("Upload THER file(s) (and ensure DEMO has event_dt) to enable this analysis.")
else:
    pair_options_onset = list(zip(aggregate_df["DRUG"], aggregate_df["AE"]))[:200]
    if not pair_options_onset:
        st.info("No drug-AE pairs available.")
    else:
        chosen_onset = st.selectbox(
            "Choose a drug-AE pair", pair_options_onset, format_func=lambda p: f"{p[0]} / {p[1]}", key="onset_pair"
        )
        # Look up onset only for THIS pair's cases (a small slice), rather than
        # attaching an onset column to the whole multi-million-row pairs table.
        pair_case_ids = pairs.loc[
            (pairs["DRUG"] == chosen_onset[0]) & (pairs["AE"] == chosen_onset[1]), "CASE_ID"
        ]
        days = pair_case_ids.map(time_to_onset_lookup).dropna()
        if days.empty:
            st.info("No time-to-onset data available for this pair (needs both a parseable event date and therapy start date).")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Cases with onset data", f"{len(days):,}")
            c2.metric("Median days to onset", f"{days.median():.0f}")
            c3.metric("% within 7 days", f"{(days <= 7).mean() * 100:.0f}%")

            bins = [-np.inf, 0, 1, 7, 30, 90, np.inf]
            bin_labels = ["Before start (data issue)", "Day 0-1", "2-7 days", "8-30 days", "31-90 days", ">90 days"]
            binned = pd.cut(days, bins=bins, labels=bin_labels)
            st.bar_chart(binned.value_counts().reindex(bin_labels))
            st.caption(
                "Time-to-onset uses each case's EARLIEST drug start date across all drugs on "
                "that case, not specifically this drug's start date -- an approximation, since "
                "drug-seq-level linkage wasn't carried through the pipeline. Very short onsets "
                "are more suggestive of causality but this is contextual evidence, not an "
                "automated signal criterion."
            )

st.header("8. Analysis Report")

# A self-contained HTML report is intentionally used: it is portable, auditable,
# opens in any browser, and does not add a fragile PDF-generation dependency.
def _df_html(df, cols, n=100):
    if df is None or df.empty:
        return "<p>No rows available.</p>"
    use = [c for c in cols if c in df.columns]
    return df[use].head(n).to_html(index=False, border=0, classes="data", float_format=lambda x: f"{x:.4g}")


def build_html_report():
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    n_cases = int(pairs["CASE_ID"].nunique())
    n_pairs = int(len(aggregate_df))
    n_signals = int(aggregate_df["is_signal"].sum())
    parts = [f"""<!doctype html><html><head><meta charset='utf-8'><title>FAERS Signal Detection Report</title>
<style>body{{font-family:Arial,sans-serif;margin:36px;line-height:1.45;color:#222}}h1,h2{{color:#17365d}}table{{border-collapse:collapse;width:100%;font-size:12px}}th,td{{border:1px solid #ccc;padding:5px;text-align:left}}th{{background:#eef3f8}}.note{{background:#fff8dc;padding:10px;border-left:4px solid #c49a00}}</style></head><body>
<h1>FAERS/AEMS Disproportionality & Stratified Signal Detection Report</h1>
<p><b>Generated:</b> {now}</p>
<h2>1. Scope and data</h2><p>Unique report IDs analysed: {n_cases:,}. Drug-event pairs meeting the aggregate minimum count: {n_pairs:,}. Screening-positive pairs: {n_signals:,}.</p>
<p>Drug-role population: {html.escape(role_report_label)}. Aggregate minimum reports: {min_reports}. Per-stratum minimum reports: {min_reports_stratum}. Heterogeneity alpha: {hetero_alpha:.2f}.</p>
<h2>2. Methods</h2><p>Case-level 2x2 tables were used to calculate PRR, ROR with 95% Wald confidence interval, and Pearson chi-square. The screening rule is a≥minimum reports, PRR≥2, and chi-square≥4. ROR CI exclusion of 1 is reported as supporting evidence but is not silently added to the PRR rule.</p>
<p>Subgroup estimates are calculated independently within the selected stratum. Cross-stratum heterogeneity uses inverse-variance Cochran Q on log(ROR), I², and Benjamini-Hochberg FDR-adjusted p-values. Zero cells receive a 0.5 correction only for log-OR heterogeneity estimation.</p>
<div class='note'><b>Interpretation:</b> Disproportionality is hypothesis-generating and does not establish incidence, risk, or causality. Signals require clinical review, duplicate/version control, product/ingredient normalization, confounding assessment, and review of individual cases.</div>
<h2>3. Aggregate results</h2>""",
        _df_html(aggregate_df.sort_values(["is_signal","ROR"], ascending=[False,False]), ["DRUG","AE","a","b","c","d","N","PRR","ROR","ROR_CI_lower","ROR_CI_upper","chi_sq","ror_ci_excludes_1","is_signal"]),
    ]
    if 'flagged' in globals() and isinstance(flagged, pd.DataFrame):
        parts += ["<h2>4. Stratified heterogeneity / masking review</h2>", _df_html(flagged.sort_values(["masked_signal","confounded_signal","heterogeneity_p_fdr"], ascending=[False,False,True]), ["DRUG","AE","is_signal","PRR","ROR","n_strata","Q_statistic","I2_pct","heterogeneity_p","heterogeneity_p_fdr","any_subgroup_signal","n_subgroup_signals","direction_inconsistent","masked_signal","confounded_signal"])]
    parts += ["<h2>5. Data-quality and limitations</h2><ul><li>Verify FAERS case-version deduplication before regulatory or publication use.</li><li>Normalize products to active ingredients; verbatim product names can split the same exposure across aliases.</li><li>Spontaneous reports are subject to under-reporting, stimulated reporting, missingness, notoriety bias, channeling, and confounding by indication.</li><li>Current THER time-to-onset is case-level and uses the earliest therapy start across drugs; it is not drug_seq-specific and should not be treated as a validated causality metric.</li><li>INDI overlap is lexical and exploratory, not a validated confounding adjustment.</li></ul><h2>6. Conclusion</h2><p>Use screening-positive and subgroup-heterogeneous pairs to prioritize manual pharmacovigilance review. Do not interpret the output as proof that a medicine caused an event or as an estimate of event incidence.</p></body></html>"]
    return "".join(parts)

report_html = build_html_report()
st.download_button("Download complete analysis report (HTML)", report_html.encode("utf-8"), file_name="faers_signal_analysis_report.html", mime="text/html")
st.download_button("Download aggregate results (CSV)", export_csv_with_context(aggregate_df), file_name="aggregate_signal_results.csv", mime="text/csv")
if 'flagged' in globals() and isinstance(flagged, pd.DataFrame):
    st.download_button("Download stratified/heterogeneity results (CSV)", export_csv_with_context(flagged), file_name="stratified_heterogeneity_results.csv", mime="text/csv")

st.header("9. Methodology Note")
st.markdown(
    """
    Stratified estimates use the **same PRR/ROR/chi-square formulas** as the pooled analysis,
    but restrict the 2x2 table's universe to cases within that stratum only -- not just a
    filtered view of the pooled table.

    **Heterogeneity test**: Cochran's Q on inverse-variance-weighted log(ROR) across strata
    (Haldane-Anscombe +0.5 correction applied to any stratum with a zero cell, standard
    practice to keep log-OR variance finite). Q ~ chi-square(k-1) under the null hypothesis
    that all strata share a common true effect. A small p-value means the strata genuinely
    disagree -- not just that they have different sample sizes.

    **Caveats**: age/sex/reporter-type stratification reduces the sample size within each
    stratum, so stratified signals are inherently noisier and more prone to small-sample
    instability than the pooled estimate -- treat flags here as hypotheses to investigate
    further, not confirmed findings. Age normalization assumes standard FAERS age_cod values
    (YR, MON, WK, DY, DEC, HR); records with missing or unrecognized units fall into an
    'UNKNOWN' age band rather than being silently dropped.

    **OUTC (seriousness)**: FAERS only records an OUTC row for serious cases at all -- a case
    with no OUTC entry is presumed non-serious, not missing data. `priority_score` is a simple,
    unvalidated heuristic combining PRR with % serious outcomes; it's a triage aid, not a
    clinical severity index.

    **INDI (confounding by indication)**: matching is a plain substring check between the event
    term and each case's stated indication(s), at the case level (not linked to a specific
    drug_seq). This will miss synonym matches (e.g. "heart attack" vs "myocardial infarction")
    and can false-positive on coincidental word overlap -- use it to prioritize manual review,
    not as a confirmed confounding determination.

    **THER (time-to-onset)**: uses each case's earliest drug start date across ALL drugs listed
    on that case, not specifically the drug in the pair being examined, since drug-seq-level
    linkage between THER and DRUG wasn't carried through this pipeline. On multi-drug cases this
    is an approximation of true time-to-onset for the specific drug.

    **RPSR (report source)**: an additional stratification dimension, useful for checking
    whether a signal holds up consistently across consumer, health-professional, literature,
    and study-derived reports, or is being driven disproportionately by one source type.
    """
)


# ---------------------------------------------------------------------------
# V2: data-quality dashboard + multidimensional signal evidence explorer
# ---------------------------------------------------------------------------
st.header("10. V2 Data Quality Dashboard")

@st.cache_data(show_spinner=False)
def case_level_coverage(_pairs: pd.DataFrame):
    base = _pairs[["CASE_ID", "AGE_BAND", "SEX_GROUP", "REPORTER_GROUP", "REPORT_SOURCE_GROUP"]].drop_duplicates("CASE_ID")
    n = len(base)
    rows = []
    for col, label in [("AGE_BAND","Age"),("SEX_GROUP","Sex"),("REPORTER_GROUP","Reporter type"),("REPORT_SOURCE_GROUP","Report source")]:
        vals = base[col].astype(str).str.upper()
        known = ~(vals.isin(["UNKNOWN", "OTHER/UNKNOWN", "NAN", ""]))
        rows.append({"Variable": label, "Known cases": int(known.sum()), "Total cases": n,
                     "Completeness %": round(100 * known.mean(), 1) if n else np.nan})
    return pd.DataFrame(rows)

quality_df = case_level_coverage(pairs)
q1, q2, q3 = st.columns(3)
q1.metric("Unique cases", f"{pairs['CASE_ID'].nunique():,}")
q2.metric("Unique drugs", f"{pairs['DRUG'].nunique():,}")
q3.metric("Unique events", f"{pairs['AE'].nunique():,}")
st.dataframe(quality_df, use_container_width=True, hide_index=True)
low_quality = quality_df[quality_df["Completeness %"] < 70]
if not low_quality.empty:
    st.warning("Low completeness (<70%) detected for: " + ", ".join(low_quality["Variable"].tolist()) +
               ". Interpret subgroup comparisons for these variables cautiously.")

st.subheader("Signal Evidence Card")
signal_pool = aggregate_df[aggregate_df["is_signal"]].copy()
if signal_pool.empty:
    signal_pool = aggregate_df.copy()
if signal_pool.empty:
    st.info("No drug-event pairs are available for the V2 evidence explorer.")
else:
    max_pairs_v2 = min(1000, len(signal_pool))
    options_v2 = list(zip(signal_pool.head(max_pairs_v2)["DRUG"], signal_pool.head(max_pairs_v2)["AE"]))
    chosen_v2 = st.selectbox("Select drug-event pair", options_v2,
                             format_func=lambda p: f"{p[0]} / {p[1]}", key="v2_pair")
    base_row = aggregate_df[(aggregate_df["DRUG"] == chosen_v2[0]) & (aggregate_df["AE"] == chosen_v2[1])].iloc[0]
    ec1, ec2, ec3, ec4, ec5 = st.columns(5)
    ec1.metric("Reports (a)", f"{int(base_row['a']):,}")
    ec2.metric("PRR", f"{base_row['PRR']:.2f}" if pd.notna(base_row['PRR']) else "NA")
    ec3.metric("ROR", f"{base_row['ROR']:.2f}" if pd.notna(base_row['ROR']) else "NA")
    ec4.metric("ROR 95% CI", f"{base_row['ROR_CI_lower']:.2f}–{base_row['ROR_CI_upper']:.2f}" if pd.notna(base_row['ROR_CI_lower']) else "NA")
    ec5.metric("Screening result", "Positive" if bool(base_row["is_signal"]) else "Not positive")

    pair_cases_v2 = pairs[(pairs["DRUG"] == chosen_v2[0]) & (pairs["AE"] == chosen_v2[1])]["CASE_ID"].drop_duplicates()
    st.caption(f"Evidence card based on {len(pair_cases_v2):,} unique reports for this drug-event pair. Disproportionality is hypothesis-generating, not proof of causality or incidence.")

    dims_v2 = ["AGE_BAND", "SEX_GROUP", "REPORTER_GROUP", "REPORT_SOURCE_GROUP"]
    dim_labels_v2 = {"AGE_BAND":"Age", "SEX_GROUP":"Sex", "REPORTER_GROUP":"Reporter type", "REPORT_SOURCE_GROUP":"Report source"}
    robustness_rows = []
    forest_frames = []
    for dim in dims_v2:
        sdf = compute_stratified_signals(pairs, dim, min_reports_stratum)
        sub = sdf[(sdf["DRUG"] == chosen_v2[0]) & (sdf["AE"] == chosen_v2[1])].copy() if not sdf.empty else pd.DataFrame()
        if sub.empty:
            robustness_rows.append({"Dimension": dim_labels_v2[dim], "Strata": 0, "Signal strata": 0,
                                    "Q p": np.nan, "FDR p": np.nan, "I² %": np.nan,
                                    "Direction reversal": False, "Assessment": "Insufficient data"})
            continue
        h = compute_heterogeneity(sub, dim)
        if h.empty:
            hp = hfdr = i2 = np.nan; flip = False
        else:
            hr = h.iloc[0]; hp = hr["heterogeneity_p"]; hfdr = hr["heterogeneity_p_fdr"]; i2 = hr["I2_pct"]; flip = bool(hr["direction_inconsistent"])
        n_sig = int(sub["is_signal"].sum())
        assessment = "Consistent / no detected heterogeneity"
        if pd.notna(hfdr) and hfdr < hetero_alpha:
            assessment = "Significant heterogeneity"
        if flip:
            assessment = "Direction reversal"
        robustness_rows.append({"Dimension": dim_labels_v2[dim], "Strata": len(sub), "Signal strata": n_sig,
                                "Q p": hp, "FDR p": hfdr, "I² %": i2,
                                "Direction reversal": flip, "Assessment": assessment})
        ff = sub[[dim,"a","ROR","ROR_CI_lower","ROR_CI_upper","is_signal"]].copy()
        ff["Dimension"] = dim_labels_v2[dim]
        ff["Stratum"] = ff[dim].astype(str)
        forest_frames.append(ff[["Dimension","Stratum","a","ROR","ROR_CI_lower","ROR_CI_upper","is_signal"]])

    robustness_df = pd.DataFrame(robustness_rows)
    st.markdown("**Multidimensional robustness profile**")
    st.dataframe(robustness_df, use_container_width=True, hide_index=True)

    if forest_frames:
        forest_df = pd.concat(forest_frames, ignore_index=True)
        st.markdown("**Subgroup forest plot**")
        try:
            import matplotlib.pyplot as plt
            fp = forest_df.dropna(subset=["ROR","ROR_CI_lower","ROR_CI_upper"]).copy()
            fp = fp[(fp["ROR"] > 0) & (fp["ROR_CI_lower"] > 0)]
            if not fp.empty:
                labels = fp["Dimension"] + ": " + fp["Stratum"]
                y = np.arange(len(fp))
                fig, ax = plt.subplots(figsize=(9, max(4, 0.38 * len(fp))))
                xerr = np.vstack([fp["ROR"].to_numpy() - fp["ROR_CI_lower"].to_numpy(),
                                  fp["ROR_CI_upper"].to_numpy() - fp["ROR"].to_numpy()])
                ax.errorbar(fp["ROR"], y, xerr=xerr, fmt="o", capsize=3)
                ax.axvline(1.0, linestyle="--", linewidth=1)
                ax.set_yticks(y); ax.set_yticklabels(labels)
                ax.set_xscale("log"); ax.set_xlabel("Reporting Odds Ratio (log scale), 95% CI")
                ax.invert_yaxis(); fig.tight_layout(); st.pyplot(fig, clear_figure=True)
            else:
                st.info("No finite subgroup confidence intervals available for a forest plot.")
        except Exception as e:
            st.info(f"Forest plot unavailable: {e}")

        st.download_button("Download V2 subgroup evidence (CSV)", export_csv_with_context(forest_df),
                           file_name="v2_subgroup_evidence.csv", mime="text/csv")
    st.download_button("Download V2 robustness profile (CSV)", export_csv_with_context(robustness_df),
                       file_name="v2_robustness_profile.csv", mime="text/csv")

st.subheader("V2 Interpretation Guardrails")
st.markdown("""
- **Positive screening result** means the configured disproportionality threshold was met; it is not a causal conclusion.
- **Masked signal** should mean an aggregate-negative pair with at least one subgroup-positive result plus statistically supported cross-stratum heterogeneity.
- **Direction reversal** is shown separately from ordinary heterogeneity and is the closer analogue of a Simpson-type reversal.
- **FDR-adjusted p-values** are preferred when many drug-event pairs are screened for heterogeneity.
- **I²** describes the magnitude of cross-stratum inconsistency; it should be interpreted together with the number and size of strata.
- **Unknown/missing subgroup data** are retained and data completeness is displayed rather than silently ignored.
""")

# ---------------------------------------------------------------------------
# V2 graphical analytics gallery
# ---------------------------------------------------------------------------
st.header("11. Graphical Signal Analytics")
try:
    from src.pv_signal_explorer.visualizations import top_signals_plot, volcano_plot, completeness_plot

    g1, g2 = st.columns(2)
    with g1:
        st.markdown("**Top screening-positive signals**")
        fig = top_signals_plot(aggregate_df, n=15)
        if fig is not None:
            st.pyplot(fig, clear_figure=True)
        else:
            st.info("No screening-positive signals available for this plot.")
    with g2:
        st.markdown("**Signal map**")
        fig = volcano_plot(aggregate_df)
        if fig is not None:
            st.pyplot(fig, clear_figure=True)
        else:
            st.info("Insufficient finite statistics for the signal map.")

    st.markdown("**Subgroup data completeness**")
    fig = completeness_plot(quality_df)
    if fig is not None:
        st.pyplot(fig, clear_figure=True)
    st.caption("Marker size in the signal map reflects pair report count. Visualizations support prioritization and do not imply causality.")
except Exception as e:
    st.info(f"Graphical analytics unavailable: {e}")

# ---------------------------------------------------------------------------
# V3 Bayesian disproportionality + temporal surveillance
# ---------------------------------------------------------------------------
st.header("12. Bayesian Signal Detection — EBGM and Information Component")
st.caption("Open research implementations: two-Gamma Poisson empirical-Bayes shrinkage (MGPS-style) and a BCPNN-style IC approximation. They do not reproduce proprietary FDA/UMC software exactly.")
try:
    from src.pv_signal_explorer.bayesian import bayesian_signal_table
    with st.spinner("Fitting empirical-Bayes prior and computing EBGM / EB05 / EB95 / IC..."):
        bayes_df, eb_params = session_cached(
            "bayesian", (build_signature, min_reports),
            lambda: bayesian_signal_table(pairs, min_reports=min_reports)
        )
    b1,b2,b3,b4 = st.columns(4)
    b1.metric("Bayesian pairs", f"{len(bayes_df):,}")
    b2.metric("EB05 ≥ 2", f"{int(bayes_df['EB_signal'].sum()):,}")
    b3.metric("IC025 > 0", f"{int(bayes_df['IC_signal'].sum()):,}")
    b4.metric("Both criteria", f"{int((bayes_df.EB_signal & bayes_df.IC_signal).sum()):,}")
    with st.expander("Empirical-Bayes fitted prior"):
        st.json({k:(round(v,6) if isinstance(v,float) else v) for k,v in eb_params.items()})
    st.dataframe(bayes_df[["DRUG","AE","observed","expected","RR_raw","EBGM","EB05","EB95","IC","IC025","IC975","EB_signal","IC_signal"]],
                 use_container_width=True, height=380)
    st.download_button("Download EBGM + IC results (CSV)", export_csv_with_context(bayes_df),
                       file_name="bayesian_signal_results.csv", mime="text/csv")

    import matplotlib.pyplot as plt
    finite = bayes_df.replace([np.inf,-np.inf],np.nan).dropna(subset=["EBGM","IC"])
    if not finite.empty:
        fig, ax = plt.subplots(figsize=(9,5))
        sizes = 12 + 5*np.sqrt(finite["observed"].clip(lower=1))
        ax.scatter(np.log2(finite["EBGM"].clip(lower=1e-9)), finite["IC"], s=sizes, alpha=.55)
        ax.axvline(1.0, linestyle="--", linewidth=1); ax.axhline(0.0, linestyle="--", linewidth=1)
        ax.set_xlabel("log2(EBGM)"); ax.set_ylabel("Information Component (IC)")
        ax.set_title("Bayesian signal landscape")
        fig.tight_layout(); st.pyplot(fig, clear_figure=True)
except Exception as e:
    st.error(f"Bayesian analysis could not be completed: {e}")
    bayes_df = pd.DataFrame(); eb_params = None

st.header("13. Temporal Surveillance")
st.write("Quarter-by-quarter surveillance recomputes the reporting background inside each period, then follows EBGM/EB05 and IC/IC025 over time. This is intended for emergence and persistence review, not incidence estimation.")
if event_dt_by_case is None or event_dt_by_case.notna().sum() == 0:
    st.info("Temporal surveillance requires parseable EVENT_DT values in DEMO.")
else:
    try:
        from src.pv_signal_explorer.temporal import temporal_surveillance, pair_temporal_summary
        temporal_min = st.slider("Minimum reports per period", 1, 20, 3, 1, key="temporal_min")
        with st.spinner("Computing quarterly Bayesian surveillance..."):
            temporal_df, temporal_params = session_cached(
                "temporal", (build_signature, temporal_min),
                lambda: temporal_surveillance(pairs, event_dt_by_case, min_reports=temporal_min, freq="Q", eb_params=eb_params)
            )
        if temporal_df.empty:
            st.info("No quarterly drug-event combinations met the selected minimum report count.")
        else:
            periods = temporal_df["PERIOD"].nunique()
            t1,t2,t3 = st.columns(3)
            t1.metric("Periods analysed", periods)
            t2.metric("Earliest period", temporal_df["PERIOD"].min())
            t3.metric("Latest period", temporal_df["PERIOD"].max())
            opts = list(zip(bayes_df["DRUG"],bayes_df["AE"])) if not bayes_df.empty else list(zip(temporal_df.DRUG,temporal_df.AE))
            opts = list(dict.fromkeys(opts))
            chosen_t = st.selectbox("Drug-event pair for temporal review", opts, format_func=lambda z:f"{z[0]} / {z[1]}", key="temporal_pair")
            tx, tsum = pair_temporal_summary(temporal_df, chosen_t[0], chosen_t[1])
            if tx.empty:
                st.info("This pair did not meet the per-period minimum in any quarter.")
            else:
                q1,q2,q3,q4 = st.columns(4)
                q1.metric("First Bayesian alert", tsum["first_signal_period"] or "None")
                q2.metric("Signal periods", tsum["signal_quarters"])
                q3.metric("Longest run", tsum["longest_signal_run"])
                q4.metric("Latest EB05", f"{tsum['latest_EB05']:.2f}")
                fig, ax = plt.subplots(figsize=(10,5))
                x=np.arange(len(tx)); ax.plot(x, tx["EBGM"], marker="o", label="EBGM")
                ax.plot(x, tx["EB05"], marker=".", linestyle="--", label="EB05")
                ax.axhline(2.0, linestyle=":", linewidth=1, label="EB05 screening reference = 2")
                ax.set_xticks(x); ax.set_xticklabels(tx["PERIOD"], rotation=45, ha="right")
                ax.set_ylabel("Empirical-Bayes reporting ratio"); ax.set_title(f"Temporal EBGM — {chosen_t[0]} / {chosen_t[1]}")
                ax.legend(); fig.tight_layout(); st.pyplot(fig, clear_figure=True)
                fig, ax = plt.subplots(figsize=(10,5))
                ax.plot(x, tx["IC"], marker="o", label="IC"); ax.plot(x, tx["IC025"], marker=".", linestyle="--", label="IC025")
                ax.axhline(0.0, linestyle=":", linewidth=1, label="IC025 screening reference = 0")
                ax.set_xticks(x); ax.set_xticklabels(tx["PERIOD"], rotation=45, ha="right")
                ax.set_ylabel("Information Component"); ax.set_title(f"Temporal IC — {chosen_t[0]} / {chosen_t[1]}")
                ax.legend(); fig.tight_layout(); st.pyplot(fig, clear_figure=True)
                st.dataframe(tx[["PERIOD","observed","expected","RR_raw","EBGM","EB05","EB95","IC","IC025","IC975","signal","EBGM_qoq_pct"]], use_container_width=True)
                st.download_button("Download selected temporal trajectory (CSV)", export_csv_with_context(tx),
                                   file_name="temporal_signal_trajectory.csv", mime="text/csv")
            st.download_button("Download all quarterly surveillance results (CSV)", export_csv_with_context(temporal_df),
                               file_name="quarterly_bayesian_surveillance.csv", mime="text/csv")
    except Exception as e:
        st.error(f"Temporal surveillance could not be completed: {e}")
