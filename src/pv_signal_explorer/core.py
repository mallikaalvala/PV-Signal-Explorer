"""Core statistical and data-processing functions for PV Signal Explorer V2."""

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

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


def build_pairs_with_subgroups(drug_df, reac_df, demo_df, suspect_only: bool, age_edges, age_labels):
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
    if suspect_only and "ROLE_COD" in d.columns:
        d = d[d["ROLE_COD"].astype(str).str.strip().str.upper().isin(["PS", "SS"])]
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


