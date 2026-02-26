# app.py
import re
from datetime import date

import pandas as pd
import plotly.express as px
import streamlit as st


# -----------------------------
# Column normalization & mapping
# -----------------------------
LOGICAL_FIELDS = [
    "idx",
    "label",
    "customerid",
    "transactionid",
    "transactiondate",
    "productcategory",
    "purchaseamount",
    "customeragegroup",
    "customergender",
    "customerregion",
    "customersatisfaction",
    "retailchannel",
]


def _normalize_colname(c: str) -> str:
    c = "" if c is None else str(c)
    c = c.strip().lower()
    c = re.sub(r"[ \-]+", "_", c)
    c = re.sub(r"_+", "_", c)
    return c


def _build_mapping(normalized_cols):
    # Heuristic patterns for robust matching without assuming exact spelling/case
    patterns = {
        "idx": [r"^idx$", r"^index$", r"^row_?id$", r"^record_?id$"],
        "label": [r"^label$", r"segment", r"customer_?segment", r"class", r"cluster"],
        "customerid": [r"customer_?id", r"cust_?id", r"client_?id", r"^customerid$"],
        "transactionid": [r"transaction_?id", r"txn_?id", r"order_?id", r"receipt_?id"],
        "transactiondate": [
            r"transaction_?date",
            r"txn_?date",
            r"order_?date",
            r"purchase_?date",
            r"date",
        ],
        "productcategory": [
            r"product_?category",
            r"category",
            r"product_?type",
            r"department",
        ],
        "purchaseamount": [
            r"purchase_?amount",
            r"amount",
            r"revenue",
            r"sales",
            r"spend",
            r"price",
            r"total",
            r"order_?value",
        ],
        "customeragegroup": [r"age_?group", r"customer_?age_?group", r"ageband", r"age_?band"],
        "customergender": [r"gender", r"customer_?gender", r"sex"],
        "customerregion": [r"region", r"customer_?region", r"state", r"province", r"area", r"territory"],
        "customersatisfaction": [
            r"satisfaction",
            r"customer_?satisfaction",
            r"rating",
            r"score",
            r"csat",
        ],
        "retailchannel": [r"channel", r"retail_?channel", r"sales_?channel", r"store_?type"],
    }

    chosen = {}
    used = set()

    def score(col, logical):
        # Prefer exact-ish and anchored matches
        s = 0
        for p in patterns[logical]:
            if re.search(p, col):
                s += 5
            if re.fullmatch(p, col):
                s += 10
        # Small boost for containing logical token
        token = logical.replace("customer", "").replace("transaction", "txn")
        if token and token in col:
            s += 1
        return s

    for logical in LOGICAL_FIELDS:
        best = None
        best_score = -1
        for c in normalized_cols:
            if c in used:
                continue
            sc = score(c, logical)
            if sc > best_score:
                best_score = sc
                best = c
        if best is not None and best_score > 0:
            chosen[logical] = best
            used.add(best)

    return chosen


# -----------------------------
# Loading & Cleaning (auditable)
# -----------------------------
@st.cache_data(show_spinner=False)
def load_and_clean():
    try:
        df_raw = pd.read_excel("NR_dataset.xlsx")
    except FileNotFoundError:
        st.error('Dataset file "NR_dataset.xlsx" not found in the app directory.')
        st.stop()
    except Exception as e:
        st.error(f'Failed to read "NR_dataset.xlsx": {e}')
        st.stop()

    raw_rows = len(df_raw)

    # Normalize columns
    orig_cols = list(df_raw.columns)
    norm_cols = [_normalize_colname(c) for c in orig_cols]

    # Deduplicate normalized names by suffixing
    seen = {}
    final_cols = []
    for c in norm_cols:
        if c not in seen:
            seen[c] = 0
            final_cols.append(c)
        else:
            seen[c] += 1
            final_cols.append(f"{c}_{seen[c]}")
    df = df_raw.copy()
    df.columns = final_cols

    mapping = _build_mapping(df.columns.tolist())

    missing_required = [f for f in LOGICAL_FIELDS if f not in mapping]
    if missing_required:
        st.error("Missing required logical fields: " + ", ".join(missing_required))
        st.write("Available columns:", df.columns.tolist())
        st.stop()

    # Rebuild a canonical dataframe with logical column names
    df = df[[mapping[f] for f in LOGICAL_FIELDS]].copy()
    df.columns = LOGICAL_FIELDS

    report_rows = []
    remaining = len(df)

    def _log(step, removed):
        nonlocal remaining
        remaining -= removed
        report_rows.append({"step": step, "rows_removed": int(removed), "rows_remaining": int(remaining)})

    # Step 1: Drop fully empty rows
    before = len(df)
    df = df.dropna(how="all").copy()
    _log("Drop fully empty rows", before - len(df))

    # Step 2: Parse dates; drop NaT
    df["transactiondate"] = pd.to_datetime(df["transactiondate"], errors="coerce")
    before = len(df)
    nat_count = int(df["transactiondate"].isna().sum())
    df = df.dropna(subset=["transactiondate"]).copy()
    _log("Drop rows with invalid transactiondate (NaT)", nat_count)

    # Step 3: Amounts (numeric, NaN, <=0)
    df["purchaseamount"] = pd.to_numeric(df["purchaseamount"], errors="coerce")
    before = len(df)
    nan_amt = int(df["purchaseamount"].isna().sum())
    df = df.dropna(subset=["purchaseamount"]).copy()
    _log("Drop rows with NaN purchaseamount", nan_amt)

    before = len(df)
    nonpos = int((df["purchaseamount"] <= 0).sum())
    df = df.loc[df["purchaseamount"] > 0].copy()
    _log("Drop rows with purchaseamount <= 0", nonpos)

    # Step 4: Satisfaction (numeric, NaN, not in 1..5)
    df["customersatisfaction"] = pd.to_numeric(df["customersatisfaction"], errors="coerce")
    before = len(df)
    nan_sat = int(df["customersatisfaction"].isna().sum())
    df = df.dropna(subset=["customersatisfaction"]).copy()
    _log("Drop rows with NaN customersatisfaction", nan_sat)

    before = len(df)
    invalid_sat = int((~df["customersatisfaction"].isin([1, 2, 3, 4, 5])).sum())
    df = df.loc[df["customersatisfaction"].isin([1, 2, 3, 4, 5])].copy()
    _log("Drop rows with customersatisfaction not in [1..5]", invalid_sat)

    # Step 5: Critical dimensions must not be missing/blank
    critical = ["label", "customerregion", "productcategory", "retailchannel", "customerid", "transactionid"]
    # Trim strings for accurate blank detection
    for c in critical:
        df[c] = df[c].astype("string")

    trimmed = df[critical].apply(lambda s: s.astype("string").str.strip())
    missing_any = trimmed.isna() | (trimmed == "")
    before = len(df)
    removed_critical = int(missing_any.any(axis=1).sum())
    df = df.loc[~missing_any.any(axis=1)].copy()
    _log("Drop rows with missing/blank critical fields (any of label, region, category, channel, customerid, transactionid)", removed_critical)

    # Step 6: Standardize strings (trim + consistent casing)
    # Convert IDs to string, trimmed (no title-casing)
    df["customerid"] = df["customerid"].astype("string").str.strip()
    df["transactionid"] = df["transactionid"].astype("string").str.strip()

    # Strip whitespace for all string columns
    for c in df.columns:
        if df[c].dtype == "object" or str(df[c].dtype).startswith("string"):
            df[c] = df[c].astype("string").str.strip()

    # Consistent casing for categorical dimensions
    cat_cols = ["label", "customerregion", "productcategory", "retailchannel", "customergender", "customeragegroup"]
    for c in cat_cols:
        df[c] = df[c].astype("string").str.strip()
        df[c] = df[c].where(df[c].isna(), df[c].str.replace(r"\s+", " ", regex=True))
        df[c] = df[c].str.title()

    # Step 7: Derived fields
    df["year_month"] = df["transactiondate"].dt.to_period("M").astype(str)
    df["month_start"] = df["transactiondate"].dt.to_period("M").dt.to_timestamp()

    # Profiling summary
    min_date = df["transactiondate"].min() if len(df) else pd.NaT
    max_date = df["transactiondate"].max() if len(df) else pd.NaT
    profiling = {
        "raw_rows_loaded": int(raw_rows),
        "rows_after_cleaning": int(len(df)),
        "min_transactiondate": None if pd.isna(min_date) else min_date,
        "max_transactiondate": None if pd.isna(max_date) else max_date,
        "unique_customers": int(df["customerid"].nunique()) if len(df) else 0,
        "unique_transactions": int(df["transactionid"].nunique()) if len(df) else 0,
        "unique_segments": int(df["label"].nunique()) if len(df) else 0,
        "unique_regions": int(df["customerregion"].nunique()) if len(df) else 0,
        "unique_categories": int(df["productcategory"].nunique()) if len(df) else 0,
        "unique_channels": int(df["retailchannel"].nunique()) if len(df) else 0,
        "months_covered": int(df["month_start"].nunique()) if len(df) else 0,
    }

    cleaning_report = pd.DataFrame(report_rows)
    return df, cleaning_report, profiling


# -----------------------------
# Filtering helpers
# -----------------------------
def _multiselect_with_all(label, options, default_all=True, help_text=None):
    opts = ["All"] + options
    default = ["All"] if default_all else []
    sel = st.multiselect(label, opts, default=default, help=help_text)
    if not sel:
        sel = ["All"]
    if "All" in sel:
        return None  # None means no filter
    return sel


def _safe_nunique(series):
    try:
        return int(series.nunique())
    except Exception:
        return int(len(series))


def _format_currency(x):
    try:
        return f"${x:,.2f}"
    except Exception:
        return str(x)


def _compute_mom_decline_metrics(df_filt):
    # Returns (txn_decline_pct, cust_decline_share, cust_decline_n, cust_decline_den, last_month, prev_month)
    if df_filt.empty:
        return None, None, 0, 0, None, None

    months = sorted(df_filt["month_start"].dropna().unique())
    if len(months) < 2:
        return None, None, 0, 0, (months[-1] if months else None), None

    last_m = months[-1]
    prev_m = months[-2]

    last_df = df_filt[df_filt["month_start"] == last_m]
    prev_df = df_filt[df_filt["month_start"] == prev_m]

    last_txn = _safe_nunique(last_df["transactionid"]) if "transactionid" in last_df else len(last_df)
    prev_txn = _safe_nunique(prev_df["transactionid"]) if "transactionid" in prev_df else len(prev_df)

    txn_decline_pct = None
    if prev_txn and prev_txn > 0:
        txn_decline_pct = max(0.0, (prev_txn - last_txn) / prev_txn) * 100.0

    prev_c = prev_df.groupby("customerid", dropna=False)["purchaseamount"].sum()
    last_c = last_df.groupby("customerid", dropna=False)["purchaseamount"].sum()
    common = prev_c.index.intersection(last_c.index)
    den = int(len(common))
    if den == 0:
        cust_decline_share = None
        decline_n = 0
    else:
        decline_n = int((last_c.loc[common] < prev_c.loc[common]).sum())
        cust_decline_share = (decline_n / den) * 100.0

    return txn_decline_pct, cust_decline_share, decline_n, den, last_m, prev_m


# -----------------------------
# App
# -----------------------------
st.set_page_config(layout="wide", page_title="NovaRetail Customer Intelligence Dashboard")

st.title("NovaRetail Customer Intelligence Dashboard")
st.caption(
    "Explore customer behavior, revenue patterns, satisfaction, and early warning signals across segments, regions, categories, channels, and demographics."
)

df_clean, cleaning_report, profiling = load_and_clean()

with st.sidebar:
    st.header("Filters")

    # Base options from cleaned data
    seg_opts = sorted(df_clean["label"].dropna().unique().tolist())
    reg_opts = sorted(df_clean["customerregion"].dropna().unique().tolist())
    cat_opts = sorted(df_clean["productcategory"].dropna().unique().tolist())
    ch_opts = sorted(df_clean["retailchannel"].dropna().unique().tolist())
    gen_opts = sorted(df_clean["customergender"].dropna().unique().tolist())
    age_opts = sorted(df_clean["customeragegroup"].dropna().unique().tolist())

    sel_segments = _multiselect_with_all("Segment (Label)", seg_opts)
    sel_regions = _multiselect_with_all("Region", reg_opts)
    sel_categories = _multiselect_with_all("Product Category", cat_opts)
    sel_channels = _multiselect_with_all("Channel", ch_opts)
    sel_genders = _multiselect_with_all("Gender", gen_opts)
    sel_agegroups = _multiselect_with_all("Age Group", age_opts)

    min_d = df_clean["transactiondate"].min().date() if len(df_clean) else date.today()
    max_d = df_clean["transactiondate"].max().date() if len(df_clean) else date.today()
    date_range = st.date_input("Date range", value=(min_d, max_d), min_value=min_d, max_value=max_d)

    sat_min, sat_max = st.slider("Satisfaction range", 1, 5, (1, 5))

    amt_min = float(df_clean["purchaseamount"].min()) if len(df_clean) else 0.0
    amt_max = float(df_clean["purchaseamount"].max()) if len(df_clean) else 0.0
    if amt_min == amt_max:
        amt_range = (amt_min, amt_max)
        st.slider("Purchase amount range", min_value=amt_min, max_value=amt_max, value=amt_range, disabled=True)
    else:
        amt_range = st.slider("Purchase amount range", min_value=amt_min, max_value=amt_max, value=(amt_min, amt_max))

    st.divider()
    st.subheader("Customer selector")
    cust_list = sorted(df_clean["customerid"].dropna().unique().tolist())
    include_only = st.toggle("Include only selected customer", value=False)
    selected_customer = st.selectbox("Customer", options=cust_list, index=0 if cust_list else None, disabled=(not cust_list))

    st.divider()
    top_n = st.selectbox("Top-N for ranking views", options=[5, 10, 15, 20], index=1)

    with st.expander("Data Cleaning & Quality Log", expanded=False):
        st.write("Total raw rows loaded:", profiling["raw_rows_loaded"])
        st.write("Total rows after cleaning:", profiling["rows_after_cleaning"])
        st.dataframe(cleaning_report, use_container_width=True, hide_index=True)

        st.write("Min transaction date (clean):", profiling["min_transactiondate"])
        st.write("Max transaction date (clean):", profiling["max_transactiondate"])
        st.write("Unique customers (clean):", profiling["unique_customers"])
        st.write("Unique transactions (clean):", profiling["unique_transactions"])

        dropped = profiling["raw_rows_loaded"] - profiling["rows_after_cleaning"]
        drop_pct = (dropped / profiling["raw_rows_loaded"] * 100.0) if profiling["raw_rows_loaded"] else 0.0
        if profiling["raw_rows_loaded"] and drop_pct > 20:
            st.warning(f"High drop rate during cleaning: {drop_pct:.1f}% of rows removed.")

        months_cov = profiling.get("months_covered", 0)
        if months_cov is not None and months_cov <= 1:
            st.warning("Date range after cleaning is very small (≤ 1 month). Trend and growth views may be limited.")


# Apply filters (never mutate df_clean)
df_filt = df_clean.copy()

# Date range filter
if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start_d, end_d = date_range
else:
    start_d, end_d = min_d, max_d

df_filt = df_filt[(df_filt["transactiondate"].dt.date >= start_d) & (df_filt["transactiondate"].dt.date <= end_d)]

# Dimensional filters
if sel_segments is not None:
    df_filt = df_filt[df_filt["label"].isin(sel_segments)]
if sel_regions is not None:
    df_filt = df_filt[df_filt["customerregion"].isin(sel_regions)]
if sel_categories is not None:
    df_filt = df_filt[df_filt["productcategory"].isin(sel_categories)]
if sel_channels is not None:
    df_filt = df_filt[df_filt["retailchannel"].isin(sel_channels)]
if sel_genders is not None:
    df_filt = df_filt[df_filt["customergender"].isin(sel_genders)]
if sel_agegroups is not None:
    df_filt = df_filt[df_filt["customeragegroup"].isin(sel_agegroups)]

# Satisfaction & amount filters
df_filt = df_filt[(df_filt["customersatisfaction"] >= sat_min) & (df_filt["customersatisfaction"] <= sat_max)]
df_filt = df_filt[(df_filt["purchaseamount"] >= float(amt_range[0])) & (df_filt["purchaseamount"] <= float(amt_range[1]))]

# Customer selector filter
if include_only and selected_customer:
    df_filt = df_filt[df_filt["customerid"] == str(selected_customer)]

# Always show filtered date range on-screen
col_a, col_b, col_c = st.columns([1.2, 1.2, 2.6])
with col_a:
    st.metric("Filtered Rows", f"{len(df_filt):,}")
with col_b:
    if not df_filt.empty:
        st.metric("Filtered Date Range", f"{df_filt['transactiondate'].min().date()} → {df_filt['transactiondate'].max().date()}")
    else:
        st.metric("Filtered Date Range", "—")
with col_c:
    st.caption(
        f"Active filters apply to all KPIs, charts, and tables. Top-N: {top_n}. Satisfaction: {sat_min}–{sat_max}. Amount: {_format_currency(amt_range[0])}–{_format_currency(amt_range[1])}."
    )

if df_filt.empty:
    st.warning("No data matches the current filters.")
    st.divider()
else:
    # -----------------------------
    # KPI row (executive-friendly)
    # -----------------------------
    revenue = float(df_filt["purchaseamount"].sum())
    txns = int(df_filt["transactionid"].nunique()) if "transactionid" in df_filt else int(len(df_filt))
    custs = int(df_filt["customerid"].nunique()) if "customerid" in df_filt else 0
    aov = (revenue / txns) if txns > 0 else 0.0
    avg_sat = float(df_filt["customersatisfaction"].mean()) if len(df_filt) else 0.0

    txn_decline_pct, cust_decline_share, decline_n, decline_den, last_m, prev_m = _compute_mom_decline_metrics(df_filt)

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Total Revenue", _format_currency(revenue))
    k2.metric("Total Transactions", f"{txns:,}")
    k3.metric("Unique Customers", f"{custs:,}")
    k4.metric("Average Order Value", _format_currency(aov))
    k5.metric("Avg Satisfaction", f"{avg_sat:.2f}")

    if txn_decline_pct is None:
        k6.metric("MoM Transaction Decline %", "—")
    else:
        k6.metric("MoM Transaction Decline %", f"{txn_decline_pct:.1f}%")

    if cust_decline_share is None:
        st.caption("Decline share by customers: — (need at least 2 months and overlapping customers).")
    else:
        st.caption(f"Decline share by customers (MoM spend down, last vs prior month): {cust_decline_share:.1f}% (n={decline_n:,} of {decline_den:,}).")

st.divider()

# -----------------------------
# Sanity Checks
# -----------------------------
st.subheader("Sanity Checks")
sc1, sc2, sc3 = st.columns([1.2, 1.2, 2.6])
with sc1:
    st.write("Rows after filters:", f"{len(df_filt):,}")
with sc2:
    st.write("Total revenue after filters:", _format_currency(float(df_filt["purchaseamount"].sum()) if not df_filt.empty else 0.0))
with sc3:
    if df_filt.empty:
        st.info("No data for top months.")
    else:
        top_months = (
            df_filt.groupby("month_start", as_index=False)["purchaseamount"]
            .sum()
            .sort_values("purchaseamount", ascending=False)
            .head(3)
        )
        top_months["month_start"] = top_months["month_start"].dt.date.astype(str)
        top_months.rename(columns={"month_start": "month", "purchaseamount": "revenue"}, inplace=True)
        top_months["revenue"] = top_months["revenue"].map(lambda x: _format_currency(float(x)))
        st.dataframe(top_months, use_container_width=True, hide_index=True)

st.divider()

# -----------------------------
# Charts (Plotly only)
# -----------------------------
c1, c2 = st.columns([1, 1])

with c1:
    st.subheader("Revenue by Segment")
    if df_filt.empty:
        st.info("No data to display.")
    else:
        seg_rev = (
            df_filt.groupby("label", as_index=False)["purchaseamount"]
            .sum()
            .sort_values("purchaseamount", ascending=False)
        )
        fig = px.bar(seg_rev, x="label", y="purchaseamount", title="Revenue by Segment (Filtered)")
        fig.update_layout(xaxis_title="Segment", yaxis_title="Revenue")
        st.plotly_chart(fig, use_container_width=True)

with c2:
    st.subheader("Monthly Revenue Trend")
    split = st.checkbox("Split by segment", value=True, key="split_by_segment")
    if df_filt.empty:
        st.info("No data to display.")
    else:
        if split:
            trend = (
                df_filt.groupby(["month_start", "label"], as_index=False)["purchaseamount"]
                .sum()
                .sort_values("month_start")
            )
            fig = px.line(trend, x="month_start", y="purchaseamount", color="label", title="Monthly Revenue Trend (Split by Segment)")
        else:
            trend = (
                df_filt.groupby("month_start", as_index=False)["purchaseamount"]
                .sum()
                .sort_values("month_start")
            )
            fig = px.line(trend, x="month_start", y="purchaseamount", title="Monthly Revenue Trend (Total)")
        fig.update_layout(xaxis_title="Month", yaxis_title="Revenue")
        st.plotly_chart(fig, use_container_width=True)

c3, c4 = st.columns([1, 1])

with c3:
    st.subheader("Satisfaction vs Spend (Customer-level)")
    if df_filt.empty:
        st.info("No data to display.")
    else:
        cust_scatter = (
            df_filt.groupby(["customerid", "label"], as_index=False)
            .agg(total_spend=("purchaseamount", "sum"), avg_satisfaction=("customersatisfaction", "mean"))
        )
        fig = px.scatter(
            cust_scatter,
            x="avg_satisfaction",
            y="total_spend",
            color="label",
            hover_data=["customerid"],
            title="Customer-level: Total Spend vs Avg Satisfaction",
        )
        fig.update_layout(xaxis_title="Avg Satisfaction", yaxis_title="Total Spend")
        st.plotly_chart(fig, use_container_width=True)

with c4:
    st.subheader("Channel Mix (Revenue)")
    if df_filt.empty:
        st.info("No data to display.")
    else:
        ch_rev = (
            df_filt.groupby("retailchannel", as_index=False)["purchaseamount"]
            .sum()
            .sort_values("purchaseamount", ascending=False)
        )
        fig = px.pie(ch_rev, names="retailchannel", values="purchaseamount", hole=0.5, title="Revenue Share by Channel")
        st.plotly_chart(fig, use_container_width=True)

st.subheader("Region × Category Performance (Revenue Heatmap)")
if df_filt.empty:
    st.info("No data to display.")
else:
    reg_tot = df_filt.groupby("customerregion")["purchaseamount"].sum().sort_values(ascending=False)
    cat_tot = df_filt.groupby("productcategory")["purchaseamount"].sum().sort_values(ascending=False)
    top_regions = reg_tot.head(top_n).index.tolist()
    top_cats = cat_tot.head(top_n).index.tolist()

    heat_df = df_filt[df_filt["customerregion"].isin(top_regions) & df_filt["productcategory"].isin(top_cats)]
    pivot = (
        heat_df.pivot_table(
            index="customerregion",
            columns="productcategory",
            values="purchaseamount",
            aggfunc="sum",
            fill_value=0.0,
        )
        .reindex(index=top_regions, columns=top_cats)
    )
    fig = px.imshow(
        pivot,
        aspect="auto",
        title=f"Top {top_n} Regions × Top {top_n} Categories (Revenue)",
        labels=dict(x="Product Category", y="Region", color="Revenue"),
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# -----------------------------
# At Risk / Early Warning view
# -----------------------------
st.subheader("At Risk / Early Warning Signals")
if df_filt.empty:
    st.info("No data to display.")
else:
    months = sorted(df_filt["month_start"].dropna().unique())
    if len(months) < 2:
        st.info("Not enough monthly coverage for growth/decline calculations (need at least 2 distinct months in the filtered range).")
    else:
        last_m = months[-1]
        prev_m = months[-2]

        last_r = df_filt[df_filt["month_start"] == last_m].groupby("customerregion")["purchaseamount"].sum()
        prev_r = df_filt[df_filt["month_start"] == prev_m].groupby("customerregion")["purchaseamount"].sum()

        regions = sorted(set(last_r.index).union(set(prev_r.index)))
        rows = []
        for r in regions:
            l = float(last_r.get(r, 0.0))
            p = float(prev_r.get(r, 0.0))
            growth = None
            if p > 0:
                growth = (l - p) / p * 100.0
            # Decline share by customers within region (MoM spend down, last vs prior)
            last_df_r = df_filt[(df_filt["month_start"] == last_m) & (df_filt["customerregion"] == r)]
            prev_df_r = df_filt[(df_filt["month_start"] == prev_m) & (df_filt["customerregion"] == r)]
            prev_c = prev_df_r.groupby("customerid")["purchaseamount"].sum()
            last_c = last_df_r.groupby("customerid")["purchaseamount"].sum()
            common = prev_c.index.intersection(last_c.index)
            den = int(len(common))
            decline_share = None
            if den > 0:
                decline_share = float((last_c.loc[common] < prev_c.loc[common]).mean() * 100.0)

            avg_sat = float(df_filt[df_filt["customerregion"] == r]["customersatisfaction"].mean())
            total_rev = float(df_filt[df_filt["customerregion"] == r]["purchaseamount"].sum())
            rows.append(
                {
                    "region": r,
                    "total_revenue": total_rev,
                    "avg_satisfaction": avg_sat,
                    "mom_growth_pct": growth,
                    "customer_decline_share_pct": decline_share,
                }
            )

        risk = pd.DataFrame(rows)
        # Rank: low growth + low satisfaction + high decline share
        # Keep NAs at the bottom
        risk["mom_growth_rank"] = risk["mom_growth_pct"].rank(ascending=True, na_option="bottom")
        risk["sat_rank"] = risk["avg_satisfaction"].rank(ascending=True, na_option="bottom")
        risk["decline_rank"] = risk["customer_decline_share_pct"].rank(ascending=False, na_option="bottom")
        risk["risk_score"] = risk[["mom_growth_rank", "sat_rank", "decline_rank"]].sum(axis=1)

        risk_view = risk.sort_values("risk_score", ascending=False).head(top_n).copy()
        risk_view_display = risk_view[
            ["region", "total_revenue", "avg_satisfaction", "mom_growth_pct", "customer_decline_share_pct", "risk_score"]
        ].copy()
        risk_view_display["total_revenue"] = risk_view_display["total_revenue"].map(lambda x: _format_currency(float(x)))
        risk_view_display["avg_satisfaction"] = risk_view_display["avg_satisfaction"].map(lambda x: f"{float(x):.2f}")
        risk_view_display["mom_growth_pct"] = risk_view_display["mom_growth_pct"].map(lambda x: "—" if pd.isna(x) else f"{float(x):.1f}%")
        risk_view_display["customer_decline_share_pct"] = risk_view_display["customer_decline_share_pct"].map(
            lambda x: "—" if pd.isna(x) else f"{float(x):.1f}%"
        )
        risk_view_display["risk_score"] = risk_view_display["risk_score"].map(lambda x: f"{float(x):.1f}")

        st.caption(f"Scored using MoM growth (last month {pd.to_datetime(last_m).date()} vs prior {pd.to_datetime(prev_m).date()}), satisfaction, and customer decline share.")
        st.dataframe(risk_view_display, use_container_width=True, hide_index=True)

        # Optional visual: revenue growth vs satisfaction (bubble = revenue)
        fig = px.scatter(
            risk,
            x="mom_growth_pct",
            y="avg_satisfaction",
            size="total_revenue",
            hover_name="region",
            title="Regions: MoM Growth vs Avg Satisfaction (Bubble size = Total Revenue)",
        )
        fig.update_layout(xaxis_title="MoM Growth % (Revenue)", yaxis_title="Avg Satisfaction")
        st.plotly_chart(fig, use_container_width=True)

st.divider()

# -----------------------------
# Insights & Recommended Actions
# -----------------------------
st.subheader("Insights & Recommended Actions (Computed)")
insights = []

if df_filt.empty:
    insights.append("No insights available because the current filters match zero rows.")
else:
    # Best performers
    seg_rev = df_filt.groupby("label")["purchaseamount"].sum().sort_values(ascending=False)
    if not seg_rev.empty:
        best_seg = seg_rev.index[0]
        insights.append(f"Top segment by revenue: **{best_seg}** ({_format_currency(float(seg_rev.iloc[0]))}).")

    reg_rev = df_filt.groupby("customerregion")["purchaseamount"].sum().sort_values(ascending=False)
    if not reg_rev.empty:
        best_reg = reg_rev.index[0]
        insights.append(f"Top region by revenue: **{best_reg}** ({_format_currency(float(reg_rev.iloc[0]))}).")

    cat_rev = df_filt.groupby("productcategory")["purchaseamount"].sum().sort_values(ascending=False)
    if not cat_rev.empty:
        best_cat = cat_rev.index[0]
        insights.append(f"Top product category by revenue: **{best_cat}** ({_format_currency(float(cat_rev.iloc[0]))}).")

    # Lowest satisfaction pockets
    sat_by_seg = df_filt.groupby("label")["customersatisfaction"].mean().sort_values()
    if len(sat_by_seg) >= 1:
        worst_seg = sat_by_seg.index[0]
        insights.append(f"Lowest average satisfaction segment: **{worst_seg}** ({float(sat_by_seg.iloc[0]):.2f}). Consider targeted service recovery and post-purchase follow-ups.")

    # Decline signals
    txn_decline_pct, cust_decline_share, decline_n, decline_den, last_m, prev_m = _compute_mom_decline_metrics(df_filt)
    if txn_decline_pct is None or last_m is None or prev_m is None:
        insights.append("MoM decline signals not available (need at least 2 distinct months in the filtered range).")
    else:
        if txn_decline_pct > 0:
            insights.append(
                f"Early warning: transactions declined **{txn_decline_pct:.1f}%** month-over-month (last vs prior month). Prioritize retention offers and channel/category diagnostics in the affected time window."
            )
        else:
            insights.append("No transaction decline detected month-over-month (last vs prior month) within the filtered range.")

        if cust_decline_share is not None and decline_den > 0:
            insights.append(
                f"Customer-level decline: **{cust_decline_share:.1f}%** of customers with activity in both months spent less in the latest month. Consider win-back messaging and tailored bundles for slipping customers."
            )

    # Action grounded in channel/category mix (top lever)
    ch_mix = df_filt.groupby("retailchannel")["purchaseamount"].sum().sort_values(ascending=False)
    if len(ch_mix) >= 2:
        top_ch = ch_mix.index[0]
        insights.append(f"Channel focus: **{top_ch}** drives the largest share of revenue in the filtered view—optimize merchandising and promotions here first.")

# Show 3–5 bullets (adaptively)
for b in insights[:5]:
    st.markdown(f"- {b}")

st.divider()

# -----------------------------
# Filtered table + download
# -----------------------------
st.subheader("Filtered Transactions Table")
if df_filt.empty:
    st.info("No rows to show.")
else:
    default_cols = [
        "transactiondate",
        "month_start",
        "year_month",
        "label",
        "customerregion",
        "productcategory",
        "retailchannel",
        "customergender",
        "customeragegroup",
        "customersatisfaction",
        "purchaseamount",
        "customerid",
        "transactionid",
    ]
    available_cols = [c for c in default_cols if c in df_filt.columns] + [c for c in df_filt.columns if c not in default_cols]
    cols_to_show = st.multiselect("Columns to display", options=available_cols, default=default_cols if set(default_cols).issubset(set(available_cols)) else available_cols[: min(12, len(available_cols))])

    st.dataframe(df_filt[cols_to_show], use_container_width=True)

    csv = df_filt[cols_to_show].to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download filtered data (CSV)",
        data=csv,
        file_name="nova_retail_filtered.csv",
        mime="text/csv",
    )
