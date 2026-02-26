# app.py
import re
from typing import Dict, List, Tuple, Optional

import streamlit as st
import pandas as pd
import plotly.express as px


REQUIRED_LOGICAL_FIELDS: List[str] = [
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


def _normalize_colname(name: str) -> str:
    s = str(name).strip().lower()
    s = s.replace(" ", "_").replace("-", "_")
    s = re.sub(r"__+", "_", s)
    s = s.strip("_")
    return s


def _strip_underscores(s: str) -> str:
    return re.sub(r"_+", "", s)


def _build_column_map(df_cols: List[str]) -> Tuple[Dict[str, str], List[str]]:
    """
    Returns:
      col_map: logical_field -> actual_normalized_col_name
      missing: list of missing logical fields
    """
    cols_set = set(df_cols)
    cols_stripped_map: Dict[str, List[str]] = {}
    for c in df_cols:
        cols_stripped_map.setdefault(_strip_underscores(c), []).append(c)

    col_map: Dict[str, str] = {}
    missing: List[str] = []

    for req in REQUIRED_LOGICAL_FIELDS:
        if req in cols_set:
            col_map[req] = req
            continue

        req_stripped = _strip_underscores(req)
        candidates = cols_stripped_map.get(req_stripped, [])

        if len(candidates) == 1:
            col_map[req] = candidates[0]
            continue

        # Fallback scoring: prefer exact stripped match, then exact contains, then suffix/prefix.
        best: Optional[str] = None
        best_score = -1
        for c in df_cols:
            c_strip = _strip_underscores(c)
            score = 0
            if c_strip == req_stripped:
                score += 100
            if c == req:
                score += 50
            if req in c or c in req:
                score += 20
            if c.endswith(req):
                score += 10
            if c.startswith(req):
                score += 8
            # Prefer shorter names when tie (less likely to be a composite)
            score -= max(0, len(c) - len(req))
            if score > best_score:
                best_score = score
                best = c

        # Accept only if we have a reasonably confident match and uniqueness by stripped form
        if best is not None and best_score >= 80:
            col_map[req] = best
        else:
            missing.append(req)

    return col_map, missing


@st.cache_data(show_spinner=False)
def load_and_clean_data() -> Tuple[pd.DataFrame, Dict[str, str]]:
    try:
        raw = pd.read_excel("NR_dataset.xlsx")
    except FileNotFoundError:
        st.error("Dataset file not found in repository.")
        st.stop()
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")
        st.stop()

    if raw is None or raw.empty:
        st.error("Dataset loaded but appears to be empty.")
        st.stop()

    # Normalize column names
    norm_cols = [_normalize_colname(c) for c in raw.columns]
    df = raw.copy()
    df.columns = norm_cols

    col_map, missing = _build_column_map(df.columns.tolist())
    if missing:
        st.error(
            "Missing required logical fields: "
            + ", ".join(sorted(set(missing)))
        )
        st.write(df.columns)
        st.stop()

    # Standardize to canonical column names (rename)
    rename_dict = {col_map[k]: k for k in col_map}
    df = df.rename(columns=rename_dict)

    # Clean / type-cast
    dropped_msgs: List[str] = []

    # transactiondate -> datetime
    df["transactiondate"] = pd.to_datetime(df["transactiondate"], errors="coerce")
    bad_dates = int(df["transactiondate"].isna().sum())
    if bad_dates > 0:
        dropped_msgs.append(f"Dropped {bad_dates} rows with invalid transaction dates.")
        df = df.dropna(subset=["transactiondate"])

    # purchaseamount -> numeric; drop invalid
    df["purchaseamount"] = pd.to_numeric(df["purchaseamount"], errors="coerce")
    bad_amt = int(df["purchaseamount"].isna().sum())
    if bad_amt > 0:
        dropped_msgs.append(f"Dropped {bad_amt} rows with invalid purchase amounts.")
        df = df.dropna(subset=["purchaseamount"])

    # customersatisfaction -> numeric 1-5; drop invalid
    df["customersatisfaction"] = pd.to_numeric(df["customersatisfaction"], errors="coerce")
    invalid_sat = int((df["customersatisfaction"].isna() | ~df["customersatisfaction"].between(1, 5)).sum())
    if invalid_sat > 0:
        dropped_msgs.append(f"Dropped {invalid_sat} rows with invalid customer satisfaction (must be 1–5).")
        df = df[df["customersatisfaction"].between(1, 5)]

    # Drop missing critical fields for KPIs/charts
    critical = ["label", "customerregion", "retailchannel", "productcategory"]
    missing_critical = int(df[critical].isna().any(axis=1).sum())
    if missing_critical > 0:
        dropped_msgs.append(f"Dropped {missing_critical} rows missing critical fields for analysis.")
        df = df.dropna(subset=critical)

    # Ensure string-like categorical fields are clean
    for c in ["label", "productcategory", "customeragegroup", "customergender", "customerregion", "retailchannel"]:
        df[c] = df[c].astype(str).str.strip()

    # Derived field: year_month
    df["year_month"] = df["transactiondate"].dt.to_period("M").astype(str)

    # Show warnings (via return note pattern handled in caller)
    df.attrs["dropped_msgs"] = dropped_msgs
    return df, col_map


def _add_all_option(options: List[str]) -> List[str]:
    opts = [o for o in options if o is not None and str(o).strip() != ""]
    opts = sorted(pd.unique(pd.Series(opts)).tolist())
    return ["All"] + opts


def apply_filters(
    df: pd.DataFrame,
    segments: List[str],
    regions: List[str],
    categories: List[str],
    channels: List[str],
    genders: List[str],
    age_groups: List[str],
    date_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]],
) -> pd.DataFrame:
    out = df.copy()

    def _apply_multi(col: str, sel: List[str]) -> None:
        nonlocal out
        if sel and "All" not in sel:
            out = out[out[col].isin(sel)]

    _apply_multi("label", segments)
    _apply_multi("customerregion", regions)
    _apply_multi("productcategory", categories)
    _apply_multi("retailchannel", channels)
    _apply_multi("customergender", genders)
    _apply_multi("customeragegroup", age_groups)

    if date_range is not None:
        start_dt, end_dt = date_range
        if pd.notna(start_dt) and pd.notna(end_dt):
            start_dt = pd.to_datetime(start_dt).normalize()
            end_dt = pd.to_datetime(end_dt).normalize() + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            out = out[(out["transactiondate"] >= start_dt) & (out["transactiondate"] <= end_dt)]

    return out


st.set_page_config(layout="wide", page_title="NovaRetail Customer Intelligence Dashboard")

st.title("NovaRetail Customer Intelligence Dashboard")
st.subheader("Profitability, Retention Risk, and Satisfaction-Driven Performance")
st.write(
    "Explore revenue performance, customer segment health, satisfaction-driven patterns, and channel/product/region drivers. "
    "Use filters to isolate cohorts, monitor early warning signals, and surface commercial actions grounded in current data."
)

df, _col_map = load_and_clean_data()
for msg in df.attrs.get("dropped_msgs", []):
    st.warning(msg)

# Sidebar filters (dynamic, no hardcoding)
st.sidebar.header("Filters")

seg_options = _add_all_option(df["label"].dropna().astype(str).tolist())
reg_options = _add_all_option(df["customerregion"].dropna().astype(str).tolist())
cat_options = _add_all_option(df["productcategory"].dropna().astype(str).tolist())
chn_options = _add_all_option(df["retailchannel"].dropna().astype(str).tolist())
gen_options = _add_all_option(df["customergender"].dropna().astype(str).tolist())
age_options = _add_all_option(df["customeragegroup"].dropna().astype(str).tolist())

sel_segments = st.sidebar.multiselect("Segment", options=seg_options, default=["All"])
sel_regions = st.sidebar.multiselect("Region", options=reg_options, default=["All"])
sel_categories = st.sidebar.multiselect("Product Category", options=cat_options, default=["All"])
sel_channels = st.sidebar.multiselect("Channel", options=chn_options, default=["All"])
sel_genders = st.sidebar.multiselect("Gender", options=gen_options, default=["All"])
sel_age_groups = st.sidebar.multiselect("Age Group", options=age_options, default=["All"])

min_date = df["transactiondate"].min()
max_date = df["transactiondate"].max()
date_input = st.sidebar.date_input(
    "Date range",
    value=(min_date.date(), max_date.date()) if pd.notna(min_date) and pd.notna(max_date) else None,
)
date_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None
try:
    if isinstance(date_input, (list, tuple)) and len(date_input) == 2:
        date_range = (pd.to_datetime(date_input[0]), pd.to_datetime(date_input[1]))
except Exception:
    date_range = None

df_f = apply_filters(
    df=df,
    segments=sel_segments,
    regions=sel_regions,
    categories=sel_categories,
    channels=sel_channels,
    genders=sel_genders,
    age_groups=sel_age_groups,
    date_range=date_range,
)

if df_f.empty:
    st.warning("No data matches the current filters.")

# KPI Row
kpi_cols = st.columns(6)
total_revenue = float(df_f["purchaseamount"].sum()) if not df_f.empty else 0.0
total_txn = int(df_f["transactionid"].nunique()) if (not df_f.empty and "transactionid" in df_f.columns) else int(len(df_f))
unique_customers = int(df_f["customerid"].nunique()) if not df_f.empty else 0
aov = (total_revenue / total_txn) if total_txn > 0 else 0.0
avg_sat = float(df_f["customersatisfaction"].mean()) if (not df_f.empty and df_f["customersatisfaction"].notna().any()) else 0.0

if not df_f.empty and (df_f["label"] == "Decline").any():
    decline_share = float((df_f["label"] == "Decline").mean())
else:
    decline_share = 0.0

kpi_cols[0].metric("Total Revenue", f"${total_revenue:,.2f}")
kpi_cols[1].metric("Total Transactions", f"{total_txn:,}")
kpi_cols[2].metric("Unique Customers", f"{unique_customers:,}")
kpi_cols[3].metric("Average Order Value (AOV)", f"${aov:,.2f}")
kpi_cols[4].metric("Average Satisfaction", f"{avg_sat:,.2f}")
kpi_cols[5].metric("Decline Segment Share", f"{decline_share:.1%}")

st.divider()

# Visual layout
left, right = st.columns([1.2, 1.0], gap="large")

with left:
    st.subheader("Revenue Performance & Customer Health")

    if not df_f.empty:
        # Chart 1 — Revenue by Segment
        seg_rev = (
            df_f.groupby("label", as_index=False)["purchaseamount"]
            .sum()
            .sort_values("purchaseamount", ascending=False)
        )
        fig1 = px.bar(
            seg_rev,
            x="label",
            y="purchaseamount",
            title="Revenue by Customer Segment",
            labels={"label": "Segment", "purchaseamount": "Revenue"},
        )
        fig1.update_layout(showlegend=False)
        st.plotly_chart(fig1, use_container_width=True)

        # Chart 2 — Monthly Revenue Trend
        trend_color_by_segment = st.checkbox("Split trend by segment", value=True)
        trend = (
            df_f.groupby(["year_month"] + (["label"] if trend_color_by_segment else []), as_index=False)["purchaseamount"]
            .sum()
        )
        # Chronological ordering
        ym_sorted = sorted(df_f["year_month"].dropna().unique().tolist())
        fig2 = px.line(
            trend,
            x="year_month",
            y="purchaseamount",
            color="label" if trend_color_by_segment else None,
            title="Monthly Revenue Trend",
            category_orders={"year_month": ym_sorted},
            labels={"year_month": "Month", "purchaseamount": "Revenue", "label": "Segment"},
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Charts will appear when the filtered dataset is non-empty.")

with right:
    st.subheader("Satisfaction, Regional, Category & Channel Drivers")

    if not df_f.empty:
        # Chart 3 — Satisfaction vs Spend
        hover_cols = [c for c in ["customerid", "customerregion", "retailchannel", "productcategory"] if c in df_f.columns]
        fig3 = px.scatter(
            df_f,
            x="customersatisfaction",
            y="purchaseamount",
            color="label",
            title="Satisfaction vs Purchase Amount",
            labels={"customersatisfaction": "Satisfaction (1–5)", "purchaseamount": "Purchase Amount", "label": "Segment"},
            hover_data=hover_cols if hover_cols else None,
        )
        st.plotly_chart(fig3, use_container_width=True)

        # Chart 4 — Heatmap: Revenue by Region and Product Category
        pivot = (
            df_f.pivot_table(
                index="customerregion",
                columns="productcategory",
                values="purchaseamount",
                aggfunc="sum",
                fill_value=0.0,
            )
        )
        if pivot.shape[0] > 0 and pivot.shape[1] > 0:
            fig4 = px.imshow(
                pivot,
                aspect="auto",
                title="Revenue by Region and Product Category",
                labels={"x": "Product Category", "y": "Region", "color": "Revenue"},
            )
            st.plotly_chart(fig4, use_container_width=True)
        else:
            st.info("Not enough data to render the region/category heatmap.")

        # Chart 5 — Channel Mix
        ch_rev = df_f.groupby("retailchannel", as_index=False)["purchaseamount"].sum()
        fig5 = px.pie(
            ch_rev,
            names="retailchannel",
            values="purchaseamount",
            hole=0.5,
            title="Revenue by Retail Channel",
        )
        st.plotly_chart(fig5, use_container_width=True)
    else:
        st.info("Charts will appear when the filtered dataset is non-empty.")

st.divider()

# Insights & Actions
st.subheader("Insights & Recommended Actions")

if df_f.empty:
    st.write("No insights available because the current filters return no data.")
else:
    seg_rev = df_f.groupby("label", as_index=False)["purchaseamount"].sum().sort_values("purchaseamount", ascending=False)
    top_segment = seg_rev.iloc[0]["label"] if not seg_rev.empty else "N/A"

    region_sat = (
        df_f.groupby("customerregion", as_index=False)["customersatisfaction"]
        .mean()
        .dropna()
        .sort_values("customersatisfaction", ascending=True)
    )
    has_sat = not region_sat.empty

    region_decline = (
        df_f.assign(_is_decline=(df_f["label"] == "Decline"))
        .groupby("customerregion", as_index=False)["_is_decline"]
        .mean()
        .sort_values("_is_decline", ascending=False)
    )

    if has_sat:
        focus_region = region_sat.iloc[0]["customerregion"]
        focus_metric = f"lowest average satisfaction ({region_sat.iloc[0]['customersatisfaction']:.2f})"
    else:
        focus_region = region_decline.iloc[0]["customerregion"] if not region_decline.empty else "N/A"
        focus_metric = f"highest Decline share ({float(region_decline.iloc[0]['_is_decline']):.1%})" if not region_decline.empty else "N/A"

    top_channel = (
        df_f.groupby("retailchannel", as_index=False)["purchaseamount"].sum().sort_values("purchaseamount", ascending=False)
    )
    best_channel = top_channel.iloc[0]["retailchannel"] if not top_channel.empty else "N/A"

    st.write(f"Top revenue segment: **{top_segment}**.")
    st.write(f"Region to watch: **{focus_region}** ({focus_metric}).")
    st.write("Recommended actions:")
    st.write(f"• Prioritize retention and win-back offers for at-risk customers in **{focus_region}**.")
    st.write(f"• Protect revenue in **{top_segment}** with targeted upsell/cross-sell bundles aligned to top categories.")
    st.write(f"• Optimize experience in the **{best_channel}** channel by addressing satisfaction drivers and reducing friction.")
    st.write("• Monitor month-over-month changes in Decline share and satisfaction to catch early engagement drop-offs.")

st.divider()

# Filtered table (bottom)
st.subheader("Filtered Data")

if df_f.empty:
    st.write("No rows to display.")
else:
    preferred_order = [
        "idx",
        "customerid",
        "transactionid",
        "transactiondate",
        "year_month",
        "label",
        "productcategory",
        "purchaseamount",
        "customersatisfaction",
        "retailchannel",
        "customerregion",
        "customergender",
        "customeragegroup",
    ]
    cols_existing = [c for c in preferred_order if c in df_f.columns]
    remaining = [c for c in df_f.columns if c not in cols_existing]
    display_cols = cols_existing + remaining

    selected_cols = st.multiselect(
        "Select columns to display",
        options=display_cols,
        default=cols_existing if cols_existing else display_cols,
    )
    table_df = df_f[selected_cols].copy()
    st.dataframe(table_df.reset_index(drop=True), use_container_width=True)
