"""
Customer Segmentation & Churn Pattern Analytics in European Banking
Streamlit dashboard  |  Data: European_Bank.csv (10,000 customers)
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="European Banking - Churn Analytics",
    page_icon="🏦",
    layout="wide",
)

DATA_FILE = Path(__file__).parent / "European_Bank.csv"

AGE_ORDER = ["<30", "30-45", "46-60", "60+"]
CREDIT_ORDER = ["Low", "Medium", "High"]
TENURE_ORDER = ["New", "Mid-term", "Long-term"]
BALANCE_ORDER = ["Zero", "Low", "High"]
PRODUCT_ORDER = ["1", "2", "3", "4"]
RED = "#D64545"
GREY = "#8A94A6"


# ----------------------------------------------------------------------------
# DATA
# ----------------------------------------------------------------------------
@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_FILE)

    # Cleaning: drop non-analytical fields
    df = df.drop(columns=[c for c in ["Surname", "Year"] if c in df.columns])

    # Derived segmentation fields
    df["AgeGroup"] = pd.cut(
        df["Age"], bins=[0, 29, 45, 60, 200], labels=AGE_ORDER
    ).astype(str)
    df["CreditBand"] = pd.cut(
        df["CreditScore"], bins=[0, 579, 669, 900], labels=CREDIT_ORDER
    ).astype(str)
    df["TenureGroup"] = pd.cut(
        df["Tenure"], bins=[-1, 2, 6, 10], labels=TENURE_ORDER
    ).astype(str)

    median_pos_balance = df.loc[df["Balance"] > 0, "Balance"].median()
    df["BalanceSegment"] = np.where(
        df["Balance"] == 0,
        "Zero",
        np.where(df["Balance"] < median_pos_balance, "Low", "High"),
    )
    df["Products"] = df["NumOfProducts"].astype(str)
    df["Status"] = np.where(df["Exited"] == 1, "Churned", "Retained")
    df["Activity"] = np.where(df["IsActiveMember"] == 1, "Active", "Inactive")
    return df


full = load_data()
BASE_CHURN = full["Exited"].mean() * 100  # bank-wide churn, used as the baseline
DEFAULT_HV_PCT = 75  # "high-value" = top 25% of balances


def seg_table(d: pd.DataFrame, col: str, order=None) -> pd.DataFrame:
    t = (
        d.groupby(col)
        .agg(Customers=("Exited", "size"), Churned=("Exited", "sum"))
        .reset_index()
    )
    t["ChurnRate"] = t["Churned"] / t["Customers"] * 100
    if order:
        t[col] = pd.Categorical(t[col], categories=order, ordered=True)
        t = t.sort_values(col)
        t[col] = t[col].astype(str)
    return t


def churn_bar(d: pd.DataFrame, col: str, order=None, title="", baseline=None):
    t = seg_table(d, col, order)
    t["Label"] = t["ChurnRate"].round(1).astype(str) + "%"
    fig = px.bar(
        t,
        x=col,
        y="ChurnRate",
        text="Label",
        hover_data={"Customers": True, "Churned": True, "ChurnRate": ":.1f"},
        category_orders={col: order} if order else None,
    )
    fig.update_traces(marker_color=RED, textposition="outside")
    if baseline is not None:
        fig.add_hline(
            y=baseline,
            line_dash="dash",
            line_color=GREY,
            annotation_text=f"Overall {baseline:.1f}%",
        )
    fig.update_layout(
        title=title,
        yaxis_title="Churn rate (%)",
        xaxis_title="",
        showlegend=False,
        margin=dict(t=60, b=20),
    )
    return fig


# ----------------------------------------------------------------------------
# SIDEBAR FILTERS (segment filters)
# ----------------------------------------------------------------------------
st.sidebar.title("🔎 Segment filters")


def opts(col, order=None):
    vals = list(full[col].unique())
    return [v for v in order if v in vals] if order else sorted(vals)


geo_sel = st.sidebar.multiselect("Geography", opts("Geography"), default=opts("Geography"))
gen_sel = st.sidebar.multiselect("Gender", opts("Gender"), default=opts("Gender"))
age_sel = st.sidebar.multiselect("Age group", opts("AgeGroup", AGE_ORDER), default=opts("AgeGroup", AGE_ORDER))
cred_sel = st.sidebar.multiselect("Credit score band", opts("CreditBand", CREDIT_ORDER), default=opts("CreditBand", CREDIT_ORDER))
ten_sel = st.sidebar.multiselect("Tenure group", opts("TenureGroup", TENURE_ORDER), default=opts("TenureGroup", TENURE_ORDER))
bal_sel = st.sidebar.multiselect("Balance segment", opts("BalanceSegment", BALANCE_ORDER), default=opts("BalanceSegment", BALANCE_ORDER))

df = full[
    full["Geography"].isin(geo_sel)
    & full["Gender"].isin(gen_sel)
    & full["AgeGroup"].isin(age_sel)
    & full["CreditBand"].isin(cred_sel)
    & full["TenureGroup"].isin(ten_sel)
    & full["BalanceSegment"].isin(bal_sel)
]

st.sidebar.markdown("---")
st.sidebar.caption(f"Showing **{len(df):,}** of {len(full):,} customers")

# ----------------------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------------------
st.title("🏦 Customer Segmentation & Churn Pattern Analytics")
st.caption("European retail banking  |  France, Spain, Germany  |  10,000 customers")

if df.empty:
    st.warning("No customers match the selected filters. Please widen the filters in the sidebar.")
    st.stop()

# ----------------------------------------------------------------------------
# KPI CALCULATIONS
# ----------------------------------------------------------------------------
hv_threshold = full["Balance"].quantile(DEFAULT_HV_PCT / 100)

overall_churn = df["Exited"].mean() * 100

hv = df[df["Balance"] >= hv_threshold]
hv_ratio = hv["Exited"].mean() * 100 if len(hv) else np.nan

geo_t = seg_table(df, "Geography")
geo_t["RiskIndex"] = geo_t["ChurnRate"] / BASE_CHURN * 100
top_geo = geo_t.sort_values("RiskIndex", ascending=False).iloc[0]

inactive = df[df["IsActiveMember"] == 0]
active = df[df["IsActiveMember"] == 1]
if len(inactive) and len(active):
    engagement_drop = (inactive["Exited"].mean() - active["Exited"].mean()) * 100
else:
    engagement_drop = np.nan

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Overall Churn Rate", f"{overall_churn:.1f}%", f"{overall_churn - BASE_CHURN:+.1f} pts vs bank avg", delta_color="inverse")
k2.metric("Customers / Churned", f"{len(df):,}", f"{int(df['Exited'].sum()):,} churned", delta_color="off")
k3.metric("High-Value Churn Ratio", "n/a" if np.isnan(hv_ratio) else f"{hv_ratio:.1f}%", f"Balance ≥ {hv_threshold:,.0f}", delta_color="off")
k4.metric("Geographic Risk Index", f"{top_geo['RiskIndex']:.0f}", f"{top_geo['Geography']} (100 = bank avg)", delta_color="off")
k5.metric("Engagement Drop Indicator", "n/a" if np.isnan(engagement_drop) else f"{engagement_drop:.1f} pts", "Inactive vs active churn gap", delta_color="off")

with st.expander("How are these KPIs defined?"):
    st.markdown(
        f"""
- **Overall Churn Rate** = churned customers / total customers in the current selection.
- **Segment Churn Rate** = churn % within each segment (shown in every chart).
- **High-Value Churn Ratio** = churn % among customers whose balance is in the top 25% of the bank (balance ≥ {hv_threshold:,.0f}).
- **Geographic Risk Index** = a country's churn rate / bank-wide churn rate x 100 (100 = average, above 100 = riskier). The riskiest country in the current selection is shown.
- **Engagement Drop Indicator** = churn rate of inactive members minus churn rate of active members, in percentage points.
        """
    )

tab_over, tab_geo, tab_age, tab_hv, tab_drill, tab_data = st.tabs(
    ["📊 Overview", "🌍 Geography", "🎂 Age & Tenure", "💎 High-Value Explorer", "🔬 Drill-down", "🗂 Data"]
)

# ----------------------------------------------------------------------------
# TAB 1: OVERVIEW
# ----------------------------------------------------------------------------
with tab_over:
    c1, c2 = st.columns(2)
    with c1:
        status = df["Status"].value_counts().reset_index()
        status.columns = ["Status", "Customers"]
        fig = px.pie(
            status,
            names="Status",
            values="Customers",
            hole=0.55,
            color="Status",
            color_discrete_map={"Churned": RED, "Retained": "#4C9F70"},
            title="Churned vs retained customers",
        )
        st.plotly_chart(fig)
    with c2:
        st.plotly_chart(
            churn_bar(df, "Gender", title="Churn rate by gender", baseline=overall_churn),
        )

    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(
            churn_bar(df, "Products", PRODUCT_ORDER, "Churn rate by number of products", overall_churn),
        )
    with c4:
        st.plotly_chart(
            churn_bar(df, "Activity", None, "Churn rate: active vs inactive members", overall_churn),
        )

    st.subheader("Churned vs retained: average profile")
    prof = (
        df.groupby("Status")[["CreditScore", "Age", "Tenure", "Balance", "EstimatedSalary", "NumOfProducts"]]
        .mean()
        .round(1)
        .T
    )
    st.dataframe(prof)

# ----------------------------------------------------------------------------
# TAB 2: GEOGRAPHY
# ----------------------------------------------------------------------------
with tab_geo:
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            churn_bar(df, "Geography", None, "Churn rate by country", overall_churn),
        )
    with c2:
        piv = df.pivot_table(index="Geography", columns="AgeGroup", values="Exited", aggfunc="mean") * 100
        piv = piv.reindex(columns=[a for a in AGE_ORDER if a in piv.columns])
        fig = px.imshow(
            piv,
            text_auto=".1f",
            color_continuous_scale="Reds",
            aspect="auto",
            title="Churn % by country x age group",
        )
        st.plotly_chart(fig)

    st.subheader("Country summary")
    geo_show = geo_t.copy()
    geo_show["ChurnRate"] = geo_show["ChurnRate"].round(1)
    geo_show["RiskIndex"] = geo_show["RiskIndex"].round(0)
    st.dataframe(geo_show, hide_index=True)

    bal_geo = df.groupby("Geography").agg(
        AvgBalance=("Balance", "mean"),
        ZeroBalancePct=("Balance", lambda s: (s == 0).mean() * 100),
        InactivePct=("IsActiveMember", lambda s: (s == 0).mean() * 100),
    ).round(1).reset_index()
    st.caption("Financial and engagement profile by country")
    st.dataframe(bal_geo, hide_index=True)

# ----------------------------------------------------------------------------
# TAB 3: AGE & TENURE
# ----------------------------------------------------------------------------
with tab_age:
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            churn_bar(df, "AgeGroup", AGE_ORDER, "Churn rate by age group", overall_churn),
        )
    with c2:
        st.plotly_chart(
            churn_bar(df, "TenureGroup", TENURE_ORDER, "Churn rate by tenure group", overall_churn),
        )

    c3, c4 = st.columns(2)
    with c3:
        fig = px.histogram(
            df,
            x="Age",
            color="Status",
            nbins=40,
            barmode="overlay",
            opacity=0.65,
            color_discrete_map={"Churned": RED, "Retained": "#4C9F70"},
            title="Age distribution: churned vs retained",
        )
        st.plotly_chart(fig)
    with c4:
        st.plotly_chart(
            churn_bar(df, "CreditBand", CREDIT_ORDER, "Churn rate by credit score band", overall_churn),
        )

# ----------------------------------------------------------------------------
# TAB 4: HIGH-VALUE EXPLORER
# ----------------------------------------------------------------------------
with tab_hv:
    st.markdown("Define *high-value* by account balance and see how churn behaves for those customers.")
    pct = st.slider("High-value = balance in the top ...% of all bank customers", 5, 50, 100 - DEFAULT_HV_PCT, 5)
    thr = full["Balance"].quantile(1 - pct / 100)
    st.caption(f"Balance threshold: **{thr:,.0f}**")

    d = df.copy()
    d["Tier"] = np.where(d["Balance"] >= thr, "High-value", "Other customers")
    hv_d = d[d["Tier"] == "High-value"]

    if hv_d.empty:
        st.info("No high-value customers in the current selection.")
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("High-value customers", f"{len(hv_d):,}")
        m2.metric("High-value churn rate", f"{hv_d['Exited'].mean() * 100:.1f}%")
        lost = hv_d.loc[hv_d["Exited"] == 1, "Balance"].sum()
        m3.metric("Balance lost (high-value)", f"{lost:,.0f}")
        total_lost = d.loc[d["Exited"] == 1, "Balance"].sum()
        share = lost / total_lost * 100 if total_lost else 0
        m4.metric("Share of all lost balance", f"{share:.1f}%")

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(
                churn_bar(d, "Tier", None, "Churn rate: high-value vs others", overall_churn),
            )
        with c2:
            sample = d.sample(min(3000, len(d)), random_state=1)
            fig = px.scatter(
                sample,
                x="EstimatedSalary",
                y="Balance",
                color="Status",
                opacity=0.6,
                color_discrete_map={"Churned": RED, "Retained": "#4C9F70"},
                title="Salary vs balance (sample of customers)",
            )
            st.plotly_chart(fig)

        st.subheader("High-value churn by country")
        st.plotly_chart(
            churn_bar(hv_d, "Geography", None, "High-value churn rate by country", hv_d["Exited"].mean() * 100),
        )

# ----------------------------------------------------------------------------
# TAB 5: DRILL-DOWN
# ----------------------------------------------------------------------------
with tab_drill:
    st.markdown("Pick a segment, then break it down further to find where churn concentrates.")
    dims = {
        "Geography": None,
        "Gender": None,
        "Age group": AGE_ORDER,
        "Credit band": CREDIT_ORDER,
        "Tenure group": TENURE_ORDER,
        "Balance segment": BALANCE_ORDER,
        "Products": PRODUCT_ORDER,
        "Activity": None,
    }
    colmap = {
        "Geography": "Geography",
        "Gender": "Gender",
        "Age group": "AgeGroup",
        "Credit band": "CreditBand",
        "Tenure group": "TenureGroup",
        "Balance segment": "BalanceSegment",
        "Products": "Products",
        "Activity": "Activity",
    }
    c1, c2, c3 = st.columns(3)
    dim1 = c1.selectbox("Segment by", list(dims.keys()), index=0)
    present = set(df[colmap[dim1]].unique())
    values = [v for v in dims[dim1] if v in present] if dims[dim1] else sorted(present)
    val1 = c2.selectbox("Value", values)
    other_dims = [k for k in dims if k != dim1]
    dim2 = c3.selectbox("Then break down by", other_dims, index=min(2, len(other_dims) - 1))

    seg = df[df[colmap[dim1]] == val1]
    if seg.empty:
        st.info("No customers in this segment.")
    else:
        s1, s2, s3 = st.columns(3)
        s1.metric(f"{dim1}: {val1}", f"{len(seg):,} customers")
        s2.metric("Segment churn rate", f"{seg['Exited'].mean() * 100:.1f}%", f"{seg['Exited'].mean() * 100 - overall_churn:+.1f} pts vs selection", delta_color="inverse")
        s3.metric("Churned customers", f"{int(seg['Exited'].sum()):,}")
        st.plotly_chart(
            churn_bar(seg, colmap[dim2], dims[dim2], f"{val1}: churn rate by {dim2.lower()}", seg["Exited"].mean() * 100),
        )

# ----------------------------------------------------------------------------
# TAB 6: DATA
# ----------------------------------------------------------------------------
with tab_data:
    st.dataframe(df.head(500), hide_index=True)
    st.download_button(
        "Download filtered data (CSV)",
        df.to_csv(index=False).encode("utf-8"),
        file_name="filtered_customers.csv",
        mime="text/csv",
    )
