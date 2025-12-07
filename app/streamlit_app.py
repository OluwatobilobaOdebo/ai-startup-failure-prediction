import pandas as pd
import streamlit as st
import altair as alt
from pathlib import Path

# ---------- Load data ----------
@st.cache_data
def load_data():
    # Directory of this file: /mount/src/ai-startup-failure-prediction/app
    here = Path(__file__).parent

    # Repo root is one level up from app/
    repo_root = here.parent

    csv_path = repo_root / "data" / "processed" / "startup_risk_dashboard.csv"

    df = pd.read_csv(csv_path)
    return df

df = load_data()

# ---------- Page config ----------
st.set_page_config(
    page_title="AI Startup Failure Risk Dashboard",
    layout="wide",
)

st.title("AI Startup Failure Prediction Dashboard")

st.markdown(
    """
This dashboard uses a machine learning model (Random Forest) trained on historical startup data
to estimate the **probability of failure (closure)** for each startup.

**Personas:** VCs, angel investors, accelerators, founders.  
**Target:** 1 = Failure (closed), 0 = Success (acquired).
"""
)

# ---------- Sidebar filters ----------
st.sidebar.header("Filters")

# Region selector – we expect 'state_code' in the dashboard CSV
if "state_code" in df.columns:
    region_options = ["All"] + sorted(df["state_code"].dropna().unique().tolist())
    selected_region = st.sidebar.selectbox("State / Region", region_options, index=0)
else:
    st.sidebar.write("No state/region column found in data.")
    selected_region = "All"

risk_options = ["All", "Low", "Medium", "High"]
selected_risk = st.sidebar.selectbox("Risk bucket (RF)", risk_options, index=0)

filtered_df = df.copy()

if "state_code" in df.columns and selected_region != "All":
    filtered_df = filtered_df[filtered_df["state_code"] == selected_region]

if selected_risk != "All":
    filtered_df = filtered_df[filtered_df["risk_bucket_rf"] == selected_risk]

# If filters removed everything, show a friendly message and stop
if filtered_df.empty:
    st.warning("No startups match the current filters. Try relaxing the filters.")
    st.stop()

# ---------- Portfolio-level KPIs ----------
total_startups = len(filtered_df)
failure_rate_actual = filtered_df["target_failure"].mean() if total_startups > 0 else 0.0
avg_pred_failure = filtered_df["pred_failure_prob_rf"].mean() if total_startups > 0 else 0.0
high_risk_count = (filtered_df["risk_bucket_rf"] == "High").sum()

kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

kpi_col1.metric("Startups (filtered)", f"{total_startups:,}")
kpi_col2.metric("Actual failure rate", f"{failure_rate_actual:.1%}")
kpi_col3.metric("Avg predicted failure prob.", f"{avg_pred_failure:.1%}")
kpi_col4.metric("High-risk startups (RF)", f"{high_risk_count:,}")

# ---------- Portfolio charts ----------
if total_startups > 0:
    st.markdown("### Portfolio risk overview")

    # 1) Risk distribution (histogram of predicted failure prob)
    st.subheader("Risk distribution (predicted failure probability)")

    risk_hist = (
        alt.Chart(filtered_df)
        .mark_bar()
        .encode(
            x=alt.X(
                "pred_failure_prob_rf",
                bin=alt.Bin(maxbins=20),
                title="Predicted failure probability (RF)",
            ),
            y=alt.Y("count()", title="Number of startups"),
            tooltip=["count()"],
        )
        .properties(height=250)
    )

    st.altair_chart(risk_hist, use_container_width=True)

    # 2) Funding vs risk scatter
    st.subheader("Funding vs. predicted failure risk")

    scatter = (
        alt.Chart(filtered_df)
        .mark_circle(size=60, opacity=0.7)
        .encode(
            x=alt.X(
                "funding_total_usd",
                scale=alt.Scale(type="log"),
                title="Total funding (USD, log scale)",
            ),
            y=alt.Y(
                "pred_failure_prob_rf",
                title="Predicted failure probability (RF)",
            ),
            color=alt.Color(
                "risk_bucket_rf",
                title="Risk bucket (RF)",
            ),
            tooltip=[
                "name",
                "state_code",
                "funding_total_usd",
                alt.Tooltip("pred_failure_prob_rf", format=".1%"),
                "risk_bucket_rf",
            ],
        )
        .interactive()
        .properties(height=350)
    )

    st.altair_chart(scatter, use_container_width=True)

st.markdown("---")

# ---------- Single-startup view ----------
st.subheader("Startup-level Risk View")

startup_names = filtered_df["name"].dropna().unique().tolist()
selected_startup = st.selectbox("Select a startup", startup_names)

startup_row = filtered_df[filtered_df["name"] == selected_startup].iloc[0]

col_a, col_b, col_c = st.columns(3)

col_a.metric(
    "Predicted failure probability (RF)",
    f"{startup_row['pred_failure_prob_rf']:.1%}",
)

actual_label = "Failure (closed)" if startup_row["target_failure"] == 1 else "Success (acquired)"
col_b.metric("Actual status", actual_label)

col_c.metric("Risk bucket (RF)", startup_row["risk_bucket_rf"])

st.markdown("#### Context")

st.write(
    f"**State:** {startup_row['state_code']}  &nbsp;&nbsp; "
    f"**Funding rounds:** {int(startup_row['funding_rounds'])} &nbsp;&nbsp; "
    f"**Total funding (USD):** {startup_row['funding_total_usd']:,.0f}"
)

st.write(
    f"**Milestones:** {int(startup_row['milestones'])} &nbsp;&nbsp; "
    f"**Relationships:** {int(startup_row['relationships'])} &nbsp;&nbsp; "
    f"**Top 500 startup?** {'Yes' if startup_row['is_top500'] == 1 else 'No'}"
)

st.markdown("---")

# ---------- Underlying data table ----------
st.subheader("Underlying data (filtered)")

st.dataframe(
    filtered_df.sort_values("pred_failure_prob_rf", ascending=False),
    use_container_width=True,
)
