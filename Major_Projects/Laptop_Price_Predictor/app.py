
#Requires pipe.pkl and data.pkl in the same folder as this script.

import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import r2_score, mean_absolute_error

st.set_page_config(
    page_title="Laptop Price Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }

    :root {
        --accent: #6C5CE7;
        --accent-dark: #4834d4;
        --bg-soft: #F7F6FB;
    }

    .main {
        background-color: #FAFAFC;
    }

    /* Hero header */
    .hero {
        background: linear-gradient(120deg, #6C5CE7 0%, #4834d4 100%);
        padding: 2.2rem 2.2rem;
        border-radius: 18px;
        color: white;
        margin-bottom: 1.6rem;
        box-shadow: 0 10px 30px rgba(108, 92, 231, 0.25);
    }
    .hero h1 {
        font-weight: 800;
        font-size: 2.1rem;
        margin-bottom: 0.3rem;
        color: white;
    }
    .hero p {
        font-size: 1.0rem;
        opacity: 0.9;
        margin: 0;
    }

    /* Metric cards */
    .metric-card {
        background: white;
        border-radius: 14px;
        padding: 1.1rem 1.3rem;
        box-shadow: 0 2px 14px rgba(0,0,0,0.06);
        border: 1px solid #EFEDF7;
        text-align: center;
    }
    .metric-card .label {
        font-size: 0.8rem;
        color: #8A86A3;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .metric-card .value {
        font-size: 1.55rem;
        font-weight: 800;
        color: #2D2A4A;
        margin-top: 0.2rem;
    }

    /* Prediction result block */
    .predict-box {
        background: linear-gradient(135deg, #6C5CE7 0%, #4834d4 100%);
        border-radius: 18px;
        padding: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 10px 30px rgba(108, 92, 231, 0.3);
    }
    .predict-box .price {
        font-size: 3rem;
        font-weight: 800;
        margin: 0.3rem 0;
    }
    .predict-box .sub {
        opacity: 0.85;
        font-size: 0.95rem;
    }

    section[data-testid="stSidebar"] {
        background-color: #2D2A4A;
    }
    section[data-testid="stSidebar"] * {
        color: #F1F0FA !important;
    }

    .stButton>button {
        background: linear-gradient(135deg, #6C5CE7 0%, #4834d4 100%);
        color: white;
        border-radius: 10px;
        font-weight: 700;
        padding: 0.6rem 1.4rem;
        border: none;
        width: 100%;
    }
    .stButton>button:hover {
        opacity: 0.92;
        color: white;
    }

    .footer-note {
        text-align: center;
        color: #A6A2BD;
        font-size: 0.8rem;
        margin-top: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

import os
import pickle
import streamlit as st

@st.cache_resource
def load_pipeline():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pipe_path = os.path.join(current_dir, "pipe.pkl")

    with open(pipe_path, "rb") as f:
        pipe = pickle.load(f)

    return pipe


@st.cache_data
def load_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "data.pkl")

    with open(data_path, "rb") as f:
        df = pickle.load(f)

    return df


pipe = load_pipeline()
df = load_data()
RESOLUTIONS = [
    "1920x1080", "1366x768", "1600x900", "3840x2160",
    "3200x1800", "2880x1800", "2560x1600", "2560x1440", "2304x1440",
]

@st.cache_data
def model_fit_stats(_pipe, data):
    X = data.drop(columns=["Price"])
    preds = np.exp(_pipe.predict(X))
    actual = data["Price"]
    r2 = r2_score(actual, preds)
    mae = mean_absolute_error(actual, preds)
    return r2, mae, preds

r2, mae, fitted_preds = model_fit_stats(pipe, df)

# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## Laptop Price Predictor")
    st.caption("ML dashboard powered by a Random Forest regression pipeline.")
    st.markdown("---")
    st.markdown("### Dataset Snapshot")
    st.metric("Laptops in dataset", f"{len(df):,}")
    st.metric("Brands covered", df["Company"].nunique())
    st.metric("Avg. price", f"₹{df['Price'].mean():,.0f}")
    st.markdown("---")
    st.markdown("### Model")
    st.write("**Algorithm:** Random Forest Regressor")
    st.write("**Preprocessing:** One-Hot Encoding + passthrough numerics")
    st.write("**Target:** log(Price), exponentiated back at inference")
    st.markdown("---")
    st.caption(
        "Fit quality shown on the Insights tab is computed on the full "
        "dataset (not a held-out test split) — treat it as an indicative, "
        "optimistic upper bound."
    )

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>Laptop Price Predictor</h1>
        <p>Configure a laptop's specs and get an instant price estimate, backed by a model trained on 1,300+ real listings.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_predict, tab_insights = st.tabs(["Predict", "Market Insights"])

# ============================================================================
# TAB 1 — PREDICT
# ============================================================================
with tab_predict:
    col_form, col_result = st.columns([1.3, 1], gap="large")

    with col_form:
        st.markdown("#### Configure your laptop")

        c1, c2 = st.columns(2)
        with c1:
            company = st.selectbox("Brand", sorted(df["Company"].unique()))
            laptop_type = st.selectbox("Type", sorted(df["TypeName"].unique()))
            ram_options = sorted(set(int(v) for v in df["Ram"].unique()))
            ram = st.selectbox(
                "RAM (GB)", ram_options, index=2,
                format_func=lambda x: f"{x} GB",
            )
            weight = st.number_input(
                "Weight (kg)", min_value=0.5, max_value=5.0,
                value=2.0, step=0.1,
            )
            cpu_brand = st.selectbox("CPU", sorted(df["Cpu brand"].unique()))
        with c2:
            gpu_brand = st.selectbox("GPU", sorted(df["Gpu brand"].unique()))
            os_choice = st.selectbox("Operating System", sorted(df["os"].unique()))
            hdd_options = sorted(set(int(v) for v in df["HDD"].unique()))
            hdd = st.selectbox(
                "HDD (GB)", hdd_options,
                format_func=lambda x: "No HDD" if x == 0 else f"{x} GB",
            )
            ssd_options = sorted(set(int(v) for v in df["SSD"].unique()))
            ssd = st.selectbox(
                "SSD (GB)", ssd_options, index=4,
                format_func=lambda x: "No SSD" if x == 0 else f"{x} GB",
            )

        st.markdown("##### Display")
        d1, d2 = st.columns(2)
        with d1:
            screen_size = st.slider(
                "Screen size (inches)",
                float(df["Inches"].min()), float(df["Inches"].max()),
                15.6, step=0.1,
            )
            resolution = st.selectbox("Resolution", RESOLUTIONS)
        with d2:
            touchscreen = st.radio("Touchscreen", ["No", "Yes"], horizontal=True)
            ips = st.radio("IPS Display", ["No", "Yes"], horizontal=True)

        predict_clicked = st.button("Predict Price", use_container_width=True)

    with col_result:
        st.markdown("#### Estimate")
        if predict_clicked:
            x_res, y_res = (int(v) for v in resolution.split("x"))
            ppi = ((x_res ** 2 + y_res ** 2) ** 0.5) / screen_size

            query = pd.DataFrame(
                [{
                    "Company": company,
                    "TypeName": laptop_type,
                    "Inches": screen_size,
                    "Ram": ram,
                    "Weight": weight,
                    "Touchscreen": 1 if touchscreen == "Yes" else 0,
                    "IPS": 1 if ips == "Yes" else 0,
                    "ppi": ppi,
                    "Cpu brand": cpu_brand,
                    "HDD": hdd,
                    "SSD": ssd,
                    "Gpu brand": gpu_brand,
                    "os": os_choice,
                }]
            )

            log_price = pipe.predict(query)[0]
            price = float(np.exp(log_price))
            low, high = price * 0.9, price * 1.1

            st.markdown(
                f"""
                <div class="predict-box">
                    <div class="sub">Estimated Price</div>
                    <div class="price">₹{price:,.0f}</div>
                    <div class="sub">Likely range: ₹{low:,.0f} – ₹{high:,.0f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("<br>", unsafe_allow_html=True)

            similar = df[
                (df["Company"] == company) & (df["TypeName"] == laptop_type)
            ].copy()
            similar["diff"] = (similar["Price"] - price).abs()
            similar = similar.sort_values("diff").head(5)

            if not similar.empty:
                st.markdown("##### Closest matches in the dataset")
                show_cols = ["Company", "TypeName", "Ram", "SSD", "HDD", "Price"]
                st.dataframe(
                    similar[show_cols].rename(columns={"Price": "Price (₹)"}),
                    use_container_width=True,
                    hide_index=True,
                )
        else:
            st.info("Fill in the specs and hit **Predict Price** to see your estimate here.")

# ============================================================================
# TAB 2 — INSIGHTS
# ============================================================================
with tab_insights:
    m1, m2, m3, m4 = st.columns(4)
    metrics = [
        ("Total Laptops", f"{len(df):,}"),
        ("Avg. Price", f"₹{df['Price'].mean():,.0f}"),
        ("Median Price", f"₹{df['Price'].median():,.0f}"),
        ("Price Range", f"₹{df['Price'].min():,.0f} – ₹{df['Price'].max():,.0f}"),
    ]
    for col, (label, value) in zip([m1, m2, m3, m4], metrics):
        col.markdown(
            f"""<div class="metric-card">
                    <div class="label">{label}</div>
                    <div class="value">{value}</div>
                </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    row1c1, row1c2 = st.columns(2)

    with row1c1:
        avg_by_brand = (
            df.groupby("Company")["Price"].mean().sort_values(ascending=False)
        )
        fig = px.bar(
            avg_by_brand, orientation="h",
            title="Average Price by Brand",
            labels={"value": "Avg. Price (₹)", "Company": ""},
            color=avg_by_brand.values, color_continuous_scale="Purples",
        )
        fig.update_layout(showlegend=False, coloraxis_showscale=False, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

    with row1c2:
        avg_by_type = (
            df.groupby("TypeName")["Price"].mean().sort_values(ascending=False)
        )
        fig = px.bar(
            avg_by_type,
            title="Average Price by Laptop Type",
            labels={"value": "Avg. Price (₹)", "TypeName": ""},
            color=avg_by_type.values, color_continuous_scale="Purples",
        )
        fig.update_layout(showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    row2c1, row2c2 = st.columns(2)

    with row2c1:
        fig = px.histogram(
            df, x="Price", nbins=40,
            title="Price Distribution",
            color_discrete_sequence=["#6C5CE7"],
        )
        fig.update_layout(yaxis_title="Count", xaxis_title="Price (₹)")
        st.plotly_chart(fig, use_container_width=True)

    with row2c2:
        avg_by_ram = df.groupby("Ram")["Price"].mean().reset_index()
        fig = px.line(
            avg_by_ram, x="Ram", y="Price", markers=True,
            title="Average Price vs RAM",
            labels={"Ram": "RAM (GB)", "Price": "Avg. Price (₹)"},
            color_discrete_sequence=["#4834d4"],
        )
        st.plotly_chart(fig, use_container_width=True)

    row3c1, row3c2 = st.columns(2)

    with row3c1:
        premium_df = pd.DataFrame({
            "Feature": ["Touchscreen", "Touchscreen", "IPS Display", "IPS Display"],
            "Has Feature": ["No", "Yes", "No", "Yes"],
            "Avg Price": [
                df[df.Touchscreen == 0]["Price"].mean(),
                df[df.Touchscreen == 1]["Price"].mean(),
                df[df.IPS == 0]["Price"].mean(),
                df[df.IPS == 1]["Price"].mean(),
            ],
        })
        fig = px.bar(
            premium_df, x="Feature", y="Avg Price", color="Has Feature",
            barmode="group", title="Price Premium: Touchscreen & IPS",
            color_discrete_sequence=["#D6D2F0", "#6C5CE7"],
        )
        st.plotly_chart(fig, use_container_width=True)

    with row3c2:
        avg_by_cpu = (
            df.groupby("Cpu brand")["Price"].mean().sort_values(ascending=False)
        )
        fig = px.bar(
            avg_by_cpu,
            title="Average Price by CPU",
            labels={"value": "Avg. Price (₹)", "Cpu brand": ""},
            color=avg_by_cpu.values, color_continuous_scale="Purples",
        )
        fig.update_layout(showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Model Fit")
    f1, f2 = st.columns([1, 2])
    with f1:
        st.markdown(
            f"""<div class="metric-card">
                    <div class="label">R² (full dataset)</div>
                    <div class="value">{r2:.3f}</div>
                </div><br>
                <div class="metric-card">
                    <div class="label">MAE (full dataset)</div>
                    <div class="value">₹{mae:,.0f}</div>
                </div>""",
            unsafe_allow_html=True,
        )
        st.caption("Computed on the full dataset the model was trained on — not a held-out test set, so treat this as optimistic.")
    with f2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["Price"], y=fitted_preds, mode="markers",
            marker=dict(color="#6C5CE7", size=6, opacity=0.5),
            name="Predicted vs Actual",
        ))
        lims = [df["Price"].min(), df["Price"].max()]
        fig.add_trace(go.Scatter(
            x=lims, y=lims, mode="lines",
            line=dict(color="#2D2A4A", dash="dash"), name="Perfect fit",
        ))
        fig.update_layout(
            title="Predicted vs Actual Price",
            xaxis_title="Actual Price (₹)", yaxis_title="Predicted Price (₹)",
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

st.markdown(
    '<div class="footer-note">Built with Streamlit · Random Forest Regression · Data-driven laptop pricing</div>',
    unsafe_allow_html=True,
)
