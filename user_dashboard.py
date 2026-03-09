"""
pages/user_dashboard.py
-----------------------
User-facing dashboard: live grid parameters and AI distribution decisions.
All auth helpers are inlined to avoid Streamlit Cloud import path issues.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

from predict import predict, model_loaded
from database import log_prediction

# ── Inlined auth helpers ──────────────────────────────────────────────────────
def current_user():
    return st.session_state.get("username", "")

def logout():
    for key in ["logged_in", "username", "role", "user_id"]:
        st.session_state.pop(key, None)

def require_login():
    if not st.session_state.get("logged_in", False):
        st.warning("🔒 Please log in first.")
        st.stop()


def _gauge(value: float, title: str, color: str = "#00C49A", max_val: float = 100):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": title, "font": {"color": "#ccc", "size": 13}},
        gauge={
            "axis": {"range": [0, max_val], "tickcolor": "#555"},
            "bar":  {"color": color},
            "bgcolor": "#1a1a2e",
            "bordercolor": "#333",
            "steps": [
                {"range": [0,              max_val * 0.33], "color": "#0f0f1a"},
                {"range": [max_val * 0.33, max_val * 0.66], "color": "#16213e"},
                {"range": [max_val * 0.66, max_val],        "color": "#1a1a3e"},
            ],
        },
        number={"font": {"color": "#fff", "size": 26}},
    ))
    fig.update_layout(
        height=180, margin=dict(t=30, b=0, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)", font_color="#fff",
    )
    return fig


def render():
    require_login()

    # ── Sidebar ────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ☀️ Solar Microgrid AI")
        st.markdown(f"**User:** {current_user()}")
        st.markdown("---")
        st.markdown("Enter live system parameters and click **Predict** to receive an AI power distribution decision.")
        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True):
            logout()
            st.rerun()

    st.markdown(
        "<h1 style='color:#4FC3F7;font-family:monospace;'>⚡ Live Energy Dashboard</h1>",
        unsafe_allow_html=True,
    )

    if not model_loaded():
        st.error("⚠️ AI model not found. Please run `python train_model.py` first.")
        return

    # ── Input form ─────────────────────────────────────────────────────────────
    st.markdown("### 🎛️ System Parameters")
    with st.form("predict_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**☀️ Weather**")
            solar_irr    = st.slider("Solar Irradiance (W/m²)", 0.0, 1200.0, 400.0, 10.0)
            cloud_cover  = st.slider("Cloud Cover (%)",          0.0,  100.0,  50.0,  1.0)
            temperature  = st.slider("Temperature (°C)",        -10.0,  50.0,  25.0,  0.5)
            humidity     = st.slider("Humidity (%)",              0.0,  100.0,  60.0,  1.0)

        with col2:
            st.markdown("**🔋 Grid & Storage**")
            battery_level = st.slider("Battery Level (%)",    0.0, 100.0,  70.0, 1.0)
            grid_price    = st.slider("Grid Price ($/kWh)",   0.0,   1.0,   0.15, 0.01)
            solar_gen     = st.slider("Solar Generation (kW)", 0.0,  20.0,   5.0, 0.1)

        with col3:
            st.markdown("**🏗️ Load Demands (kW)**")
            hospital_load    = st.slider("Hospital Load",    0.0,  80.0, 30.0, 0.5)
            residential_load = st.slider("Residential Load", 0.0, 120.0, 60.0, 0.5)
            ev_load          = st.slider("EV Load",          0.0,  40.0, 10.0, 0.5)
            emergency_load   = st.slider("Emergency Load",   0.0,  40.0, 10.0, 0.5)

        total_load = hospital_load + residential_load + ev_load + emergency_load

        submitted = st.form_submit_button(
            "🔮 Predict Distribution Action",
            use_container_width=True,
            type="primary",
        )

    # ── Live gauges ────────────────────────────────────────────────────────────
    st.markdown("### 📊 Real-Time Metrics")
    g1, g2, g3, g4 = st.columns(4)
    g1.plotly_chart(_gauge(solar_gen,     "Solar Generation kW", "#FFD700", 20),  use_container_width=True)
    g2.plotly_chart(_gauge(battery_level, "Battery Level %",     "#00C49A", 100), use_container_width=True)
    g3.plotly_chart(_gauge(total_load,    "Total Load kW",       "#FF8C00", 250), use_container_width=True)
    g4.plotly_chart(_gauge(cloud_cover,   "Cloud Cover %",       "#4FC3F7", 100), use_container_width=True)

    # ── Load breakdown chart ───────────────────────────────────────────────────
    st.markdown("### 🏗️ Load Breakdown")
    load_df = pd.DataFrame({
        "Consumer":  ["Hospital", "Residential", "EV", "Emergency"],
        "Load (kW)": [hospital_load, residential_load, ev_load, emergency_load],
    })
    fig_bar = px.bar(
        load_df, x="Consumer", y="Load (kW)", color="Consumer",
        color_discrete_sequence=["#FF8C00","#AB47BC","#4FC3F7","#FF4444"],
        template="plotly_dark", text_auto=True,
    )
    fig_bar.update_layout(
        showlegend=False, paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", height=250,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # ── Prediction result ──────────────────────────────────────────────────────
    if submitted:
        result = predict(
            solar_irr, cloud_cover, temperature, humidity,
            battery_level, grid_price, solar_gen,
            hospital_load, residential_load, ev_load,
            emergency_load, total_load,
        )

        log_prediction(
            current_user(), solar_irr, cloud_cover,
            battery_level, total_load, result["action"],
        )

        color = result["color"]
        st.markdown("---")
        st.markdown(
            f"""
            <div style="
                background:linear-gradient(135deg,{color}22,{color}11);
                border:2px solid {color};border-radius:16px;padding:28px;
                text-align:center;box-shadow:0 0 30px {color}44;
            ">
                <div style="font-size:2.8rem;margin-bottom:8px;">{result['label']}</div>
                <div style="color:#ccc;font-size:1.05rem;">{result['description']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### 📈 Class Probabilities")
        prob_df = pd.DataFrame(
            list(result["probabilities"].items()),
            columns=["Action", "Probability (%)"],
        ).sort_values("Probability (%)", ascending=True)

        fig_prob = px.bar(
            prob_df, x="Probability (%)", y="Action", orientation="h",
            color="Probability (%)",
            color_continuous_scale=["#1a1a2e", "#00C49A"],
            template="plotly_dark", text_auto=True,
        )
        fig_prob.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            coloraxis_showscale=False, height=280,
        )
        st.plotly_chart(fig_prob, use_container_width=True)
