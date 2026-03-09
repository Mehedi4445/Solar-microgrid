"""
pages/admin_dashboard.py
------------------------
Admin-only dashboard: dataset stats, prediction logs, and user management.
All auth helpers are inlined to avoid Streamlit Cloud import path issues.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from database import get_all_users, delete_user, add_user, get_prediction_logs

# ── Inlined auth helpers ──────────────────────────────────────────────────────
def current_user():
    return st.session_state.get("username", "")

def current_role():
    return st.session_state.get("role", "")

def logout():
    for key in ["logged_in", "username", "role", "user_id"]:
        st.session_state.pop(key, None)

def require_admin():
    if not st.session_state.get("logged_in", False):
        st.warning("🔒 Please log in first.")
        st.stop()
    if current_role() != "admin":
        st.error("⛔ Admin access required.")
        st.stop()


def render():
    require_admin()

    # ── Sidebar ────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ☀️ Solar Microgrid AI")
        st.markdown(f"**Admin:** {current_user()}")
        st.markdown("---")
        section = st.radio(
            "Navigation",
            ["📊 Dataset Overview", "🔮 Prediction Logs", "👥 User Management"],
            label_visibility="collapsed",
        )
        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True):
            logout()
            st.rerun()

    st.markdown(
        "<h1 style='color:#00C49A;font-family:monospace;'>⚡ Admin Control Panel</h1>",
        unsafe_allow_html=True,
    )

    # ── Dataset Overview ───────────────────────────────────────────────────────
    if "Dataset" in section:
        st.subheader("📊 Dataset Statistics")
        try:
            df = pd.read_csv("solar_microgrid_ai_dataset.csv")
        except FileNotFoundError:
            st.error("Dataset file not found.")
            return

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Records", f"{len(df):,}")
        col2.metric("Features", len(df.columns) - 1)
        col3.metric("Classes", df["Distribution_Action"].nunique())
        col4.metric("Avg Solar kW", f"{df['Solar_Generation_kW'].mean():.2f}")

        st.markdown("#### Class Distribution")
        counts = df["Distribution_Action"].value_counts().reset_index()
        counts.columns = ["Action", "Count"]
        fig = px.bar(
            counts, x="Action", y="Count", color="Action",
            color_discrete_sequence=["#00C49A","#FF8C00","#4FC3F7","#AB47BC","#FF4444"],
            template="plotly_dark",
        )
        fig.update_layout(showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### Solar Irradiance vs Generation")
            fig2 = px.scatter(
                df.sample(500, random_state=42),
                x="Solar_Irradiance", y="Solar_Generation_kW",
                color="Distribution_Action", opacity=0.7,
                template="plotly_dark",
                color_discrete_sequence=["#00C49A","#FF8C00","#4FC3F7","#AB47BC","#FF4444"],
            )
            fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig2, use_container_width=True)

        with col_b:
            st.markdown("#### Battery Level Distribution")
            fig3 = px.histogram(
                df, x="Battery_Level", nbins=30, color="Distribution_Action",
                template="plotly_dark",
                color_discrete_sequence=["#00C49A","#FF8C00","#4FC3F7","#AB47BC","#FF4444"],
            )
            fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig3, use_container_width=True)

        st.markdown("#### Descriptive Statistics")
        st.dataframe(df.describe().round(2), use_container_width=True)

    # ── Prediction Logs ────────────────────────────────────────────────────────
    elif "Prediction" in section:
        st.subheader("🔮 AI Prediction Logs")
        logs = get_prediction_logs(100)
        if not logs:
            st.info("No predictions recorded yet.")
        else:
            log_df = pd.DataFrame(
                [dict(r) for r in logs],
                columns=["id","username","timestamp","solar_irr","cloud_cover",
                         "battery_level","total_load","prediction"],
            )
            st.dataframe(log_df, use_container_width=True)

            st.markdown("#### Prediction Frequency")
            freq = log_df["prediction"].value_counts().reset_index()
            freq.columns = ["Action", "Count"]
            fig4 = px.pie(
                freq, names="Action", values="Count",
                color_discrete_sequence=["#00C49A","#FF8C00","#4FC3F7","#AB47BC","#FF4444"],
                template="plotly_dark",
            )
            fig4.update_layout(paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig4, use_container_width=True)

    # ── User Management ────────────────────────────────────────────────────────
    elif "User" in section:
        st.subheader("👥 User Management")

        with st.expander("➕ Add New User"):
            with st.form("add_user_form"):
                new_user = st.text_input("Username")
                new_pass = st.text_input("Password", type="password")
                new_role = st.selectbox("Role", ["user", "admin"])
                submitted = st.form_submit_button("Create User")
                if submitted:
                    if new_user and new_pass:
                        ok = add_user(new_user, new_pass, new_role)
                        if ok:
                            st.success(f"User '{new_user}' created successfully.")
                        else:
                            st.error("Username already exists.")
                    else:
                        st.warning("Please fill in all fields.")

        st.markdown("#### Current Users")
        users = get_all_users()
        for u in users:
            c1, c2, c3, c4 = st.columns([1, 3, 2, 2])
            c1.write(u["id"])
            c2.write(u["username"])
            c3.write(f"🏷️ {u['role']}")
            if u["username"] != current_user():
                if c4.button("🗑️ Delete", key=f"del_{u['id']}"):
                    delete_user(u["id"])
                    st.success(f"User '{u['username']}' deleted.")
                    st.rerun()
            else:
                c4.write("*(you)*")
