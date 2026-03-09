"""
app.py
------
Entry point for the Solar Microgrid AI Power Distribution System.
Handles login screen routing and page dispatch.

Run:
    streamlit run app.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st

# ── Page config (must be first Streamlit call) ─────────────────────────────────
st.set_page_config(
    page_title="Solar Microgrid AI",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Bootstrap DB ──────────────────────────────────────────────────────────────
from database import init_db
init_db()

# ── Auto-train model if .pkl files are missing ────────────────────────────────
import os
MODEL_PATH = os.path.join("models", "random_forest_model.pkl")
if not os.path.exists(MODEL_PATH):
    with st.spinner("🤖 First-time setup: Training AI model on solar dataset… (this takes ~30 seconds)"):
        from train_model import train
        train()
    st.success("✅ Model trained and ready!")
    st.rerun()

from utils.auth import is_logged_in, login, current_role

# ── Global dark theme injection ────────────────────────────────────────────────
st.markdown("""
<style>
/* ─── Global ─── */
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Exo+2:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Exo 2', sans-serif;
    background-color: #0a0a1a;
    color: #e0e0e0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d0d2b 0%, #0a0a1a 100%) !important;
    border-right: 1px solid #1e1e3f;
}

/* Headers */
h1, h2, h3 { font-family: 'Share Tech Mono', monospace; }

/* Metric boxes */
[data-testid="metric-container"] {
    background: #111128;
    border: 1px solid #2a2a4a;
    border-radius: 10px;
    padding: 12px;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #00C49A, #007a60);
    color: #fff;
    border: none;
    border-radius: 8px;
    font-family: 'Share Tech Mono', monospace;
    letter-spacing: 0.5px;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #00e6b3, #009973);
    box-shadow: 0 0 16px #00C49A55;
}

/* Inputs */
input, textarea, select {
    background: #111128 !important;
    color: #e0e0e0 !important;
    border: 1px solid #2a2a4a !important;
}

/* Cards / expanders */
[data-testid="stExpander"] {
    background: #111128;
    border: 1px solid #2a2a4a;
    border-radius: 10px;
}

/* Dataframe */
[data-testid="stDataFrame"] { border: 1px solid #2a2a4a; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0a0a1a; }
::-webkit-scrollbar-thumb { background: #2a2a4a; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ── Routing ────────────────────────────────────────────────────────────────────
if not is_logged_in():
    # ── Login Page ─────────────────────────────────────────────────────────────
    col_left, col_mid, col_right = st.columns([1, 1.6, 1])
    with col_mid:
        st.markdown("""
        <div style="text-align:center;padding:40px 0 20px;">
            <div style="font-size:4rem;">☀️</div>
            <h1 style="font-family:'Share Tech Mono',monospace;color:#00C49A;
                       font-size:1.6rem;letter-spacing:2px;margin:0;">
                SOLAR MICROGRID AI
            </h1>
            <p style="color:#888;font-size:0.85rem;margin-top:4px;">
                Power Distribution Optimization System
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="background:linear-gradient(135deg,#111128,#0d0d2b);
                    border:1px solid #2a2a4a;border-radius:16px;padding:32px 28px;">
        """, unsafe_allow_html=True)

        st.markdown("#### 🔐 Sign In")
        username = st.text_input("Username", placeholder="admin or operator")
        password = st.text_input("Password", type="password", placeholder="••••••••")

        if st.button("Login →", use_container_width=True):
            if login(username, password):
                st.success("Login successful! Loading dashboard…")
                st.rerun()
            else:
                st.error("Invalid username or password.")

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""
        <div style="text-align:center;margin-top:20px;color:#555;font-size:0.8rem;">
            Default accounts — Admin: <code>admin / admin123</code> &nbsp;|&nbsp;
            User: <code>operator / user123</code>
        </div>
        """, unsafe_allow_html=True)

else:
    # ── Dispatch to role-specific dashboard ────────────────────────────────────
    role = current_role()
    if role == "admin":
        from pages.admin_dashboard import render
        render()
    else:
        from pages.user_dashboard import render
        render()
