"""
app.py
------
Solar Microgrid AI Power Distribution Optimization System.
Single-file version for Streamlit Cloud compatibility.

Run:
    streamlit run app.py
"""

import sys
import os
import hashlib

import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Solar Microgrid AI",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global dark theme ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Exo+2:wght@300;400;600;700&display=swap');
html, body, [class*="css"] {
    font-family: 'Exo 2', sans-serif;
    background-color: #0a0a1a;
    color: #e0e0e0;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d0d2b 0%, #0a0a1a 100%) !important;
    border-right: 1px solid #1e1e3f;
}
h1, h2, h3 { font-family: 'Share Tech Mono', monospace; }
[data-testid="metric-container"] {
    background: #111128; border: 1px solid #2a2a4a;
    border-radius: 10px; padding: 12px;
}
.stButton > button {
    background: linear-gradient(135deg, #00C49A, #007a60);
    color: #fff; border: none; border-radius: 8px;
    font-family: 'Share Tech Mono', monospace;
    letter-spacing: 0.5px; transition: all 0.2s;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #00e6b3, #009973);
    box-shadow: 0 0 16px #00C49A55;
}
input, textarea, select {
    background: #111128 !important;
    color: #e0e0e0 !important;
    border: 1px solid #2a2a4a !important;
}
[data-testid="stExpander"] {
    background: #111128; border: 1px solid #2a2a4a; border-radius: 10px;
}
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0a0a1a; }
::-webkit-scrollbar-thumb { background: #2a2a4a; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# DATABASE
# ══════════════════════════════════════════════════════════════════════════════
import sqlite3

DB_PATH = "solar_microgrid.db"

def get_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT    NOT NULL UNIQUE,
            password TEXT    NOT NULL,
            role     TEXT    NOT NULL CHECK(role IN ('admin','user'))
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS prediction_logs (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            username      TEXT    NOT NULL,
            timestamp     DATETIME DEFAULT CURRENT_TIMESTAMP,
            solar_irr     REAL,
            cloud_cover   REAL,
            battery_level REAL,
            total_load    REAL,
            prediction    TEXT
        )
    """)
    for username, pwd, role in [("admin", "admin123", "admin"), ("operator", "user123", "user")]:
        cursor.execute(
            "INSERT OR IGNORE INTO users (username, password, role) VALUES (?, ?, ?)",
            (username, hash_password(pwd), role),
        )
    conn.commit()
    conn.close()

def verify_user(username, password):
    conn = get_connection()
    user = conn.execute(
        "SELECT * FROM users WHERE username=? AND password=?",
        (username, hash_password(password)),
    ).fetchone()
    conn.close()
    return user

def get_all_users():
    conn = get_connection()
    users = conn.execute("SELECT id, username, role FROM users").fetchall()
    conn.close()
    return users

def add_user(username, password, role):
    try:
        conn = get_connection()
        conn.execute("INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
                     (username, hash_password(password), role))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False

def delete_user(user_id):
    conn = get_connection()
    conn.execute("DELETE FROM users WHERE id=?", (user_id,))
    conn.commit()
    conn.close()

def log_prediction(username, solar_irr, cloud_cover, battery_level, total_load, prediction):
    conn = get_connection()
    conn.execute(
        "INSERT INTO prediction_logs (username,solar_irr,cloud_cover,battery_level,total_load,prediction) VALUES (?,?,?,?,?,?)",
        (username, solar_irr, cloud_cover, battery_level, total_load, prediction),
    )
    conn.commit()
    conn.close()

def get_prediction_logs(limit=50):
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM prediction_logs ORDER BY timestamp DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return rows

# ── Bootstrap DB ──────────────────────────────────────────────────────────────
init_db()

# ══════════════════════════════════════════════════════════════════════════════
# MODEL TRAINING
# ══════════════════════════════════════════════════════════════════════════════
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

FEATURES = [
    "Solar_Irradiance","Cloud_Cover","Temperature","Humidity",
    "Battery_Level","Grid_Price","Solar_Generation_kW",
    "Hospital_Load_kW","Residential_Load_kW","EV_Load_kW",
    "Emergency_Load_kW","Total_Load_kW",
]
TARGET     = "Distribution_Action"
MODEL_DIR  = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "random_forest_model.pkl")
ENC_PATH   = os.path.join(MODEL_DIR, "label_encoder.pkl")

def train_model():
    df  = pd.read_csv("solar_microgrid_ai_dataset.csv")
    le  = LabelEncoder()
    y   = le.fit_transform(df[TARGET])
    X   = df[FEATURES].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(le,  ENC_PATH)
    return clf, le

# ── Auto-train if model missing ───────────────────────────────────────────────
if not os.path.exists(MODEL_PATH):
    with st.spinner("🤖 First-time setup: Training AI model… (~30 seconds)"):
        train_model()
    st.success("✅ Model trained and ready!")
    st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
ACTION_LABELS = {
    "Emergency_Priority":    "🚨 Emergency Priority Mode",
    "Hospital_Priority":     "🏥 Hospital Priority Mode",
    "Balanced_Distribution": "⚖️  Balanced Distribution Mode",
    "EV_Charging_Allowed":   "⚡ EV Charging Allowed",
    "Residential_Support":   "🏘️  Residential Support Mode",
}
ACTION_DESCRIPTIONS = {
    "Emergency_Priority":    "All available power is routed to emergency services. Non-critical loads are shed.",
    "Hospital_Priority":     "Medical facilities take precedence. Hospital and emergency loads are fully supplied.",
    "Balanced_Distribution": "Power is distributed evenly across all consumer categories.",
    "EV_Charging_Allowed":   "Sufficient surplus energy exists — EV charging stations are online.",
    "Residential_Support":   "Residential areas receive priority to maintain household stability.",
}
ACTION_COLORS = {
    "Emergency_Priority":    "#FF4444",
    "Hospital_Priority":     "#FF8C00",
    "Balanced_Distribution": "#00C49A",
    "EV_Charging_Allowed":   "#4FC3F7",
    "Residential_Support":   "#AB47BC",
}

_clf, _le = None, None

def load_model():
    global _clf, _le
    if _clf is None:
        _clf = joblib.load(MODEL_PATH)
        _le  = joblib.load(ENC_PATH)

def predict(solar_irr, cloud_cover, temperature, humidity, battery_level,
            grid_price, solar_gen, hospital_load, residential_load,
            ev_load, emergency_load, total_load):
    load_model()
    X       = np.array([[solar_irr, cloud_cover, temperature, humidity,
                         battery_level, grid_price, solar_gen,
                         hospital_load, residential_load, ev_load,
                         emergency_load, total_load]])
    encoded = _clf.predict(X)[0]
    action  = _le.inverse_transform([encoded])[0]
    probs   = _clf.predict_proba(X)[0]
    prob_map = {_le.classes_[i]: round(float(p)*100, 1) for i, p in enumerate(probs)}
    return {
        "action":        action,
        "label":         ACTION_LABELS.get(action, action),
        "description":   ACTION_DESCRIPTIONS.get(action, ""),
        "color":         ACTION_COLORS.get(action, "#FFFFFF"),
        "probabilities": prob_map,
    }

# ══════════════════════════════════════════════════════════════════════════════
# AUTH HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def is_logged_in():
    return st.session_state.get("logged_in", False)

def current_role():
    return st.session_state.get("role", "")

def current_user():
    return st.session_state.get("username", "")

def do_login(username, password):
    user = verify_user(username, password)
    if user:
        st.session_state["logged_in"] = True
        st.session_state["username"]  = user["username"]
        st.session_state["role"]      = user["role"]
        st.session_state["user_id"]   = user["id"]
        return True
    return False

def do_logout():
    for key in ["logged_in", "username", "role", "user_id"]:
        st.session_state.pop(key, None)

# ══════════════════════════════════════════════════════════════════════════════
# PAGES
# ══════════════════════════════════════════════════════════════════════════════
import plotly.express as px
import plotly.graph_objects as go

def gauge(value, title, color="#00C49A", max_val=100):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value,
        title={"text": title, "font": {"color": "#ccc", "size": 13}},
        gauge={
            "axis": {"range": [0, max_val], "tickcolor": "#555"},
            "bar":  {"color": color},
            "bgcolor": "#1a1a2e", "bordercolor": "#333",
            "steps": [
                {"range": [0,              max_val*0.33], "color": "#0f0f1a"},
                {"range": [max_val*0.33,   max_val*0.66], "color": "#16213e"},
                {"range": [max_val*0.66,   max_val],      "color": "#1a1a3e"},
            ],
        },
        number={"font": {"color": "#fff", "size": 26}},
    ))
    fig.update_layout(height=180, margin=dict(t=30,b=0,l=10,r=10),
                      paper_bgcolor="rgba(0,0,0,0)", font_color="#fff")
    return fig


def page_login():
    col_l, col_m, col_r = st.columns([1, 1.6, 1])
    with col_m:
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

        st.markdown("#### 🔐 Sign In")
        username = st.text_input("Username", placeholder="admin or operator")
        password = st.text_input("Password", type="password", placeholder="••••••••")

        if st.button("Login →", use_container_width=True):
            if do_login(username, password):
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid username or password.")

        st.markdown("""
        <div style="text-align:center;margin-top:20px;color:#555;font-size:0.8rem;">
            Default — Admin: <code>admin / admin123</code> &nbsp;|&nbsp;
            User: <code>operator / user123</code>
        </div>
        """, unsafe_allow_html=True)


def page_admin():
    with st.sidebar:
        st.markdown("### ☀️ Solar Microgrid AI")
        st.markdown(f"**Admin:** {current_user()}")
        st.markdown("---")
        section = st.radio("Navigation", [
            "📊 Dataset Overview", "🔮 Prediction Logs", "👥 User Management"
        ], label_visibility="collapsed")
        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True):
            do_logout()
            st.rerun()

    st.markdown("<h1 style='color:#00C49A;font-family:monospace;'>⚡ Admin Control Panel</h1>",
                unsafe_allow_html=True)

    if "Dataset" in section:
        st.subheader("📊 Dataset Statistics")
        try:
            df = pd.read_csv("solar_microgrid_ai_dataset.csv")
        except FileNotFoundError:
            st.error("Dataset file not found.")
            return

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Records", f"{len(df):,}")
        c2.metric("Features", len(df.columns)-1)
        c3.metric("Classes", df[TARGET].nunique())
        c4.metric("Avg Solar kW", f"{df['Solar_Generation_kW'].mean():.2f}")

        counts = df[TARGET].value_counts().reset_index()
        counts.columns = ["Action","Count"]
        fig = px.bar(counts, x="Action", y="Count", color="Action", template="plotly_dark",
                     color_discrete_sequence=["#00C49A","#FF8C00","#4FC3F7","#AB47BC","#FF4444"])
        fig.update_layout(showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

        ca, cb = st.columns(2)
        with ca:
            fig2 = px.scatter(df.sample(500, random_state=42),
                              x="Solar_Irradiance", y="Solar_Generation_kW",
                              color=TARGET, opacity=0.7, template="plotly_dark",
                              color_discrete_sequence=["#00C49A","#FF8C00","#4FC3F7","#AB47BC","#FF4444"])
            fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig2, use_container_width=True)
        with cb:
            fig3 = px.histogram(df, x="Battery_Level", nbins=30, color=TARGET,
                                template="plotly_dark",
                                color_discrete_sequence=["#00C49A","#FF8C00","#4FC3F7","#AB47BC","#FF4444"])
            fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig3, use_container_width=True)

        st.dataframe(df.describe().round(2), use_container_width=True)

    elif "Prediction" in section:
        st.subheader("🔮 AI Prediction Logs")
        logs = get_prediction_logs(100)
        if not logs:
            st.info("No predictions recorded yet.")
        else:
            log_df = pd.DataFrame([dict(r) for r in logs])
            st.dataframe(log_df, use_container_width=True)
            freq = log_df["prediction"].value_counts().reset_index()
            freq.columns = ["Action","Count"]
            fig4 = px.pie(freq, names="Action", values="Count", template="plotly_dark",
                          color_discrete_sequence=["#00C49A","#FF8C00","#4FC3F7","#AB47BC","#FF4444"])
            fig4.update_layout(paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig4, use_container_width=True)

    elif "User" in section:
        st.subheader("👥 User Management")
        with st.expander("➕ Add New User"):
            with st.form("add_user_form"):
                new_user = st.text_input("Username")
                new_pass = st.text_input("Password", type="password")
                new_role = st.selectbox("Role", ["user","admin"])
                if st.form_submit_button("Create User"):
                    if new_user and new_pass:
                        if add_user(new_user, new_pass, new_role):
                            st.success(f"User '{new_user}' created.")
                        else:
                            st.error("Username already exists.")
                    else:
                        st.warning("Fill in all fields.")

        for u in get_all_users():
            c1, c2, c3, c4 = st.columns([1,3,2,2])
            c1.write(u["id"])
            c2.write(u["username"])
            c3.write(f"🏷️ {u['role']}")
            if u["username"] != current_user():
                if c4.button("🗑️ Delete", key=f"del_{u['id']}"):
                    delete_user(u["id"])
                    st.rerun()
            else:
                c4.write("*(you)*")


def page_user():
    with st.sidebar:
        st.markdown("### ☀️ Solar Microgrid AI")
        st.markdown(f"**User:** {current_user()}")
        st.markdown("---")
        st.markdown("Enter live system parameters and click **Predict** to receive an AI power distribution decision.")
        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True):
            do_logout()
            st.rerun()

    st.markdown("<h1 style='color:#4FC3F7;font-family:monospace;'>⚡ Live Energy Dashboard</h1>",
                unsafe_allow_html=True)

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
            battery_level = st.slider("Battery Level (%)",    0.0, 100.0, 70.0, 1.0)
            grid_price    = st.slider("Grid Price ($/kWh)",   0.0,   1.0,  0.15, 0.01)
            solar_gen     = st.slider("Solar Generation (kW)", 0.0,  20.0,  5.0, 0.1)
        with col3:
            st.markdown("**🏗️ Load Demands (kW)**")
            hospital_load    = st.slider("Hospital Load",    0.0,  80.0, 30.0, 0.5)
            residential_load = st.slider("Residential Load", 0.0, 120.0, 60.0, 0.5)
            ev_load          = st.slider("EV Load",          0.0,  40.0, 10.0, 0.5)
            emergency_load   = st.slider("Emergency Load",   0.0,  40.0, 10.0, 0.5)

        total_load = hospital_load + residential_load + ev_load + emergency_load
        submitted  = st.form_submit_button("🔮 Predict Distribution Action",
                                           use_container_width=True, type="primary")

    st.markdown("### 📊 Real-Time Metrics")
    g1, g2, g3, g4 = st.columns(4)
    g1.plotly_chart(gauge(solar_gen,     "Solar Generation kW", "#FFD700", 20),  use_container_width=True)
    g2.plotly_chart(gauge(battery_level, "Battery Level %",     "#00C49A", 100), use_container_width=True)
    g3.plotly_chart(gauge(total_load,    "Total Load kW",       "#FF8C00", 250), use_container_width=True)
    g4.plotly_chart(gauge(cloud_cover,   "Cloud Cover %",       "#4FC3F7", 100), use_container_width=True)

    load_df = pd.DataFrame({
        "Consumer":  ["Hospital","Residential","EV","Emergency"],
        "Load (kW)": [hospital_load, residential_load, ev_load, emergency_load],
    })
    fig_bar = px.bar(load_df, x="Consumer", y="Load (kW)", color="Consumer", text_auto=True,
                     color_discrete_sequence=["#FF8C00","#AB47BC","#4FC3F7","#FF4444"],
                     template="plotly_dark")
    fig_bar.update_layout(showlegend=False, paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", height=250)
    st.plotly_chart(fig_bar, use_container_width=True)

    if submitted:
        result = predict(solar_irr, cloud_cover, temperature, humidity,
                         battery_level, grid_price, solar_gen,
                         hospital_load, residential_load, ev_load,
                         emergency_load, total_load)
        log_prediction(current_user(), solar_irr, cloud_cover,
                       battery_level, total_load, result["action"])
        color = result["color"]
        st.markdown("---")
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,{color}22,{color}11);
                    border:2px solid {color};border-radius:16px;padding:28px;
                    text-align:center;box-shadow:0 0 30px {color}44;">
            <div style="font-size:2.8rem;margin-bottom:8px;">{result['label']}</div>
            <div style="color:#ccc;font-size:1.05rem;">{result['description']}</div>
        </div>""", unsafe_allow_html=True)

        prob_df = pd.DataFrame(list(result["probabilities"].items()),
                               columns=["Action","Probability (%)"]).sort_values("Probability (%)", ascending=True)
        fig_prob = px.bar(prob_df, x="Probability (%)", y="Action", orientation="h",
                          color="Probability (%)", color_continuous_scale=["#1a1a2e","#00C49A"],
                          template="plotly_dark", text_auto=True)
        fig_prob.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               coloraxis_showscale=False, height=280)
        st.plotly_chart(fig_prob, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# ROUTER
# ══════════════════════════════════════════════════════════════════════════════
if not is_logged_in():
    page_login()
elif current_role() == "admin":
    page_admin()
else:
    page_user()
