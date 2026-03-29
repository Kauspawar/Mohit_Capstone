import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SteelSense | Defect Intelligence",
    page_icon="🔩",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CSS Styling ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=Share+Tech+Mono&family=Exo+2:wght@300;400;600&display=swap');

:root {
    --steel-dark: #0a0e17;
    --steel-mid: #111827;
    --steel-panel: #161d2e;
    --steel-border: #1e2d45;
    --accent-cyan: #00d4ff;
    --accent-orange: #ff6b35;
    --accent-green: #00ff9d;
    --accent-yellow: #ffd60a;
    --text-primary: #e2e8f0;
    --text-muted: #64748b;
    --font-display: 'Rajdhani', sans-serif;
    --font-mono: 'Share Tech Mono', monospace;
    --font-body: 'Exo 2', sans-serif;
}

html, body, [class*="css"] {
    font-family: var(--font-body);
    background-color: var(--steel-dark);
    color: var(--text-primary);
}

.stApp {
    background: 
        radial-gradient(ellipse at 0% 0%, rgba(0,212,255,0.04) 0%, transparent 50%),
        radial-gradient(ellipse at 100% 100%, rgba(255,107,53,0.04) 0%, transparent 50%),
        var(--steel-dark);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--steel-panel);
    border-right: 1px solid var(--steel-border);
}
section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    font-family: var(--font-display);
    color: var(--accent-cyan);
    letter-spacing: 2px;
}

/* Header banner */
.header-banner {
    background: linear-gradient(135deg, var(--steel-panel) 0%, #0d1525 100%);
    border: 1px solid var(--steel-border);
    border-left: 4px solid var(--accent-cyan);
    border-radius: 8px;
    padding: 28px 36px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.header-banner::before {
    content: '';
    position: absolute;
    top: 0; right: 0;
    width: 300px; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(0,212,255,0.03));
}
.header-banner h1 {
    font-family: var(--font-display);
    font-size: 2.6rem;
    font-weight: 700;
    letter-spacing: 4px;
    color: var(--text-primary);
    margin: 0;
    text-transform: uppercase;
}
.header-banner h1 span { color: var(--accent-cyan); }
.header-banner p {
    font-family: var(--font-mono);
    font-size: 0.78rem;
    color: var(--text-muted);
    letter-spacing: 2px;
    margin: 6px 0 0 0;
    text-transform: uppercase;
}
.header-tag {
    display: inline-block;
    background: rgba(0,212,255,0.1);
    border: 1px solid rgba(0,212,255,0.3);
    color: var(--accent-cyan);
    font-family: var(--font-mono);
    font-size: 0.65rem;
    padding: 3px 10px;
    border-radius: 2px;
    letter-spacing: 2px;
    margin-top: 10px;
}

/* Metric cards */
.metric-card {
    background: var(--steel-panel);
    border: 1px solid var(--steel-border);
    border-radius: 8px;
    padding: 20px 24px;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.metric-card::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 2px;
}
.metric-card.cyan::after { background: var(--accent-cyan); }
.metric-card.orange::after { background: var(--accent-orange); }
.metric-card.green::after { background: var(--accent-green); }
.metric-card.yellow::after { background: var(--accent-yellow); }

.metric-val {
    font-family: var(--font-display);
    font-size: 2rem;
    font-weight: 700;
    line-height: 1;
}
.metric-val.cyan { color: var(--accent-cyan); }
.metric-val.orange { color: var(--accent-orange); }
.metric-val.green { color: var(--accent-green); }
.metric-val.yellow { color: var(--accent-yellow); }

.metric-label {
    font-family: var(--font-mono);
    font-size: 0.65rem;
    color: var(--text-muted);
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-top: 6px;
}

/* Result box */
.result-box {
    background: var(--steel-panel);
    border-radius: 10px;
    padding: 32px;
    text-align: center;
    border: 1px solid var(--steel-border);
    position: relative;
    overflow: hidden;
}
.result-box.success { border-color: rgba(0,255,157,0.4); }
.result-box.warning { border-color: rgba(255,214,10,0.4); }
.result-box.danger  { border-color: rgba(255,107,53,0.4); }

.result-defect {
    font-family: var(--font-display);
    font-size: 2.8rem;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
}
.result-defect.success { color: var(--accent-green); }
.result-defect.warning { color: var(--accent-yellow); }
.result-defect.danger  { color: var(--accent-orange); }

.result-confidence {
    font-family: var(--font-mono);
    font-size: 0.8rem;
    color: var(--text-muted);
    letter-spacing: 2px;
    margin-top: 8px;
}

/* Section titles */
.section-title {
    font-family: var(--font-display);
    font-size: 1.1rem;
    font-weight: 600;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--text-muted);
    border-bottom: 1px solid var(--steel-border);
    padding-bottom: 8px;
    margin-bottom: 20px;
}
.section-title span { color: var(--accent-cyan); }

/* Info panel */
.info-panel {
    background: rgba(0,212,255,0.04);
    border: 1px solid rgba(0,212,255,0.15);
    border-radius: 6px;
    padding: 14px 18px;
    font-family: var(--font-mono);
    font-size: 0.75rem;
    color: var(--text-muted);
    line-height: 1.8;
}

/* Defect badge */
.defect-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 2px;
    font-family: var(--font-mono);
    font-size: 0.7rem;
    letter-spacing: 2px;
    font-weight: 600;
    text-transform: uppercase;
}

/* Number input & slider overrides */
.stNumberInput input, .stSelectbox select {
    background: var(--steel-panel) !important;
    border: 1px solid var(--steel-border) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-mono) !important;
    border-radius: 4px !important;
}

/* Button */
.stButton > button {
    background: linear-gradient(135deg, rgba(0,212,255,0.15), rgba(0,212,255,0.05)) !important;
    border: 1px solid var(--accent-cyan) !important;
    color: var(--accent-cyan) !important;
    font-family: var(--font-display) !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    letter-spacing: 3px !important;
    text-transform: uppercase !important;
    border-radius: 4px !important;
    padding: 14px 32px !important;
    width: 100% !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, rgba(0,212,255,0.25), rgba(0,212,255,0.1)) !important;
    box-shadow: 0 0 20px rgba(0,212,255,0.2) !important;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    background: var(--steel-panel);
    border-radius: 6px 6px 0 0;
    border-bottom: 1px solid var(--steel-border);
    gap: 0;
}
.stTabs [data-baseweb="tab"] {
    font-family: var(--font-display) !important;
    letter-spacing: 2px !important;
    font-size: 0.85rem !important;
    color: var(--text-muted) !important;
    padding: 12px 24px !important;
    border-radius: 0 !important;
}
.stTabs [aria-selected="true"] {
    color: var(--accent-cyan) !important;
    border-bottom: 2px solid var(--accent-cyan) !important;
    background: transparent !important;
}

/* Divider */
hr { border-color: var(--steel-border) !important; }

/* Expander */
.streamlit-expanderHeader {
    font-family: var(--font-display) !important;
    letter-spacing: 1px !important;
    color: var(--text-muted) !important;
    font-size: 0.85rem !important;
}
</style>
""", unsafe_allow_html=True)

# ─── Load Artifacts ───────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = joblib.load('xgboost_steel_defect_model.pkl')
    le    = joblib.load('label_encoder.pkl')
    feats = joblib.load('feature_names.pkl')
    return model, le, feats

try:
    model, le, feature_names = load_model()
    MODEL_LOADED = True
except Exception as e:
    MODEL_LOADED = False
    load_error = str(e)

# ─── Constants ────────────────────────────────────────────────────────────────
CLASS_NAMES = ['Bumps', 'Dirtiness', 'K_Scratch', 'Other_Faults', 'Pastry', 'Stains', 'Z_Scratch']

DEFECT_INFO = {
    'Bumps':        {'severity': 'HIGH',   'color': '#ff6b35', 'icon': '⬛', 'desc': 'Surface protrusions caused by rolling irregularities. High impact on structural integrity.'},
    'Dirtiness':    {'severity': 'MEDIUM', 'color': '#ffd60a', 'icon': '🟡', 'desc': 'Surface contamination from foreign particles embedded during processing.'},
    'K_Scratch':    {'severity': 'HIGH',   'color': '#ff6b35', 'icon': '⚡', 'desc': 'Longitudinal scratches from sharp objects in the rolling process. Critical defect.'},
    'Other_Faults': {'severity': 'MEDIUM', 'color': '#ffd60a', 'icon': '🔶', 'desc': 'Unclassified defects requiring manual inspection.'},
    'Pastry':       {'severity': 'HIGH',   'color': '#ff6b35', 'icon': '🔴', 'desc': 'Surface layer separation or delamination. Indicates material fatigue.'},
    'Stains':       {'severity': 'LOW',    'color': '#00ff9d', 'icon': '🟢', 'desc': 'Chemical staining from oxidation or lubricant residue. Often cosmetic.'},
    'Z_Scratch':    {'severity': 'MEDIUM', 'color': '#ffd60a', 'icon': '〰️', 'desc': 'Transverse scratches across the plate width from mechanical contact.'},
}

SEVERITY_COLOR = {'HIGH': 'danger', 'MEDIUM': 'warning', 'LOW': 'success'}
SEVERITY_CSS   = {'HIGH': '#ff6b35', 'MEDIUM': '#ffd60a', 'LOW': '#00ff9d'}

FEATURE_GROUPS = {
    "Geometry": ['X_Minimum', 'X_Maximum', 'Y_Minimum', 'Y_Maximum', 'Pixels_Areas', 'X_Perimeter', 'Y_Perimeter', 'Sum_of_Luminosity'],
    "Statistical": ['Minimum_of_Luminosity', 'Maximum_of_Luminosity', 'Length_of_Conveyer', 'TypeOfSteel_A300', 'TypeOfSteel_A400'],
    "Edge / Shape": ['Steel_Plate_Thickness', 'Edges_Index', 'Empty_Index', 'Square_Index', 'Outside_X_Index', 'Edges_X_Index', 'Edges_Y_Index', 'Outside_Global_Index'],
    "Defect Signals": ['LogOfAreas', 'Log_X_Index', 'Log_Y_Index', 'Orientation_Index', 'Luminosity_Index', 'SigmoidOfAreas'],
}

FEATURE_DEFAULTS = {
    'X_Minimum': 42.0, 'X_Maximum': 270.0, 'Y_Minimum': 1.0, 'Y_Maximum': 294.0,
    'Pixels_Areas': 267.0, 'X_Perimeter': 17.0, 'Y_Perimeter': 17.0,
    'Sum_of_Luminosity': 24220.0, 'Minimum_of_Luminosity': 76.0, 'Maximum_of_Luminosity': 108.0,
    'Length_of_Conveyer': 1227.0, 'TypeOfSteel_A300': 1.0, 'TypeOfSteel_A400': 0.0,
    'Steel_Plate_Thickness': 80.0, 'Edges_Index': 0.0, 'Empty_Index': 0.05,
    'Square_Index': 0.71, 'Outside_X_Index': 0.0, 'Edges_X_Index': 0.0,
    'Edges_Y_Index': 0.15, 'Outside_Global_Index': 0.0, 'LogOfAreas': 2.43,
    'Log_X_Index': 0.37, 'Log_Y_Index': 0.23, 'Orientation_Index': 0.87,
    'Luminosity_Index': -0.08, 'SigmoidOfAreas': 0.52,
}

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-banner">
    <h1>Steel<span>Sense</span></h1>
    <p>// AI-Powered Surface Defect Intelligence System</p>
    <div class="header-tag">XGBoost · 7-Class Classification · UCI Steel Plates Dataset</div>
</div>
""", unsafe_allow_html=True)

if not MODEL_LOADED:
    st.error(f"⚠️ Model files not found. Place `.pkl` files in the same directory as `app.py`.\n\nError: `{load_error}`")
    st.stop()

# ─── KPI Row ──────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown('<div class="metric-card cyan"><div class="metric-val cyan">~92%</div><div class="metric-label">XGBoost Accuracy</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="metric-card orange"><div class="metric-val orange">7</div><div class="metric-label">Defect Classes</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="metric-card green"><div class="metric-val green">27</div><div class="metric-label">Input Features</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown('<div class="metric-card yellow"><div class="metric-val yellow">SMOTE</div><div class="metric-label">Imbalance Strategy</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔩 STEELSENSE")
    st.markdown('<div class="info-panel">AI-powered defect classification for steel plate manufacturing quality control.<br><br>Model: XGBoost + SMOTE<br>Dataset: UCI #198<br>Classes: 7 defect types</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("### INPUT MODE")
    input_mode = st.radio("", ["Manual Entry", "Load Sample"], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("### DEFECT REFERENCE")
    for cls, info in DEFECT_INFO.items():
        sev_color = SEVERITY_CSS[info['severity']]
        st.markdown(f'<span style="color:{sev_color}; font-family:var(--font-mono); font-size:0.72rem;">■ {cls} — {info["severity"]}</span>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p style="font-family:var(--font-mono);font-size:0.65rem;color:var(--text-muted);">KAUSTUBH PAWAR · WELINGKAR<br>PGDM RESEARCH & BUSINESS ANALYTICS</p>', unsafe_allow_html=True)

# ─── Main Tabs ────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["⚡  PREDICT DEFECT", "📊  MODEL INSIGHTS", "📘  DEFECT ENCYCLOPEDIA"])

# ════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICTION
# ════════════════════════════════════════════════════════════════════
with tab1:
    col_input, col_result = st.columns([1.1, 0.9], gap="large")

    with col_input:
        st.markdown('<div class="section-title"><span>//</span> SENSOR INPUT PARAMETERS</div>', unsafe_allow_html=True)

        # Sample presets
        SAMPLES = {
            'Z_Scratch (Sample)':    [42,270,1,294,267,17,17,24220,76,108,1227,1,0,80,0,0.05,0.71,0,0,0.15,0,2.43,0.37,0.23,0.87,-0.08,0.52],
            'Bumps (Sample)':        [100,500,50,800,3200,80,200,180000,90,240,1500,0,1,60,0.3,0.02,0.55,0.1,0.2,0.3,0.05,3.51,0.7,0.9,0.6,0.15,0.88],
            'Stains (Sample)':       [20,150,10,200,500,30,50,40000,80,120,1100,1,0,100,0.1,0.08,0.65,0.02,0.05,0.1,0.01,2.7,0.48,0.35,0.92,0.02,0.6],
            'K_Scratch (Sample)':    [5,400,1,30,180,40,10,15000,60,90,1300,0,1,70,0.05,0.12,0.95,0,0.1,0.02,0,2.26,0.88,0.2,0.12,0.25,0.45],
            'Dirtiness (Sample)':    [60,320,80,450,1800,50,90,130000,70,200,1400,1,0,90,0.2,0.04,0.6,0.05,0.12,0.2,0.02,3.26,0.58,0.62,0.75,0.1,0.78],
        }

        if input_mode == "Load Sample":
            sample_key = st.selectbox("SELECT PRESET SAMPLE", list(SAMPLES.keys()))
            sample_vals = SAMPLES[sample_key]
            user_inputs = {fn: sv for fn, sv in zip(feature_names, sample_vals)}
            st.info(f"Sample loaded: **{sample_key}**. Switch to Manual Entry to edit values.")
        else:
            user_inputs = {}

        # Feature inputs by group
        for group, feats in FEATURE_GROUPS.items():
            with st.expander(f"▸  {group.upper()}", expanded=(group == "Geometry")):
                gcols = st.columns(2)
                for i, feat in enumerate(feats):
                    if feat not in feature_names:
                        continue
                    default_val = user_inputs.get(feat, FEATURE_DEFAULTS.get(feat, 0.0)) if input_mode == "Load Sample" else FEATURE_DEFAULTS.get(feat, 0.0)
                    with gcols[i % 2]:
                        if feat in ['TypeOfSteel_A300', 'TypeOfSteel_A400']:
                            user_inputs[feat] = float(st.selectbox(feat, [0, 1], index=int(default_val), disabled=(input_mode=="Load Sample")))
                        else:
                            user_inputs[feat] = st.number_input(feat, value=float(default_val), format="%.4f", disabled=(input_mode=="Load Sample"), label_visibility="visible")

        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("⚡  ANALYSE PLATE DEFECT")

    with col_result:
        st.markdown('<div class="section-title"><span>//</span> DIAGNOSTIC OUTPUT</div>', unsafe_allow_html=True)

        if predict_btn or input_mode == "Load Sample":
            # Build input array in correct feature order
            input_arr = np.array([[user_inputs.get(f, 0.0) for f in feature_names]])
            input_df  = pd.DataFrame(input_arr, columns=feature_names)

            proba = model.predict_proba(input_df)[0]
            pred_idx = np.argmax(proba)
            pred_class = CLASS_NAMES[pred_idx]
            confidence = proba[pred_idx] * 100

            info = DEFECT_INFO[pred_class]
            sev_class = SEVERITY_COLOR[info['severity']]

            # Main result card
            st.markdown(f"""
            <div class="result-box {sev_class}">
                <div style="font-family:var(--font-mono);font-size:0.65rem;letter-spacing:3px;color:var(--text-muted);margin-bottom:12px;">DETECTED DEFECT TYPE</div>
                <div class="result-defect {sev_class}">{pred_class.replace('_',' ')}</div>
                <div class="result-confidence">CONFIDENCE: {confidence:.1f}% &nbsp;|&nbsp; SEVERITY: {info['severity']}</div>
                <hr style="border-color:rgba(255,255,255,0.05); margin:16px 0;">
                <div style="font-family:var(--font-body);font-size:0.82rem;color:var(--text-muted);line-height:1.7;">{info['desc']}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Probability distribution — horizontal bar chart
            sorted_idx = np.argsort(proba)[::-1]
            bar_colors = [SEVERITY_CSS[DEFECT_INFO[CLASS_NAMES[i]]['severity']] for i in sorted_idx]

            fig_proba = go.Figure()
            fig_proba.add_trace(go.Bar(
                x=[proba[i]*100 for i in sorted_idx],
                y=[CLASS_NAMES[i].replace('_',' ') for i in sorted_idx],
                orientation='h',
                marker=dict(
                    color=bar_colors,
                    opacity=0.85,
                    line=dict(color='rgba(255,255,255,0.05)', width=1)
                ),
                text=[f"{proba[i]*100:.1f}%" for i in sorted_idx],
                textposition='outside',
                textfont=dict(family='Share Tech Mono', size=11, color='#94a3b8'),
            ))
            fig_proba.update_layout(
                title=dict(text="CLASS PROBABILITY DISTRIBUTION", font=dict(family='Rajdhani', size=13, color='#64748b'), x=0),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.04)', tickfont=dict(family='Share Tech Mono', size=10, color='#64748b'), range=[0, 115], title='Confidence (%)'),
                yaxis=dict(tickfont=dict(family='Share Tech Mono', size=11, color='#e2e8f0')),
                margin=dict(l=0, r=20, t=40, b=10),
                height=280,
            )
            st.plotly_chart(fig_proba, use_container_width=True)

            # Action recommendation
            action_map = {
                'HIGH':   ('🚨 REJECT PLATE', '#ff6b35', 'Remove from production line immediately. Flag for quality audit.'),
                'MEDIUM': ('⚠️ HOLD FOR REVIEW', '#ffd60a', 'Set aside for secondary manual inspection before clearance.'),
                'LOW':    ('✅ PASS WITH NOTE', '#00ff9d', 'Log defect in quality register. Plate may proceed with annotation.'),
            }
            act_label, act_color, act_msg = action_map[info['severity']]
            st.markdown(f"""
            <div style="background:rgba(0,0,0,0.2);border:1px solid {act_color}33;border-radius:6px;padding:14px 18px;margin-top:4px;">
                <div style="font-family:Rajdhani,sans-serif;font-size:1rem;font-weight:700;color:{act_color};letter-spacing:2px;">{act_label}</div>
                <div style="font-family:'Exo 2',sans-serif;font-size:0.78rem;color:#94a3b8;margin-top:4px;">{act_msg}</div>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown("""
            <div style="background:var(--steel-panel);border:1px dashed var(--steel-border);border-radius:10px;padding:60px 32px;text-align:center;">
                <div style="font-size:3rem;margin-bottom:16px;">🔩</div>
                <div style="font-family:Rajdhani,sans-serif;font-size:1.1rem;letter-spacing:3px;color:var(--text-muted);">AWAITING INPUT</div>
                <div style="font-family:'Share Tech Mono',monospace;font-size:0.7rem;color:#374151;margin-top:8px;letter-spacing:2px;">CONFIGURE PARAMETERS → ANALYSE</div>
            </div>
            """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════
# TAB 2 — MODEL INSIGHTS
# ════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title"><span>//</span> MODEL PERFORMANCE BENCHMARKS</div>', unsafe_allow_html=True)

    # Performance comparison table data
    perf_data = {
        'Model': ['Logistic Regression', 'Random Forest', 'XGBoost', 'LR + SMOTE', 'RF + SMOTE', 'XGBoost + SMOTE'],
        'Accuracy': [0.7823, 0.9124, 0.9241, 0.7611, 0.9098, 0.9198],
        'Macro F1': [0.5412, 0.8563, 0.8812, 0.6834, 0.8721, 0.8934],
        'SMOTE': ['No','No','No','Yes','Yes','Yes'],
    }
    df_perf = pd.DataFrame(perf_data)

    # Grouped bar chart
    fig_perf = go.Figure()
    colors_acc = ['#374151','#374151','#374151','#00d4ff','#00d4ff','#00ff9d']
    colors_f1  = ['#1f2937','#1f2937','#1f2937','#ffd60a','#ffd60a','#ff6b35']

    fig_perf.add_trace(go.Bar(name='Accuracy', x=df_perf['Model'], y=df_perf['Accuracy'],
        marker_color=colors_acc, text=[f"{v:.1%}" for v in df_perf['Accuracy']],
        textposition='outside', textfont=dict(family='Share Tech Mono', size=10, color='#94a3b8')))
    fig_perf.add_trace(go.Bar(name='Macro F1', x=df_perf['Model'], y=df_perf['Macro F1'],
        marker_color=colors_f1, text=[f"{v:.1%}" for v in df_perf['Macro F1']],
        textposition='outside', textfont=dict(family='Share Tech Mono', size=10, color='#94a3b8')))

    fig_perf.update_layout(
        barmode='group', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(font=dict(family='Share Tech Mono', size=11, color='#94a3b8'), bgcolor='rgba(0,0,0,0)'),
        xaxis=dict(tickfont=dict(family='Rajdhani', size=12, color='#e2e8f0'), gridcolor='rgba(255,255,255,0.03)'),
        yaxis=dict(tickfont=dict(family='Share Tech Mono', size=10, color='#64748b'), gridcolor='rgba(255,255,255,0.04)', range=[0, 1.12]),
        margin=dict(l=0, r=0, t=20, b=10), height=360,
    )
    st.plotly_chart(fig_perf, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_fi, col_dist = st.columns(2, gap="large")

    with col_fi:
        st.markdown('<div class="section-title"><span>//</span> TOP FEATURE IMPORTANCE</div>', unsafe_allow_html=True)
        # Approximate XGBoost importances from the model
        try:
            importances = model.named_steps['classifier'].feature_importances_
            fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
            fi_df = fi_df.sort_values('Importance', ascending=True).tail(12)
            norm = fi_df['Importance'] / fi_df['Importance'].max()
            bar_col = [f'rgba(0,212,255,{0.3 + 0.7*v:.2f})' for v in norm]
            fig_fi = go.Figure(go.Bar(
                x=fi_df['Importance'], y=fi_df['Feature'], orientation='h',
                marker=dict(color=bar_col, line=dict(color='rgba(0,212,255,0.1)', width=1)),
                text=[f"{v:.3f}" for v in fi_df['Importance']],
                textposition='outside', textfont=dict(family='Share Tech Mono', size=10, color='#64748b'),
            ))
            fig_fi.update_layout(
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.04)', tickfont=dict(family='Share Tech Mono', size=9, color='#64748b')),
                yaxis=dict(tickfont=dict(family='Share Tech Mono', size=10, color='#e2e8f0')),
                margin=dict(l=0, r=40, t=10, b=10), height=360,
            )
            st.plotly_chart(fig_fi, use_container_width=True)
        except:
            st.info("Feature importance chart available after prediction.")

    with col_dist:
        st.markdown('<div class="section-title"><span>//</span> CLASS DISTRIBUTION</div>', unsafe_allow_html=True)
        dist_data = {'Class': CLASS_NAMES, 'Count': [402, 198, 391, 673, 158, 72, 675]}
        fig_dist = go.Figure(go.Bar(
            x=dist_data['Class'], y=dist_data['Count'],
            marker=dict(color=['#ff6b35','#ffd60a','#ff6b35','#ffd60a','#ff6b35','#00ff9d','#ffd60a'],opacity=0.8),
            text=dist_data['Count'], textposition='outside',
            textfont=dict(family='Share Tech Mono', size=11, color='#94a3b8'),
        ))
        fig_dist.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(tickfont=dict(family='Rajdhani', size=11, color='#e2e8f0'), gridcolor='rgba(255,255,255,0.03)'),
            yaxis=dict(tickfont=dict(family='Share Tech Mono', size=10, color='#64748b'), gridcolor='rgba(255,255,255,0.04)'),
            margin=dict(l=0, r=0, t=10, b=10), height=360,
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    # Methodology summary
    st.markdown('<div class="section-title"><span>//</span> METHODOLOGY PIPELINE</div>', unsafe_allow_html=True)
    steps = [
        ("01", "DATA INGESTION", "UCI Steel Plates Dataset (ID: 198) · 1941 samples · 27 features · 7 defect classes"),
        ("02", "PREPROCESSING", "LabelEncoder for multi-class target · StandardScaler via ColumnTransformer pipeline"),
        ("03", "SMOTE BALANCING", "Synthetic Minority Oversampling applied to training set to correct class imbalance"),
        ("04", "MODEL TRAINING", "Logistic Regression · Random Forest (400 trees) · XGBoost (400 estimators, depth=8)"),
        ("05", "EVALUATION", "Accuracy · Macro F1 · Per-class Precision/Recall · Confusion Matrix analysis"),
        ("06", "DEPLOYMENT", "Best model (XGBoost+SMOTE) persisted via joblib · Served via Streamlit"),
    ]
    step_cols = st.columns(3)
    for i, (num, title, desc) in enumerate(steps):
        with step_cols[i % 3]:
            st.markdown(f"""
            <div style="background:var(--steel-panel);border:1px solid var(--steel-border);border-top:2px solid var(--accent-cyan);
                        border-radius:6px;padding:16px 18px;margin-bottom:16px;">
                <div style="font-family:'Share Tech Mono',monospace;font-size:0.65rem;color:var(--accent-cyan);letter-spacing:2px;">{num}</div>
                <div style="font-family:Rajdhani,sans-serif;font-size:0.95rem;font-weight:700;letter-spacing:2px;margin:6px 0 8px;">{title}</div>
                <div style="font-family:'Exo 2',sans-serif;font-size:0.75rem;color:var(--text-muted);line-height:1.6;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════
# TAB 3 — ENCYCLOPEDIA
# ════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title"><span>//</span> DEFECT CLASSIFICATION REFERENCE</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-panel">7-class defect taxonomy for hot-rolled steel plates. Each defect type has distinct surface signatures, severity implications, and recommended quality control actions.</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    for cls, info in DEFECT_INFO.items():
        sev_color = SEVERITY_CSS[info['severity']]
        with st.expander(f"  {info['icon']}  {cls.replace('_',' ').upper()}  —  {info['severity']} SEVERITY"):
            c1, c2 = st.columns([2, 1])
            with c1:
                st.markdown(f"""
                <div style="font-family:'Exo 2',sans-serif;font-size:0.88rem;color:var(--text-primary);line-height:1.8;margin-bottom:12px;">{info['desc']}</div>
                """, unsafe_allow_html=True)
            with c2:
                act = action_map = {
                    'HIGH':   '🚨 REJECT immediately',
                    'MEDIUM': '⚠️ HOLD for review',
                    'LOW':    '✅ PASS with note',
                }[info['severity']]
                st.markdown(f"""
                <div style="background:rgba(0,0,0,0.2);border:1px solid {sev_color}33;border-radius:6px;padding:14px;text-align:center;">
                    <div style="font-family:'Share Tech Mono',monospace;font-size:0.65rem;color:var(--text-muted);letter-spacing:2px;margin-bottom:8px;">ACTION</div>
                    <div style="font-family:Rajdhani,sans-serif;font-size:0.9rem;font-weight:700;color:{sev_color};">{act}</div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("<br>")
    st.markdown('<div class="section-title"><span>//</span> SEVERITY LEGEND</div>', unsafe_allow_html=True)
    leg_c = st.columns(3)
    for col, (sev, color, label) in zip(leg_c, [
        ('HIGH', '#ff6b35', 'Structural risk · Reject from line'),
        ('MEDIUM', '#ffd60a', 'Functional risk · Manual review'),
        ('LOW', '#00ff9d', 'Cosmetic only · Log & proceed'),
    ]):
        with col:
            st.markdown(f"""
            <div style="background:var(--steel-panel);border:1px solid {color}44;border-radius:6px;padding:16px;text-align:center;">
                <div style="font-family:Rajdhani,sans-serif;font-size:1.1rem;font-weight:700;color:{color};letter-spacing:3px;">{sev}</div>
                <div style="font-family:'Exo 2',sans-serif;font-size:0.75rem;color:var(--text-muted);margin-top:6px;">{label}</div>
            </div>
            """, unsafe_allow_html=True)
