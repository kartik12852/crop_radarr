"""
utils/theme.py
Global Streamlit theme — light agricultural-tech aesthetic.
Inject once per page with inject_theme().
"""

import streamlit as st

# ------------------------------------------------------------------
# Design tokens
# ------------------------------------------------------------------
PRIMARY  = "#059669"
ACCENT   = "#0d9488"
DEEP     = "#F0FDF4"
CARD     = "rgba(255, 255, 255, 0.90)"
TEXT     = "#1a2e1a"
MUTED    = "#374151"

RISK_COLORS = {
    0: "#059669",   # Safe — emerald green
    1: "#D97706",   # Drought — amber
    2: "#EA580C",   # Heat Stress — orange
    3: "#0284C7",   # Flood — sky blue
    4: "#7C3AED",   # Soil Risk — violet
}

CSS = f"""
<style>
/* ── Global ──────────────────────────────────────────────── */
.stApp {{
    background:
        radial-gradient(circle at top left,  rgba(16,185,129,0.10),  transparent 30%),
        radial-gradient(circle at top right, rgba(56,189,248,0.08), transparent 30%),
        linear-gradient(180deg, #f0fdf4 0%, #f8fafc 50%, #f0fdf4 100%);
    color: {TEXT};
}}
[data-testid="stAppViewContainer"] > .main {{ background: transparent; }}

/* ── Sidebar ─────────────────────────────────────────────── */
[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #ecfdf5 0%, #d1fae5 42%, #e0f2fe 100%);
    border-right: 1px solid rgba(5,150,105,0.18);
}}
[data-testid="stSidebar"] * {{ color: #065f46 !important; }}

/* ── Layout ──────────────────────────────────────────────── */
.block-container {{
    padding-top: 1.1rem;
    padding-bottom: 2rem;
    max-width: 86rem;
}}
h1, h2, h3, h4 {{ color: #064e3b !important; letter-spacing: -0.02em; }}
p, li, label, .stMarkdown, .stCaption {{ color: {TEXT}; }}

/* ── Hero panel ──────────────────────────────────────────── */
.hero-panel {{
    background: linear-gradient(135deg,
        rgba(16,185,129,0.15), rgba(59,130,246,0.10));
    border: 1px solid rgba(5,150,105,0.22);
    box-shadow: 0 8px 24px rgba(5,150,105,0.10);
    backdrop-filter: blur(8px);
    border-radius: 22px;
    padding: 1.2rem 1.3rem;
    margin-bottom: 1rem;
}}
.hero-kicker {{
    display: inline-block;
    padding: 0.28rem 0.65rem;
    border-radius: 999px;
    background: rgba(5,150,105,0.12);
    color: #065f46;
    font-size: 0.84rem;
    border: 1px solid rgba(5,150,105,0.20);
    margin-bottom: 0.7rem;
}}

/* ── Metric tiles ────────────────────────────────────────── */
.metric-tile {{
    background: linear-gradient(180deg,
        rgba(255,255,255,0.95), rgba(240,253,244,0.90));
    border: 1px solid rgba(5,150,105,0.18);
    border-radius: 18px;
    padding: 0.95rem 1rem;
    box-shadow: 0 4px 16px rgba(5,150,105,0.08);
    text-align: center;
}}
.metric-label {{
    color: #374151;
    font-size: 0.82rem;
    opacity: 0.92;
    margin-bottom: 0.25rem;
}}
.metric-value {{
    color: #064e3b;
    font-size: 1.8rem;
    font-weight: 700;
    line-height: 1.15;
}}

/* ── Glass card ──────────────────────────────────────────── */
.glass-card {{
    background: {CARD};
    border: 1px solid rgba(5,150,105,0.14);
    border-radius: 18px;
    padding: 1rem 1.1rem;
    box-shadow: 0 4px 20px rgba(5,150,105,0.08);
    margin-bottom: 0.5rem;
}}

/* ── Info pills ──────────────────────────────────────────── */
.info-pill {{
    display: inline-block;
    margin: 0.2rem 0.35rem 0.2rem 0;
    padding: 0.34rem 0.7rem;
    border-radius: 999px;
    background: rgba(5,150,105,0.10);
    border: 1px solid rgba(5,150,105,0.18);
    color: #065f46;
    font-size: 0.82rem;
}}

/* ── Alert severity badges ───────────────────────────────── */
.badge-critical {{ background:#fee2e2; color:#991b1b; border-radius:8px; padding:2px 10px; font-size:0.82rem; border:1px solid #fca5a5; }}
.badge-high     {{ background:#ffedd5; color:#9a3412; border-radius:8px; padding:2px 10px; font-size:0.82rem; border:1px solid #fdba74; }}
.badge-medium   {{ background:#fef9c3; color:#854d0e; border-radius:8px; padding:2px 10px; font-size:0.82rem; border:1px solid #fde047; }}
.badge-low      {{ background:#dcfce7; color:#166534; border-radius:8px; padding:2px 10px; font-size:0.82rem; border:1px solid #86efac; }}

/* ── Tabs ────────────────────────────────────────────────── */
div[data-baseweb="tab-list"]           {{ gap: 0.4rem; }}
button[data-baseweb="tab"] {{
    background: rgba(255,255,255,0.80) !important;
    border-radius: 12px !important;
    border: 1px solid rgba(5,150,105,0.14) !important;
    color: #064e3b !important;
}}
button[data-baseweb="tab"][aria-selected="true"] {{
    background: linear-gradient(90deg,
        rgba(16,185,129,0.20), rgba(59,130,246,0.12)) !important;
    border-color: rgba(5,150,105,0.30) !important;
    font-weight: 600 !important;
}}

/* ── Widgets ─────────────────────────────────────────────── */
.stDataFrame, .stPlotlyChart, .stPydeckChart {{
    border-radius: 16px;
    overflow: hidden;
}}
.stAlert    {{ border-radius: 14px; }}
.stButton > button {{
    background: linear-gradient(135deg, #059669, #0d9488);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    transition: opacity 0.2s;
}}
.stButton > button:hover {{ opacity: 0.88; }}

/* ── Input fields ────────────────────────────────────────── */
.stSelectbox > div, .stMultiSelect > div {{
    background: white !important;
    border-color: rgba(5,150,105,0.25) !important;
}}
.stTextInput > div > input {{
    background: white !important;
    color: #1a2e1a !important;
    border-color: rgba(5,150,105,0.25) !important;
}}
</style>
"""


def inject_theme() -> None:
    """Call this once at the top of every page."""
    st.markdown(CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str, kicker: str = "Climate Intelligence") -> None:
    st.markdown(
        f"""
        <div class="hero-panel">
            <div class="hero-kicker">{kicker}</div>
            <h1 style="margin:0 0 0.3rem 0;">{title}</h1>
            <p style="margin:0;color:{MUTED};font-size:1.02rem;">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_tile(label: str, value: str) -> str:
    return (
        f'<div class="metric-tile">'
        f'  <div class="metric-label">{label}</div>'
        f'  <div class="metric-value">{value}</div>'
        f'</div>'
    )


def pills(items: list[str]) -> None:
    html = "".join(f'<span class="info-pill">{item}</span>' for item in items)
    st.markdown(html, unsafe_allow_html=True)


def severity_badge(severity: str) -> str:
    cls = {"Critical": "critical", "High": "high", "Medium": "medium"}.get(severity, "low")
    return f'<span class="badge-{cls}">{severity}</span>'