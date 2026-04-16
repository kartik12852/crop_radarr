"""
utils/theme.py
Shared Streamlit theme helpers for Climate Crop Radar.
"""

import streamlit as st

PRIMARY = "#7CFFB2"
ACCENT = "#55D6BE"
DEEP = "#0F172A"
CARD = "rgba(15, 23, 42, 0.70)"
TEXT = "#E5F7EE"
MUTED = "#A7F3D0"

RISK_COLORS = {
    0: "#34D399",
    1: "#F59E0B",
    2: "#F97316",
    3: "#38BDF8",
    4: "#A78BFA",
}


def inject_theme() -> None:
    st.markdown(
        f"""
        <style>
        .stApp {{
            background:
                radial-gradient(circle at top left, rgba(34,197,94,0.18), transparent 26%),
                radial-gradient(circle at top right, rgba(56,189,248,0.16), transparent 24%),
                linear-gradient(180deg, #07111f 0%, #0b1220 48%, #07111f 100%);
            color: {TEXT};
        }}
        [data-testid="stAppViewContainer"] > .main {{
            background: transparent;
        }}
        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #0b3b2e 0%, #0f5132 42%, #0b2136 100%);
            border-right: 1px solid rgba(124,255,178,0.14);
        }}
        [data-testid="stSidebar"] * {{
            color: #E8FFF1 !important;
        }}
        .block-container {{
            padding-top: 1.1rem;
            padding-bottom: 2rem;
            max-width: 86rem;
        }}
        h1, h2, h3, h4 {{
            color: #F8FFFB !important;
            letter-spacing: -0.02em;
        }}
        p, li, label, .stMarkdown, .stCaption {{
            color: {TEXT};
        }}
        .hero-panel {{
            background: linear-gradient(135deg, rgba(16,185,129,0.20), rgba(59,130,246,0.18));
            border: 1px solid rgba(124,255,178,0.16);
            box-shadow: 0 16px 40px rgba(0,0,0,0.24);
            backdrop-filter: blur(12px);
            border-radius: 22px;
            padding: 1.2rem 1.3rem;
            margin-bottom: 1rem;
        }}
        .hero-kicker {{
            display: inline-block;
            padding: 0.28rem 0.65rem;
            border-radius: 999px;
            background: rgba(124,255,178,0.14);
            color: #D1FAE5;
            font-size: 0.84rem;
            border: 1px solid rgba(124,255,178,0.16);
            margin-bottom: 0.7rem;
        }}
        .metric-tile {{
            background: linear-gradient(180deg, rgba(15,23,42,0.84), rgba(17,24,39,0.72));
            border: 1px solid rgba(124,255,178,0.12);
            border-radius: 18px;
            padding: 0.95rem 1rem;
            box-shadow: 0 12px 30px rgba(0,0,0,0.18);
        }}
        .metric-label {{
            color: #A7F3D0;
            font-size: 0.82rem;
            opacity: 0.92;
        }}
        .metric-value {{
            color: #F8FFFB;
            font-size: 1.8rem;
            font-weight: 700;
            line-height: 1.15;
        }}
        .glass-card {{
            background: {CARD};
            border: 1px solid rgba(124,255,178,0.10);
            border-radius: 18px;
            padding: 1rem 1.1rem;
            box-shadow: 0 12px 32px rgba(0,0,0,0.16);
        }}
        .info-pill {{
            display: inline-block;
            margin: 0.2rem 0.35rem 0.2rem 0;
            padding: 0.34rem 0.7rem;
            border-radius: 999px;
            background: rgba(148,163,184,0.16);
            border: 1px solid rgba(148,163,184,0.20);
            color: #E2E8F0;
            font-size: 0.82rem;
        }}
        div[data-baseweb="tab-list"] {{
            gap: 0.4rem;
        }}
        button[data-baseweb="tab"] {{
            background: rgba(15,23,42,0.72) !important;
            border-radius: 12px !important;
            border: 1px solid rgba(124,255,178,0.10) !important;
            color: #EAFBF3 !important;
        }}
        button[data-baseweb="tab"][aria-selected="true"] {{
            background: linear-gradient(90deg, rgba(16,185,129,0.18), rgba(59,130,246,0.18)) !important;
            border-color: rgba(124,255,178,0.22) !important;
        }}
        .stDataFrame, .stPlotlyChart, .stPydeckChart {{
            border-radius: 16px;
            overflow: hidden;
        }}
        .stAlert {{
            border-radius: 14px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


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
    return f"""
    <div class="metric-tile">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
    </div>
    """


def pills(items: list[str]) -> None:
    html = "".join([f'<span class="info-pill">{item}</span>' for item in items])
    st.markdown(html, unsafe_allow_html=True)
