"""
Pocket Fund — Acquisition Intelligence Layer
India Micro-PE Agent Swarm | Investment Committee Interface

Run: streamlit run app.py

DISCLAIMER: Prototype with mock/simulated data. Public sources only.
Not financial advice. India regulatory signals are simulated, not from actual MCA/GST APIs.
"""

# ── streamlit page config must be FIRST ──────────────────────────────────────
import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="Pocket Fund | Acquisition Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Optional favicon injection for hosts that request /favicon.ico
favicon_path = Path("static/favicon.ico")
if favicon_path.exists():
    st.markdown(
        """
        <link rel="shortcut icon" href="/static/favicon.ico">
        """,
        unsafe_allow_html=True,
    )

import json
import logging
import os
import re
import time
import uuid
from datetime import datetime
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# ── Auto-load secrets into env vars on startup ────────────────────────────────
def _load_secrets():
    """Pull Streamlit secrets into os.environ so all modules can access them."""
    try:
        secrets = st.secrets
        for key in (
            "OPENAI_API_KEY",
            "REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET", "REDDIT_USER_AGENT",
            "FIRECRAWL_API_KEY",
            "ACQUIRE_EMAIL", "ACQUIRE_PASSWORD",
        ):
            if key in secrets and not os.environ.get(key):
                os.environ[key] = secrets[key]
    except Exception:
        pass  # secrets.toml not present — keys must be entered in sidebar

_load_secrets()

# ── Internal imports ──────────────────────────────────────────────────────────
from agents.analyst import ValuationAnalyst
from agents.psychologist import FounderPsychologist, MOTIVATION_PROFILES
from agents.legal_reg import LegalRegAgent, RISK_LEVELS
from agents.strategy import StrategyAgent
from engine.memory import DealMemory
from engine.crawler import (
    scrape_listing_url,
    crawl_acquire_listings,
    scrape_founder_profile,
    research_url,
    check_firecrawl_status,
)

logging.basicConfig(level=logging.WARNING)
KB_PATH = Path(__file__).parent / "data" / "sme_knowledge.json"

# ─────────────────────────────────────────────────────────────────────────────
# IC-Style CSS
# ─────────────────────────────────────────────────────────────────────────────

IC_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background: #0b0f1a !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', -apple-system, sans-serif !important;
}
[data-testid="stSidebar"] {
    background: #0d1120 !important;
    border-right: 1px solid #1e2d4a !important;
}
[data-testid="stSidebar"] * { color: #94a3b8 !important; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3, [data-testid="stSidebar"] strong { color: #e2e8f0 !important; }

/* ── IC Header ── */
.ic-header {
    background: linear-gradient(135deg, #0d1120 0%, #131d35 100%);
    border: 1px solid #1e3a5f;
    border-left: 4px solid #2563eb;
    border-radius: 8px;
    padding: 20px 28px;
    margin-bottom: 24px;
}
.ic-header .fund-name {
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 4px;
    color: #60a5fa;
    text-transform: uppercase;
}
.ic-header .page-title {
    font-size: 24px;
    font-weight: 700;
    color: #f1f5f9;
    margin: 4px 0;
}
.ic-header .subtitle {
    font-size: 13px;
    color: #475569;
    font-family: 'JetBrains Mono', monospace;
}
.live-dot {
    display: inline-block; width:7px; height:7px;
    background:#10b981; border-radius:50%;
    animation: blink 1.5s infinite;
    margin-right:5px;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.2} }

/* ── Metric Cards ── */
.metric-grid { display:flex; gap:12px; margin:16px 0; flex-wrap:wrap; }
.metric-card {
    background:#0d1120; border:1px solid #1e3a5f; border-radius:8px;
    padding:16px 20px; flex:1; min-width:130px;
}
.metric-card .m-label {
    font-family:'JetBrains Mono',monospace; font-size:9px;
    letter-spacing:2px; color:#475569; text-transform:uppercase; margin-bottom:6px;
}
.metric-card .m-value {
    font-family:'JetBrains Mono',monospace; font-size:24px;
    font-weight:700; color:#f59e0b;
}
.metric-card .m-sub { font-size:11px; color:#374151; margin-top:3px; }

/* ── Panels ── */
.ic-panel {
    background:#0d1120; border:1px solid #1e3a5f; border-radius:8px;
    padding:20px 24px; margin-bottom:16px;
}
.panel-header {
    font-family:'JetBrains Mono',monospace; font-size:10px;
    letter-spacing:3px; color:#475569; text-transform:uppercase;
    border-bottom:1px solid #1e2d4a; padding-bottom:10px; margin-bottom:16px;
}

/* ── Risk Badges ── */
.risk-badge {
    display:inline-flex; align-items:center; gap:4px;
    padding:3px 10px 3px 6px; border-radius:4px;
    font-family:'JetBrains Mono',monospace; font-size:10px; font-weight:600;
    letter-spacing:1px; margin:3px 3px 3px 0;
}
.badge-green  { background:#022c22; color:#10b981; border:1px solid #065f46; }
.badge-yellow { background:#2d1f00; color:#f59e0b; border:1px solid #92400e; }
.badge-red    { background:#2d0000; color:#ef4444; border:1px solid #7f1d1d; }
.badge-purple { background:#1e1145; color:#a78bfa; border:1px solid #4c1d95; }
.badge-blue   { background:#0c1a2e; color:#60a5fa; border:1px solid #1e3a5f; }
.badge-gray   { background:#111827; color:#6b7280; border:1px solid #374151; }

/* ── Score bar ── */
.score-row { margin:6px 0; }
.score-label {
    font-family:'JetBrains Mono',monospace; font-size:10px; color:#475569;
    letter-spacing:1px; text-transform:uppercase; margin-bottom:3px;
    display:flex; justify-content:space-between;
}
.score-track { background:#1e2d4a; border-radius:2px; height:5px; overflow:hidden; }
.score-fill { height:5px; border-radius:2px; }

/* ── Deal Memory Card ── */
.memory-card {
    background:#0a1628; border:1px solid #1e3a5f; border-left:3px solid #2563eb;
    border-radius:6px; padding:12px 16px; margin:6px 0; font-size:12px;
}
.memory-match { font-family:'JetBrains Mono',monospace; font-size:10px; color:#2563eb; }

/* ── Input tabs ── */
.stTabs [data-baseweb="tab-list"] { background:transparent !important; gap:4px; }
.stTabs [data-baseweb="tab"] {
    background:#0d1120 !important; border:1px solid #1e2d4a !important;
    color:#6b7280 !important; font-family:'JetBrains Mono',monospace !important;
    font-size:11px !important; letter-spacing:1px !important;
    border-radius:6px !important; padding:8px 20px !important;
}
.stTabs [aria-selected="true"] {
    background:#131d35 !important; color:#60a5fa !important; border-color:#2563eb !important;
}

/* ── Buttons ── */
.stButton > button {
    background:transparent !important; border:1px solid #2563eb !important;
    color:#60a5fa !important; font-family:'JetBrains Mono',monospace !important;
    font-size:11px !important; letter-spacing:1px !important;
    border-radius:6px !important; transition:all 0.2s !important;
}
.stButton > button:hover { background:#2563eb !important; color:#fff !important; }

/* ── Text inputs / selects ── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea,
.stSelectbox > div > div {
    background:#0d1120 !important; border:1px solid #1e2d4a !important;
    color:#e2e8f0 !important; border-radius:6px !important;
}
label { color:#94a3b8 !important; font-size:12px !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width:4px; height:4px; }
::-webkit-scrollbar-track { background:#0b0f1a; }
::-webkit-scrollbar-thumb { background:#1e3a5f; border-radius:2px; }
hr { border-color:#1e2d4a !important; }

/* ── Expander ── */
.streamlit-expanderHeader {
    background:#0d1120 !important; border:1px solid #1e2d4a !important;
    color:#94a3b8 !important; border-radius:6px !important;
    font-family:'JetBrains Mono',monospace !important; font-size:12px !important;
}

/* ── Agent log ── */
.agent-log {
    background:#060a12; border:1px solid #1e2d4a; border-radius:6px;
    padding:12px 16px; font-family:'JetBrains Mono',monospace; font-size:11px;
    color:#475569; max-height:200px; overflow-y:auto; margin-top:8px;
}
.log-sourcer  { color:#2563eb; }
.log-analyst  { color:#f59e0b; }
.log-psych    { color:#a78bfa; }
.log-legal    { color:#ef4444; }
.log-strategy { color:#10b981; }
.log-memory   { color:#60a5fa; }
</style>
"""
st.markdown(IC_CSS, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Session State
# ─────────────────────────────────────────────────────────────────────────────

def _init_state():
    defaults = {
        "memory": None,
        "analysis_result": None,
        "agent_logs": [],
        "llm": None,
        "llm_provider": "none",
        "run_count": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ─────────────────────────────────────────────────────────────────────────────
# Memory Initialization (cached per session)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _get_memory() -> DealMemory:
    mem = DealMemory()
    mem.seed_demo_data()
    return mem

MEMORY: DealMemory = _get_memory()

# ─────────────────────────────────────────────────────────────────────────────
# LLM Factory
# ─────────────────────────────────────────────────────────────────────────────

def _build_llm(provider: str, api_key: str, model: str):
    if not api_key or provider == "none":
        return None
    try:
        if provider == "openai":
            from langchain_openai import ChatOpenAI
            from langchain_core.messages import HumanMessage

            llm = ChatOpenAI(model=model, api_key=api_key, temperature=0.3)
            # Warm-up ping to verify key is valid
            llm.invoke([HumanMessage(content="ping")])
            return llm
        elif provider == "groq":
            from langchain_groq import ChatGroq
            return ChatGroq(model=model, api_key=api_key, temperature=0.3)
    except Exception as e:
        st.sidebar.warning(f"LLM init failed: {e}")
    return None

# ─────────────────────────────────────────────────────────────────────────────
# Agent Log Helper
# ─────────────────────────────────────────────────────────────────────────────

def _log(msg: str, agent: str = "system"):
    entry = f"[{datetime.utcnow().strftime('%H:%M:%S')}] [{agent.upper()}] {msg}"
    st.session_state.agent_logs.append({"text": entry, "agent": agent})

# ─────────────────────────────────────────────────────────────────────────────
# Master Orchestration Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_analysis_pipeline(deal_input: dict, llm=None) -> dict:
    """
    Chain: Memory Lookup → Valuation → Psychology → Legal/Reg → Strategy → Save
    deal_input: {title, description, sector, state, revenue_cr, ebitda_l,
                 is_family_run, gst_registered, udyam_registered, digital_ready,
                 business_type, seller_name, transcript, founder_signals}
    """
    st.session_state.agent_logs = []
    t_start = time.time()

    title = deal_input.get("title", "Unnamed Deal")
    description = deal_input.get("description", "")
    transcript = deal_input.get("transcript", "")
    founder_signals = deal_input.get("founder_signals", {})

    # ── 1. Memory: find similar past deals ───────────────────────────────────
    _log("Querying vector memory for similar deals...", "memory")
    search_query = f"{deal_input.get('sector','')} {deal_input.get('state','')} {title} {description[:200]}"
    similar_deals = MEMORY.find_similar(search_query, k=3)
    memory_context = MEMORY.build_agent_context()
    memory_notes = [d.get("memory_note", "") for d in similar_deals if d.get("memory_note")]
    _log(f"Found {len(similar_deals)} similar deals in memory (backend={MEMORY.get_stats()['backend']})", "memory")

    # ── 2. Psychologist Agent ─────────────────────────────────────────────────
    _log("Analyzing founder psychology (VADER + keyword NLP)...", "psych")
    psych_agent = FounderPsychologist(llm=llm)
    combined_text = description + " " + transcript
    psych_data = psych_agent.analyze(
        listing_text=combined_text,
        founder_signals=founder_signals,
        transcript=transcript if transcript else None,
    )
    psych_narrative = psych_agent.generate_psych_narrative(psych_data, title)
    _log(f"Motivation: {psych_data['motivation_type']} | Burnout={psych_data['burnout_score']} | Conviction={psych_data['conviction_score']}", "psych")

    # Use transcript extracts to fill in missing deal data
    t_extracts = psych_data.get("transcript_extracts", {})
    if t_extracts.get("detected_sector") and not deal_input.get("sector"):
        deal_input["sector"] = t_extracts["detected_sector"]

    # ── 3. Valuation Analyst ─────────────────────────────────────────────────
    _log("Running India-adjusted valuation model...", "analyst")
    analyst = ValuationAnalyst(llm=llm)
    val_data = analyst.compute_valuation(
        sector_key=deal_input.get("sector", "manufacturing_general"),
        revenue_cr=deal_input.get("revenue_cr", 1.0),
        ebitda_l=deal_input.get("ebitda_l", 10.0),
        is_family_run=deal_input.get("is_family_run", False),
        gst_clean=deal_input.get("gst_registered", True),
        udyam_registered=deal_input.get("udyam_registered", False),
        digital_ready=deal_input.get("digital_ready", False),
        state=deal_input.get("state", "Maharashtra"),
        conviction_score=psych_data["conviction_score"],
        burnout_score=psych_data["burnout_score"],
    )
    val_narrative = analyst.generate_narrative(val_data, title, memory_notes)
    comp_chart_data = analyst.generate_comps_chart_data(
        deal_input.get("sector", "manufacturing_general"),
        val_data["blended_valuation_mid_cr"],
        similar_deals,
    )
    _log(f"Valuation: ₹{val_data['blended_valuation_low_cr']}–{val_data['blended_valuation_high_cr']}Cr | Adj={val_data['total_adjustment_pct']}%", "analyst")

    # ── 4. Legal/Reg Agent ───────────────────────────────────────────────────
    _log("Flagging India legal & regulatory risks...", "legal")
    legal_agent = LegalRegAgent(llm=llm)
    reg_data = legal_agent.analyze(
        sector_key=deal_input.get("sector", "manufacturing_general"),
        state=deal_input.get("state", "Maharashtra"),
        business_type=deal_input.get("business_type", "Private Limited"),
        gst_registered=deal_input.get("gst_registered", True),
        udyam_registered=deal_input.get("udyam_registered", False),
        is_family_run=deal_input.get("is_family_run", False),
        revenue_cr=deal_input.get("revenue_cr", 1.0),
        special_risks=deal_input.get("special_risks", []),
    )
    reg_narrative = legal_agent.generate_reg_narrative(reg_data, title)
    _log(f"Compliance={reg_data['compliance_score']}/100 | India Risk={reg_data['india_risk_score']}/10 | Flags={len(reg_data['flags'])}", "legal")

    # ── 5. Strategy Agent ────────────────────────────────────────────────────
    _log("Generating outreach + deal strategy...", "strategy")
    strategy_agent = StrategyAgent(llm=llm)
    outreach = strategy_agent.generate_outreach(
        motivation_type=psych_data["motivation_type"],
        seller_name=deal_input.get("seller_name", "Founder"),
        business_name=title,
        psych_data=psych_data,
        reg_data=reg_data,
        memory_context=memory_notes[0] if memory_notes else "",
    )
    playbook = strategy_agent.generate_negotiation_playbook(
        motivation_type=psych_data["motivation_type"],
        valuation_data=val_data,
        psych_data=psych_data,
        reg_data=reg_data,
        similar_deals=similar_deals,
    )
    strategy_summary = strategy_agent.generate_strategy_summary(
        deal_context=f"{title}: {description[:300]}",
        motivation_type=psych_data["motivation_type"],
        valuation_data=val_data,
        psych_data=psych_data,
        reg_data=reg_data,
        memory_context=memory_notes[0] if memory_notes else "",
    )
    _log("Outreach drafted | Playbook generated", "strategy")

    # ── 6. Compute Conviction Score + Verdict ────────────────────────────────
    motivation_score = psych_data["burnout_score"] * 0.35 + psych_data["opportunity_exit_score"] * 0.15
    price_score = 10 - min(reg_data["india_risk_score"], 10)
    compliance_score_norm = reg_data["compliance_score"] / 10
    conviction_raw = motivation_score * 0.4 + psych_data["conviction_score"] * 0.2 + compliance_score_norm * 0.2 + price_score * 0.2
    conviction_final = round(min(conviction_raw, 10), 1)

    if conviction_final >= 7.5:
        verdict = "HOT"
        verdict_color = "#ef4444"
    elif conviction_final >= 5:
        verdict = "WARM"
        verdict_color = "#f59e0b"
    else:
        verdict = "PASS"
        verdict_color = "#6b7280"

    # ── 7. LOI ───────────────────────────────────────────────────────────────
    loi_text = None
    if conviction_final >= 6.5:
        loi_text = strategy_agent.generate_loi(
            business_name=title,
            seller_name=deal_input.get("seller_name", "Founder"),
            purchase_price_inr_cr=val_data["blended_valuation_mid_cr"],
            structure_type=reg_data["deal_structure_recommendation"]["recommended_structure"],
            business_type=deal_input.get("business_type", "Private Limited"),
        )

    elapsed = round(time.time() - t_start, 1)
    _log(f"Pipeline complete in {elapsed}s | Verdict={verdict} | Conviction={conviction_final}/10", "system")

    # ── 8. Save to Memory ────────────────────────────────────────────────────
    deal_record = {
        "id": f"deal-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{title[:10].replace(' ','')}",
        "title": title,
        "summary": f"{deal_input.get('state','?')} | {val_data['sector_label']} | Rev=₹{deal_input.get('revenue_cr','?')}Cr | {psych_data['motivation_type']}",
        "sector": deal_input.get("sector", ""),
        "state": deal_input.get("state", ""),
        "revenue_cr": deal_input.get("revenue_cr", 0),
        "ebitda_l": deal_input.get("ebitda_l", 0),
        "verdict": verdict,
        "motivation": psych_data["motivation_type"],
        "conviction_score": conviction_final,
        "burnout_score": psych_data["burnout_score"],
        "india_risk": reg_data["india_risk_score"],
        "deal_outcome": "Analysed — pending action",
    }
    MEMORY.add_deal(deal_record)
    st.session_state.run_count += 1

    return {
        "title": title,
        "verdict": verdict,
        "verdict_color": verdict_color,
        "conviction_score": conviction_final,
        "elapsed": elapsed,
        "val_data": val_data,
        "val_narrative": val_narrative,
        "comp_chart_data": comp_chart_data,
        "psych_data": psych_data,
        "psych_narrative": psych_narrative,
        "reg_data": reg_data,
        "reg_narrative": reg_narrative,
        "strategy_summary": strategy_summary,
        "outreach": outreach,
        "playbook": playbook,
        "loi_text": loi_text,
        "similar_deals": similar_deals,
        "memory_notes": memory_notes,
        "deal_input": deal_input,
    }

# ─────────────────────────────────────────────────────────────────────────────
# Chart Builders
# ─────────────────────────────────────────────────────────────────────────────

def chart_valuation_gauge(val_data: dict) -> go.Figure:
    low = val_data["blended_valuation_low_cr"]
    mid = val_data["blended_valuation_mid_cr"]
    high = val_data["blended_valuation_high_cr"]
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=mid,
        number={"suffix": " Cr", "font": {"family": "JetBrains Mono", "size": 28, "color": "#f59e0b"}},
        delta={"reference": low, "suffix": " Cr", "increasing": {"color": "#10b981"}, "decreasing": {"color": "#ef4444"}},
        title={"text": "Blended Valuation (₹ Crore)", "font": {"family": "Inter", "size": 12, "color": "#94a3b8"}},
        gauge={
            "axis": {"range": [0, high * 1.4], "tickcolor": "#475569", "tickfont": {"family": "JetBrains Mono", "size": 9}},
            "bar": {"color": "#f59e0b", "thickness": 0.25},
            "bgcolor": "#0d1120",
            "borderwidth": 0,
            "steps": [
                {"range": [0, low], "color": "#1e2d4a"},
                {"range": [low, high], "color": "#193044"},
                {"range": [high, high * 1.4], "color": "#0d1120"},
            ],
            "threshold": {
                "line": {"color": "#10b981", "width": 2},
                "thickness": 0.75,
                "value": mid,
            },
        },
    ))
    fig.update_layout(
        paper_bgcolor="#0d1120", plot_bgcolor="#0d1120",
        font={"family": "Inter", "color": "#94a3b8"},
        height=220, margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def chart_momentum_radar(radar_data: dict) -> go.Figure:
    categories = list(radar_data.keys())
    values = list(radar_data.values())
    values_closed = values + [values[0]]
    categories_closed = categories + [categories[0]]

    fig = go.Figure(go.Scatterpolar(
        r=values_closed,
        theta=categories_closed,
        fill="toself",
        fillcolor="rgba(239, 68, 68, 0.12)",
        line=dict(color="#ef4444", width=1.5),
        marker=dict(color="#ef4444", size=6),
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="#0d1120",
            radialaxis=dict(
                visible=True, range=[0, 10],
                gridcolor="#1e2d4a", tickcolor="#475569",
                tickfont={"family": "JetBrains Mono", "size": 8, "color": "#475569"},
                tickvals=[2, 4, 6, 8, 10],
            ),
            angularaxis=dict(
                gridcolor="#1e2d4a", tickfont={"family": "JetBrains Mono", "size": 9, "color": "#94a3b8"}
            ),
        ),
        paper_bgcolor="#0d1120",
        font={"family": "Inter", "color": "#94a3b8"},
        height=240, margin=dict(l=30, r=30, t=20, b=20),
        showlegend=False,
    )
    return fig


def chart_comps_bar(comp_data: list[dict]) -> go.Figure:
    labels = [c["label"] for c in comp_data]
    values = [c["value"] for c in comp_data]
    colors = [c["color"] for c in comp_data]

    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker_color=colors, text=[f"₹{v:.1f}Cr" for v in values],
        textfont={"family": "JetBrains Mono", "size": 10, "color": "#e2e8f0"},
        textposition="outside",
    ))
    fig.update_layout(
        paper_bgcolor="#0d1120", plot_bgcolor="#0d1120",
        font={"family": "JetBrains Mono", "color": "#94a3b8"},
        xaxis=dict(gridcolor="#1e2d4a", tickfont={"size": 9}, title="Valuation (₹ Crore)"),
        yaxis=dict(gridcolor="#1e2d4a", tickfont={"size": 9}),
        height=max(140, len(comp_data) * 40 + 40),
        margin=dict(l=10, r=80, t=10, b=30),
        bargap=0.3,
    )
    return fig


def chart_state_density(state_name: str) -> go.Figure:
    kb = json.load(open(KB_PATH))
    states = kb["india_state_profiles"]
    names = list(states.keys())
    scores = [states[s]["deal_density_score"] for s in names]
    colors = ["#2563eb" if s == state_name else "#1e3a5f" for s in names]

    fig = go.Figure(go.Bar(
        x=names, y=scores,
        marker_color=colors,
        text=scores, textfont={"size": 9, "color": "#e2e8f0"},
        textposition="outside",
    ))
    fig.update_layout(
        paper_bgcolor="#0d1120", plot_bgcolor="#0d1120",
        font={"family": "JetBrains Mono", "color": "#94a3b8"},
        xaxis=dict(gridcolor="#1e2d4a", tickfont={"size": 9}),
        yaxis=dict(gridcolor="#1e2d4a", tickfont={"size": 9}, title="Deal Density"),
        height=200, margin=dict(l=10, r=10, t=10, b=40),
    )
    return fig


def chart_compliance_waterfall(adjustments: dict) -> go.Figure:
    labels, values, colors = [], [], []
    for k, v in adjustments.items():
        adj = v["value"]
        if adj != 0:
            labels.append(k[:35])
            values.append(round(adj * 100, 1))
            colors.append("#10b981" if adj > 0 else "#ef4444")

    if not labels:
        labels = ["No adjustments"]
        values = [0]
        colors = ["#475569"]

    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker_color=colors,
        text=[f"{'+' if v > 0 else ''}{v}%" for v in values],
        textfont={"size": 9, "family": "JetBrains Mono"},
        textposition="outside",
    ))
    fig.update_layout(
        paper_bgcolor="#0d1120", plot_bgcolor="#0d1120",
        font={"family": "JetBrains Mono", "color": "#94a3b8"},
        xaxis=dict(gridcolor="#1e2d4a", tickfont={"size": 9}, title="Adjustment %"),
        yaxis=dict(gridcolor="#1e2d4a", tickfont={"size": 9}),
        height=max(140, len(labels) * 30 + 50),
        margin=dict(l=10, r=80, t=10, b=30),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# HTML Render Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _score_bar(label: str, value: float, color: str) -> str:
    pct = int(value * 10)
    return f"""
<div class="score-row">
  <div class="score-label"><span>{label}</span><span style="color:{color}">{value}/10</span></div>
  <div class="score-track"><div class="score-fill" style="width:{pct}%;background:{color}"></div></div>
</div>"""


def _badge(level: str, text: str = "") -> str:
    cls = {"GREEN": "badge-green", "YELLOW": "badge-yellow", "RED": "badge-red",
           "BLOCKER": "badge-red", "PURPLE": "badge-purple", "BLUE": "badge-blue"}.get(level, "badge-gray")
    icon = RISK_LEVELS.get(level, {}).get("icon", "")
    return f'<span class="risk-badge {cls}">{icon} {text or level}</span>'


def _verdict_pill(verdict: str, score: float) -> str:
    colors = {"HOT": ("#ef4444", "#2d0000"), "WARM": ("#f59e0b", "#2d1f00"), "PASS": ("#6b7280", "#111827")}
    c, bg = colors.get(verdict, ("#6b7280", "#111827"))
    return f'<span style="background:{bg};color:{c};border:1px solid {c};padding:4px 14px;border-radius:4px;font-family:JetBrains Mono;font-size:12px;font-weight:700;letter-spacing:2px;">{verdict} · {score}/10</span>'


def _memory_card(note: str, match_deal: dict) -> str:
    title = match_deal.get("title", "Past Deal")[:55]
    verdict = match_deal.get("verdict", "?")
    outcome = match_deal.get("deal_outcome", "N/A")
    v_color = {"HOT": "#ef4444", "WARM": "#f59e0b", "PASS": "#6b7280"}.get(verdict, "#6b7280")
    return f"""
<div class="memory-card">
  <div class="memory-match">⟳ MEMORY MATCH</div>
  <div style="margin:4px 0;color:#e2e8f0;font-size:12px;font-weight:500;">{title}...</div>
  <div style="font-size:11px;color:#475569;">{note[:180]}</div>
  <div style="margin-top:6px;">
    <span style="color:{v_color};font-family:JetBrains Mono;font-size:10px;font-weight:700;">{verdict}</span>
    <span style="color:#374151;font-size:10px;margin-left:8px;">Outcome: {outcome}</span>
  </div>
</div>"""


# ─────────────────────────────────────────────────────────────────────────────
# Results Dashboard Renderer
# ─────────────────────────────────────────────────────────────────────────────

def render_results(result: dict):
    val_data = result["val_data"]
    psych_data = result["psych_data"]
    reg_data = result["reg_data"]
    playbook = result["playbook"]

    # ── Summary Header ────────────────────────────────────────────────────────
    st.markdown(f"""
<div class="ic-panel" style="border-left:4px solid {result['verdict_color']}">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:12px;">
    <div>
      <div style="font-size:18px;font-weight:700;color:#f1f5f9;margin-bottom:6px;">{result['title']}</div>
      <div style="font-size:12px;color:#475569;font-family:JetBrains Mono;">
        {val_data['sector_label']} · {result['deal_input'].get('state','?')} · 
        Rev ₹{result['deal_input'].get('revenue_cr','?')}Cr · 
        EBITDA ₹{result['deal_input'].get('ebitda_l','?')}L · 
        Analysis: {result['elapsed']}s
      </div>
    </div>
    <div style="text-align:right;">
      {_verdict_pill(result['verdict'], result['conviction_score'])}
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── Memory Context ────────────────────────────────────────────────────────
    if result["similar_deals"]:
        st.markdown('<div class="panel-header">⟳ VECTOR MEMORY — COMPARABLE PAST DEALS</div>', unsafe_allow_html=True)
        for deal in result["similar_deals"][:2]:
            full = deal.get("full_deal", deal.get("metadata", {}))
            st.markdown(_memory_card(deal.get("memory_note", ""), full), unsafe_allow_html=True)
        st.markdown("")

    # ── Tab layout for results ────────────────────────────────────────────────
    r1, r2, r3, r4 = st.tabs(["📊 VALUATION", "🧠 PSYCHOLOGY", "⚖️ LEGAL/REG", "🎯 STRATEGY"])

    # ════════════════ VALUATION TAB ══════════════════════════════════════════
    with r1:
        col_gauge, col_adj, col_comps = st.columns([1, 1, 1])

        with col_gauge:
            st.markdown('<div class="panel-header">BLENDED VALUATION RANGE</div>', unsafe_allow_html=True)
            st.plotly_chart(chart_valuation_gauge(val_data), use_container_width=True)
            st.markdown(f"""
<div style="font-family:JetBrains Mono;font-size:11px;color:#475569;text-align:center;">
  LOW ₹{val_data['blended_valuation_low_cr']}Cr · MID ₹{val_data['blended_valuation_mid_cr']}Cr · HIGH ₹{val_data['blended_valuation_high_cr']}Cr
</div>""", unsafe_allow_html=True)

        with col_adj:
            st.markdown('<div class="panel-header">INDIA ADJUSTMENT STACK</div>', unsafe_allow_html=True)
            st.plotly_chart(chart_compliance_waterfall(val_data["adjustments"]), use_container_width=True)
            total_adj = val_data["total_adjustment_pct"]
            adj_color = "#10b981" if total_adj > 0 else "#ef4444"
            st.markdown(
                f'<div style="font-family:JetBrains Mono;font-size:11px;color:{adj_color};text-align:center;">'
                f'Net adjustment: {"+" if total_adj > 0 else ""}{total_adj}% from base</div>',
                unsafe_allow_html=True
            )

        with col_comps:
            st.markdown('<div class="panel-header">COMPARABLE DEALS</div>', unsafe_allow_html=True)
            if result["comp_chart_data"]:
                st.plotly_chart(chart_comps_bar(result["comp_chart_data"]), use_container_width=True)

        st.markdown('<div class="panel-header">ANALYST NARRATIVE</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="ic-panel" style="font-size:13px;line-height:1.7;color:#cbd5e1;">'
            f'{result["val_narrative"].replace(chr(10), "<br>")}</div>',
            unsafe_allow_html=True
        )

        # Key metrics grid
        msme = val_data["msme_class"]
        st.markdown(f"""
<div class="metric-grid">
  <div class="metric-card"><div class="m-label">EBITDA Margin</div><div class="m-value" style="color:#10b981">{val_data['ebitda_margin_pct']}%</div></div>
  <div class="metric-card"><div class="m-label">Base Multiple</div><div class="m-value">{val_data['base_ebitda_multiple_range']}</div><div class="m-sub">EBITDA</div></div>
  <div class="metric-card"><div class="m-label">Adj Multiple</div><div class="m-value" style="color:#f59e0b">{val_data['adj_ebitda_multiple_range']}</div><div class="m-sub">India-adjusted</div></div>
  <div class="metric-card"><div class="m-label">MSME Class</div><div class="m-value" style="font-size:14px;color:#60a5fa">{msme['tier'].upper()}</div><div class="m-sub">{msme['label'][:25]}</div></div>
</div>
""", unsafe_allow_html=True)

    # ════════════════ PSYCHOLOGY TAB ══════════════════════════════════════════
    with r2:
        col_radar, col_scores = st.columns([1, 1])

        with col_radar:
            st.markdown('<div class="panel-header">FOUNDER MOMENTUM RADAR</div>', unsafe_allow_html=True)
            st.plotly_chart(chart_momentum_radar(psych_data["radar_data"]), use_container_width=True)

        with col_scores:
            st.markdown('<div class="panel-header">SIGNAL BREAKDOWN</div>', unsafe_allow_html=True)
            st.markdown(
                _score_bar("BURNOUT", psych_data["burnout_score"], "#ef4444") +
                _score_bar("CONVICTION", psych_data["conviction_score"], "#10b981") +
                _score_bar("OPPORTUNITY EXIT", psych_data["opportunity_exit_score"], "#f59e0b") +
                _score_bar("FAMILY EXIT", psych_data["family_exit_score"], "#a78bfa") +
                _score_bar("FINANCIAL DISTRESS", psych_data["financial_distress_score"], "#6b7280"),
                unsafe_allow_html=True
            )
            profile = psych_data["motivation_profile"]
            m_color = profile.get("color", "#6b7280")
            st.markdown(f"""
<div style="margin-top:12px;padding:10px 14px;background:#0d1120;border:1px solid #1e2d4a;border-left:3px solid {m_color};border-radius:6px;">
  <div style="font-family:JetBrains Mono;font-size:10px;color:#475569;letter-spacing:2px;margin-bottom:4px;">MOTIVATION TYPE</div>
  <div style="font-size:14px;font-weight:600;color:{m_color};">{psych_data['motivation_type']}</div>
  <div style="font-size:11px;color:#94a3b8;margin-top:4px;">{profile.get('description','')}</div>
  <div style="font-size:11px;color:#475569;margin-top:6px;">Urgency: {profile.get('urgency','?').title()} · Price Flex: {profile.get('price_flexibility','?').upper()}</div>
</div>
""", unsafe_allow_html=True)

        # Boredom/fast-close badges
        badges = ""
        if psych_data.get("boredom_multiple_likely"):
            badges += _badge("YELLOW", "BOREDOM MULTIPLE — 2x ARR LIKELY")
        if psych_data.get("fast_close_signal"):
            badges += _badge("GREEN", "FAST CLOSE SIGNAL")
        if psych_data.get("family_signals"):
            badges += _badge("PURPLE", "FAMILY DYNAMICS DETECTED")
        if badges:
            st.markdown(f'<div style="margin:12px 0;">{badges}</div>', unsafe_allow_html=True)

        # Psychologist narrative
        st.markdown('<div class="panel-header">PSYCHOGRAPHIC PROFILE</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="ic-panel" style="font-size:13px;line-height:1.7;color:#cbd5e1;">'
            f'{result["psych_narrative"].replace(chr(10), "<br>")}</div>',
            unsafe_allow_html=True
        )

        # Key signals
        burn_sigs = psych_data.get("burnout_signals", [])
        fam_sigs = psych_data.get("family_signals", [])
        if burn_sigs or fam_sigs:
            with st.expander("🔍 Raw NLP Signals Detected"):
                if burn_sigs:
                    st.write("**Burnout Signals:**", ", ".join(f'`{s}`' for s in burn_sigs[:6]))
                if fam_sigs:
                    st.write("**Family Signals:**", ", ".join(f'`{s}`' for s in fam_sigs[:5]))
                if psych_data.get("transcript_extracts"):
                    te = psych_data["transcript_extracts"]
                    st.write("**Transcript Extracts:**")
                    if te.get("revenue_mentions"):
                        st.write(f"Revenue mentions: {', '.join(te['revenue_mentions'])}")
                    if te.get("timeline"):
                        st.write(f"Timeline hint: {te['timeline']}")
                    if te.get("motivation_hints"):
                        st.write(f"Motivation: {', '.join(te['motivation_hints'])}")

        # Momentum timeline
        timeline = psych_data.get("momentum_timeline", [])
        if timeline:
            st.markdown('<div class="panel-header" style="margin-top:16px;">FOUNDER MOMENTUM TIMELINE</div>', unsafe_allow_html=True)
            t_data = pd.DataFrame(timeline)
            if "score" in t_data.columns:
                fig_t = px.bar(
                    t_data, x="channel", y="score", color="direction",
                    color_discrete_map={"positive": "#10b981", "neutral": "#475569", "negative": "#ef4444"},
                    template="plotly_dark", text="score",
                )
                fig_t.update_layout(
                    paper_bgcolor="#0d1120", plot_bgcolor="#0d1120",
                    height=180, margin=dict(l=10, r=10, t=10, b=10),
                    showlegend=False,
                    font={"family": "JetBrains Mono", "size": 9, "color": "#94a3b8"},
                )
                st.plotly_chart(fig_t, use_container_width=True)

    # ════════════════ LEGAL/REG TAB ══════════════════════════════════════════
    with r3:
        col_score, col_state = st.columns([1, 1])

        with col_score:
            st.markdown('<div class="panel-header">COMPLIANCE OVERVIEW</div>', unsafe_allow_html=True)
            comp_score = reg_data["compliance_score"]
            comp_color = "#10b981" if comp_score >= 70 else "#f59e0b" if comp_score >= 45 else "#ef4444"
            st.markdown(f"""
<div class="ic-panel">
  <div style="text-align:center;">
    <div style="font-family:JetBrains Mono;font-size:36px;font-weight:700;color:{comp_color};">{comp_score}</div>
    <div style="font-family:JetBrains Mono;font-size:10px;color:#475569;letter-spacing:2px;">COMPLIANCE SCORE / 100</div>
    <div style="font-size:12px;color:{comp_color};margin-top:4px;">{reg_data['compliance_label']}</div>
  </div>
  <div style="margin-top:12px;">
    {_score_bar("INDIA RISK", reg_data['india_risk_score'], "#ef4444")}
  </div>
</div>
""", unsafe_allow_html=True)

            # Deal structure
            ds = reg_data["deal_structure_recommendation"]
            st.markdown(f"""
<div class="ic-panel" style="margin-top:12px;">
  <div class="panel-header">RECOMMENDED STRUCTURE</div>
  <div style="font-size:12px;color:#60a5fa;font-weight:600;margin-bottom:4px;">{ds['recommended_structure']}</div>
  <div style="font-size:11px;color:#94a3b8;">{ds['rationale']}</div>
  <div style="margin-top:8px;font-family:JetBrains Mono;font-size:10px;color:#475569;">Complexity: {ds['complexity']}</div>
</div>
""", unsafe_allow_html=True)

        with col_state:
            st.markdown('<div class="panel-header">STATE DEAL DENSITY</div>', unsafe_allow_html=True)
            st.plotly_chart(
                chart_state_density(result["deal_input"].get("state", "Maharashtra")),
                use_container_width=True
            )
            state_profile = reg_data.get("state_profile", {})
            if state_profile:
                cultural = state_profile.get("cultural_notes", "")
                if cultural:
                    st.markdown(
                        f'<div class="ic-panel" style="font-size:11px;color:#94a3b8;font-style:italic;">💡 {cultural}</div>',
                        unsafe_allow_html=True
                    )

        # Risk flags grid
        st.markdown('<div class="panel-header">RISK FLAGS</div>', unsafe_allow_html=True)
        flags = reg_data["flags"]
        flag_cols = st.columns(2)
        for i, flag in enumerate(flags):
            level = flag["level"]
            badge = RISK_LEVELS.get(level, RISK_LEVELS["YELLOW"])
            with flag_cols[i % 2]:
                st.markdown(f"""
<div style="background:#0d1120;border:1px solid {badge['color']}33;border-left:3px solid {badge['color']};
     border-radius:6px;padding:10px 14px;margin-bottom:8px;">
  <div style="display:flex;gap:6px;align-items:center;margin-bottom:4px;">
    <span style="font-family:JetBrains Mono;font-size:9px;font-weight:700;color:{badge['color']};
          background:{badge['bg']};padding:1px 6px;border-radius:3px;">{badge['icon']} {level}</span>
    <span style="font-size:11px;font-weight:600;color:#e2e8f0;">{flag['category']}</span>
  </div>
  <div style="font-size:11px;color:#94a3b8;">{flag['detail'][:120]}</div>
  <div style="font-size:10px;color:#475569;margin-top:4px;">→ {flag['action'][:90]}</div>
</div>
""", unsafe_allow_html=True)

        # Regulatory narrative
        st.markdown('<div class="panel-header">REGULATORY BRIEF</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="ic-panel" style="font-size:13px;line-height:1.7;color:#cbd5e1;">'
            f'{result["reg_narrative"].replace(chr(10), "<br>")}</div>',
            unsafe_allow_html=True
        )

    # ════════════════ STRATEGY TAB ══════════════════════════════════════════
    with r4:
        col_play, col_outreach = st.columns([1, 1])

        with col_play:
            st.markdown('<div class="panel-header">NEGOTIATION PLAYBOOK</div>', unsafe_allow_html=True)
            pb = playbook
            st.markdown(f"""
<div class="ic-panel">
  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin-bottom:12px;">
    <div>
      <div style="font-family:JetBrains Mono;font-size:9px;color:#475569;letter-spacing:2px;">OPEN AT</div>
      <div style="font-family:JetBrains Mono;font-size:18px;font-weight:700;color:#10b981;">₹{pb['opening_offer_cr']}Cr</div>
    </div>
    <div>
      <div style="font-family:JetBrains Mono;font-size:9px;color:#475569;letter-spacing:2px;">TARGET</div>
      <div style="font-family:JetBrains Mono;font-size:18px;font-weight:700;color:#f59e0b;">₹{pb['target_mid_cr']}Cr</div>
    </div>
    <div>
      <div style="font-family:JetBrains Mono;font-size:9px;color:#475569;letter-spacing:2px;">WALK AWAY</div>
      <div style="font-family:JetBrains Mono;font-size:18px;font-weight:700;color:#ef4444;">₹{pb['walk_away_cr']}Cr</div>
    </div>
  </div>
  <div style="font-size:11px;color:#94a3b8;font-style:italic;padding-top:8px;border-top:1px solid #1e2d4a;">{pb['strategy_tone']}</div>
</div>
""", unsafe_allow_html=True)

            if pb.get("first_call_agenda"):
                st.markdown('<div class="panel-header" style="margin-top:12px;">FIRST CALL AGENDA</div>', unsafe_allow_html=True)
                for step in pb["first_call_agenda"]:
                    st.markdown(f'<div style="font-size:12px;color:#94a3b8;padding:3px 0;">▸ {step}</div>', unsafe_allow_html=True)

            if pb.get("concessions_to_offer"):
                st.markdown('<div class="panel-header" style="margin-top:12px;">CONCESSIONS TO OFFER</div>', unsafe_allow_html=True)
                for c in pb["concessions_to_offer"]:
                    st.markdown(f'<div style="font-size:12px;color:#10b981;padding:2px 0;">✓ {c}</div>', unsafe_allow_html=True)

            if pb.get("walk_away_triggers"):
                st.markdown('<div class="panel-header" style="margin-top:12px;">WALK-AWAY TRIGGERS</div>', unsafe_allow_html=True)
                for w in pb["walk_away_triggers"]:
                    st.markdown(f'<div style="font-size:12px;color:#ef4444;padding:2px 0;">✗ {w}</div>', unsafe_allow_html=True)

        with col_outreach:
            st.markdown('<div class="panel-header">PERSONALIZED OUTREACH DRAFT</div>', unsafe_allow_html=True)
            st.text_area(
                "Outreach Message",
                result["outreach"],
                height=280,
                key="outreach_text",
                help="Edit and copy to WhatsApp or email"
            )
            st.download_button(
                "⬇ Download Outreach",
                result["outreach"],
                file_name=f"outreach_{result['title'][:20].replace(' ','_')}.txt",
                mime="text/plain",
            )

        # Strategy narrative
        st.markdown('<div class="panel-header" style="margin-top:16px;">STRATEGY NARRATIVE</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="ic-panel" style="font-size:13px;line-height:1.7;color:#cbd5e1;">'
            f'{result["strategy_summary"].replace(chr(10), "<br>")}</div>',
            unsafe_allow_html=True
        )

        # LOI
        if result.get("loi_text"):
            st.markdown('<div class="panel-header" style="margin-top:16px;">LETTER OF INTENT — DRAFT</div>', unsafe_allow_html=True)
            with st.expander(f"📝 LOI for {result['title'][:50]} (auto-generated, conviction ≥ 6.5)", expanded=False):
                st.text_area("LOI Document", result["loi_text"], height=500, key="loi_text")
                st.download_button(
                    "⬇ Download LOI",
                    result["loi_text"],
                    file_name=f"LOI_{result['title'][:20].replace(' ','_')}.txt",
                    mime="text/plain",
                    key="dl_loi",
                )

    # Agent log
    with st.expander("🖥 Agent Execution Log"):
        log_html = '<div class="agent-log">'
        for entry in st.session_state.agent_logs:
            css_class = f"log-{entry['agent']}"
            log_html += f'<div class="{css_class}">{entry["text"]}</div>'
        log_html += '</div>'
        st.markdown(log_html, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🏦 POCKET FUND")
    st.markdown("**Acquisition Intelligence Layer**")
    st.markdown("---")
    # Simple UI toggle for users who want a minimal workflow
    simple_mode = st.checkbox("Simple UI (quick hunt)", value=False, help="Toggle a compact, easy-to-use hunting interface")
    st.session_state.simple_mode = simple_mode

    st.markdown("### LLM Configuration")

    # Auto-detect keys from secrets
    _env_openai = os.environ.get("OPENAI_API_KEY", "")
    _env_groq   = os.environ.get("GROQ_API_KEY", "")

    # Default to openai if key found in secrets
    _default_provider_idx = 1 if _env_openai else (2 if _env_groq else 0)
    llm_provider = st.selectbox(
        "Provider",
        ["none (demo mode)", "openai", "groq"],
        index=_default_provider_idx,
    )
    provider_key = llm_provider.split(" ")[0]

    models_map = {
        "none": [],
        "openai": ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"],
        "groq": ["llama-3.3-70b-versatile", "llama-3.1-70b-versatile", "mixtral-8x7b-32768"],
    }
    model_options = models_map.get(provider_key, [])
    selected_model = st.selectbox("Model", model_options if model_options else ["N/A"]) if model_options else "N/A"

    # Pre-fill from env/secrets — show masked if present
    _prefill_key = _env_openai if provider_key == "openai" else (_env_groq if provider_key == "groq" else "")
    _placeholder  = f"{'*' * 8}...{_prefill_key[-4:]}" if _prefill_key else "Paste key here..."
    api_key_in = st.text_input("API Key", type="password", placeholder=_placeholder,
                                help="Pre-loaded from .streamlit/secrets.toml if present")

    # Resolve: explicit input > env var
    resolved_key = api_key_in.strip() if api_key_in.strip() else _prefill_key

    if resolved_key and provider_key != "none":
        if st.session_state.llm is None:  # only rebuild if not already connected
            llm_obj = _build_llm(provider_key, resolved_key, selected_model)
            if llm_obj:
                st.session_state.llm = llm_obj
        status = "connected" if st.session_state.llm else "failed"
        if status == "connected":
            st.success(f"✓ {provider_key.upper()} {selected_model}")
        else:
            st.error("Connection failed — check key")
            if api_key_in:  # only retry if user explicitly entered new key
                st.session_state.llm = _build_llm(provider_key, resolved_key, selected_model)
    elif provider_key == "none":
        st.session_state.llm = None

    # Reddit status
    _reddit_ok = bool(os.environ.get("REDDIT_CLIENT_ID"))
    st.markdown(
        f'<div style="font-family:JetBrains Mono;font-size:10px;margin-top:4px;">'
        f'Reddit: <span style="color:{"#10b981" if _reddit_ok else "#ef4444"}">'
        f'{"✓ OAuth" if _reddit_ok else "✗ Not configured"}</span></div>',
        unsafe_allow_html=True,
    )

    # Acquire.com status
    _acq_email = os.environ.get("ACQUIRE_EMAIL", "")
    st.markdown(
        f'<div style="font-family:JetBrains Mono;font-size:10px;margin-top:4px;">'
        f'Acquire.com: <span style="color:{"#10b981" if _acq_email else "#ef4444"}">'
        f'{"✓ "+_acq_email.split("@")[0] if _acq_email else "✗ No credentials"}</span></div>',
        unsafe_allow_html=True,
    )

    # Firecrawl status
    _fc_key = os.environ.get("FIRECRAWL_API_KEY", "")
    _fc_ok  = bool(_fc_key)
    st.markdown(
        f'<div style="font-family:JetBrains Mono;font-size:10px;margin-top:4px;">'
        f'Firecrawl: <span style="color:{"#10b981" if _fc_ok else "#ef4444"}">'
        f'{"✓ ..."+_fc_key[-4:] if _fc_ok else "✗ Not configured"}</span></div>',
        unsafe_allow_html=True,
    )

    # Manual Firecrawl key input (override)
    _fc_manual = st.text_input(
        "Firecrawl API Key (override)",
        type="password",
        placeholder="fc-..." if not _fc_ok else f"Using key ...{_fc_key[-4:]}",
        help="Auto-loaded from secrets.toml. Override here to use a different key.",
    )
    if _fc_manual.strip():
        os.environ["FIRECRAWL_API_KEY"] = _fc_manual.strip()

    st.markdown("---")
    st.markdown("### Memory Status")
    stats = MEMORY.get_stats()
    st.markdown(f"""
<div style="font-family:JetBrains Mono;font-size:10px;color:#475569;">
  Backend: <span style="color:#60a5fa">{stats['backend'].upper()}</span><br>
  Deals: <span style="color:#f59e0b">{stats['total']}</span> 
  (HOT={stats['hot']} WARM={stats['warm']} PASS={stats['passed']})<br>
  Runs this session: <span style="color:#10b981">{st.session_state.run_count}</span>
</div>
""", unsafe_allow_html=True)

    if st.button("🔄 Reseed Demo Data"):
        MEMORY.seed_demo_data(force=True)
        st.success("Demo data reseeded!")

    st.markdown("---")
    st.markdown("### Buyer Profile")
    buyer_name = st.text_input("Name", "Dev Shah")
    buyer_entity = st.text_input("Entity", "Pocket Fund")
    buyer_email = st.text_input("Email", "dev@pocketfund.in")

    st.markdown("---")
    st.markdown(
        '<div style="font-family:JetBrains Mono;font-size:9px;color:#374151;line-height:1.6;">'
        '⚠️ PROTOTYPE · Mock/simulated data<br>Not financial/legal advice.<br>'
        f'UTC {datetime.utcnow().strftime("%Y-%m-%d %H:%M")}</div>',
        unsafe_allow_html=True
    )

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────

kb_sectors = json.load(open(KB_PATH))["sectors"]

st.markdown("""
<div class="ic-header">
  <div class="fund-name">Pocket Fund</div>
  <div class="page-title">Acquisition Intelligence Layer</div>
  <div class="subtitle">
    <span class="live-dot"></span>
    India Micro-PE Agent Swarm · MSME/SME · Sub-₹80L Focus · 
    Valuation · Psychology · Regulatory · Strategy
  </div>
</div>
""", unsafe_allow_html=True)

# Stats row
mem_stats = MEMORY.get_stats()
st.markdown(f"""
<div class="metric-grid">
  <div class="metric-card">
    <div class="m-label">Deals in Memory</div>
    <div class="m-value">{mem_stats['total']}</div>
    <div class="m-sub">FAISS/TF-IDF indexed</div>
  </div>
  <div class="metric-card">
    <div class="m-label">🔴 Hot</div>
    <div class="m-value" style="color:#ef4444">{mem_stats['hot']}</div>
  </div>
  <div class="metric-card">
    <div class="m-label">🟡 Warm</div>
    <div class="m-value" style="color:#f59e0b">{mem_stats['warm']}</div>
  </div>
  <div class="metric-card">
    <div class="m-label">Agents Active</div>
    <div class="m-value" style="color:#60a5fa">4</div>
    <div class="m-sub">Analyst·Psych·Legal·Strategy</div>
  </div>
  <div class="metric-card">
    <div class="m-label">LLM</div>
    <div class="m-value" style="font-size:14px;color:#10b981">
      {'LIVE' if st.session_state.llm else 'DEMO'}
    </div>
    <div class="m-sub">{'GPT/Groq connected' if st.session_state.llm else 'Heuristic mode'}</div>
  </div>
  <div class="metric-card">
    <div class="m-label">Firecrawl</div>
    <div class="m-value" style="font-size:14px;color:{'#10b981' if os.environ.get('FIRECRAWL_API_KEY') else '#ef4444'}">
      {'ON' if os.environ.get('FIRECRAWL_API_KEY') else 'OFF'}
    </div>
    <div class="m-sub">{'URL scraping active' if os.environ.get('FIRECRAWL_API_KEY') else 'Add key in sidebar'}</div>
  </div>
</div>
""", unsafe_allow_html=True)

# If simple mode enabled, render a compact quick-hunt UI and stop further complex UI
if st.session_state.get("simple_mode"):
    st.markdown(
        '<div style="font-family:JetBrains Mono;font-size:18px;font-weight:700;margin-bottom:8px">Quick Hunt — Simple Mode</div>',
        unsafe_allow_html=True,
    )
    col1, col2 = st.columns([3, 1])
    with col1:
        s_max_price = st.number_input("Max Ask ($)", value=300000, step=5000, help="Cap on asking price")
        s_min_score = st.slider("Min Score", 0, 100, 25)
        s_india_only = st.checkbox("India only", value=False)
    with col2:
        quick_btn = st.button("🎯 Quick Hunt", use_container_width=True, type="primary")

    if quick_btn:
        with st.spinner("Running quick hunt across main sources..."):
            deals = hunt_live_deals(
                use_acquire_auth=bool(os.environ.get("ACQUIRE_EMAIL")),
                use_ef=False,
                use_flippa=True,
                use_reddit=True,
                use_acquire_rss=True,
                use_web_search=bool(os.environ.get("FIRECRAWL_API_KEY")),
                use_side_projectors=True,
                max_total=80,
            )

        # Filter simple results
        filtered = [d for d in deals if d.get("score", 0) >= s_min_score]
        if s_india_only:
            filtered = [
                d
                for d in filtered
                if d.get("country") == "India"
                or re.search(r"\bindia\b", (d.get("description", "") + d.get("title", "")), re.IGNORECASE)
            ]
        if s_max_price > 0:
            filtered = [d for d in filtered if d.get("asking_price_usd") is None or d.get("asking_price_usd", 0) <= s_max_price]

        st.markdown(f"**{len(filtered)} deals** — Showing top 40")
        for d in filtered[:40]:
            ask = f"${d['asking_price_usd']:,.0f}" if d.get("asking_price_usd") else "n/a"
            score = d.get("score", 0)
            st.markdown(f'- **{score:.0f}/100** • {d.get("source","?")} • {ask} — [{d["title"]}]({d["url"]})', unsafe_allow_html=True)
    else:
        st.markdown('<div style="color:#94a3b8">Press Quick Hunt to collect deals (simple mode)</div>', unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Input Tabs
# ─────────────────────────────────────────────────────────────────────────────

t_deal, t_transcript, t_research, t_memory, t_hunt = st.tabs([
    "📋 DEAL ANALYSIS",
    "🎙 VOICE → INSIGHT",
    "🔍 RESEARCH MODE",
    "🧠 DEAL MEMORY",
    "🎯 LIVE DEAL HUNT",
])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Deal Analysis
# ════════════════════════════════════════════════════════════════════════════
with t_deal:
    st.markdown('<div class="panel-header">DEAL INPUT — LISTING URL OR DESCRIPTION</div>', unsafe_allow_html=True)

    # ── URL scraping ──────────────────────────────────────────────────────────
    _fc_available = bool(os.environ.get("FIRECRAWL_API_KEY"))
    url_col, btn_col = st.columns([5, 1])
    with url_col:
        listing_url = st.text_input(
            "Paste Listing URL (Acquire.com, Flippa, Reddit post, any business listing)",
            placeholder="https://acquire.com/listings/... or https://flippa.com/...",
            help="Firecrawl will scrape and auto-fill all deal fields below",
        )
    with btn_col:
        st.markdown("<br>", unsafe_allow_html=True)
        scrape_btn = st.button(
            "🕷 Scrape",
            disabled=not _fc_available,
            help="Requires Firecrawl API key" if not _fc_available else "Extract deal data from URL",
        )

    if scrape_btn and listing_url.strip():
        with st.spinner(f"Firecrawl scraping {listing_url[:60]}..."):
            scraped = scrape_listing_url(listing_url.strip())
        if scraped.get("title"):
            st.session_state["scraped_deal"] = scraped
            badge_color = "#10b981" if scraped.get("scrape_source", "").startswith("firecrawl") else "#f59e0b"
            st.markdown(
                f'<div style="background:#022c22;border:1px solid #065f46;border-radius:6px;'
                f'padding:10px 14px;margin-bottom:12px;font-size:12px;color:#10b981;">'
                f'✓ Scraped via <b style="color:{badge_color}">{scraped.get("scrape_source","?")}</b> · '
                f'<b>{scraped["title"][:60]}</b> · '
                f'Ask: {scraped.get("asking_price_usd") and "$"+str(int(scraped["asking_price_usd"])) or "N/A"} · '
                f'Rev: ₹{scraped.get("revenue_cr",0):.2f}Cr'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.warning("Could not extract structured data. Check the URL or paste description manually.")

    elif not _fc_available:
        st.markdown(
            '<div style="font-size:11px;color:#475569;padding:4px 0;">'
            '💡 Add your Firecrawl key in the sidebar to enable URL scraping</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Pre-fill from scraped data or demo
    _scraped = st.session_state.get("scraped_deal", {})

    # Pre-fill with demo deal
    col_pre, col_clear = st.columns([2, 2])
    with col_pre:
        if st.button("⚡ Load Demo Deal (BillZap SaaS)"):
            st.session_state["demo_prefill"] = True
            st.session_state["scraped_deal"] = {}
    with col_clear:
        if _scraped and st.button("✕ Clear scraped data"):
            st.session_state["scraped_deal"] = {}
            st.rerun()

    demo = st.session_state.get("demo_prefill", False)

    col_l, col_r = st.columns([3, 2])

    with col_l:
        deal_title = st.text_input(
            "Business / Listing Title",
            value=_scraped.get("title") or ("BillZap — GST Invoice SaaS for Indian SMBs" if demo else ""),
            placeholder="e.g. Pune CNC Machining Shop — Family Exit",
        )
        deal_desc = st.text_area(
            "Listing Text / Description",
            value=_scraped.get("description") or (
                "Built this 3 years ago as a solo project. $4,200 MRR (~₹3.5L), ~340 SMB customers, "
                "87% margins, zero churn. I got a really exciting offer to join a Series A fintech as "
                "Head of Product and I genuinely cannot do both. The product is feature-complete, runs "
                "on autopilot mostly. I am the only developer. GST compliant, PVT LTD entity. "
                "Asking ₹72.5L (~$87k). Prefer quick close. No earnout."
            ) if demo else "",
            height=140,
            placeholder="Paste the listing text, notes from a call, or any deal context...",
        )

        # Founder profile URL scraping
        founder_url_col, founder_scrape_col = st.columns([4, 1])
        with founder_url_col:
            founder_profile_url = st.text_input(
                "Founder Profile URL (optional)",
                placeholder="https://twitter.com/founder or LinkedIn URL",
                help="Firecrawl will extract bio, recent posts, and exit signals",
            )
        with founder_scrape_col:
            st.markdown("<br>", unsafe_allow_html=True)
            founder_scrape_btn = st.button(
                "🕷",
                disabled=not _fc_available,
                key="founder_scrape",
                help="Scrape founder profile for momentum signals",
            )

        if founder_scrape_btn and founder_profile_url.strip():
            with st.spinner("Scraping founder profile..."):
                fp = scrape_founder_profile(founder_profile_url.strip())
            if fp.get("bio"):
                st.session_state["founder_profile"] = fp
                st.success(f"Founder profile scraped: {fp.get('name', 'unknown')}")

        _fp = st.session_state.get("founder_profile", {})
        founder_twitter  = st.text_input(
            "Founder Bio / X Signal",
            value=_fp.get("bio") or (_scraped.get("seller_handle") or ("@aakash_builds — Building BillZap | IIT Roorkee '19" if demo else "")),
        )
        founder_linkedin = st.text_input(
            "LinkedIn Activity",
            value=_fp.get("current_role") or ("Updated job title to 'Head of Product @ fintech' 3 weeks ago" if demo else ""),
        )
        founder_reddit   = st.text_input(
            "Exit / Reddit Signal",
            value=_fp.get("exit_signals") or ("Posted r/IndieHackers: 'Thinking of selling my SaaS'" if demo else ""),
        )

    with col_r:
        sector_labels = {k: v["label"] for k, v in kb_sectors.items()}
        # Auto-detect sector from scraped hint
        _scraped_sector = _scraped.get("sector_hint", "").lower()
        _default_sector_idx = 0
        for i, (k, v) in enumerate(sector_labels.items()):
            if k == "saas" and (demo or "saas" in _scraped_sector):
                _default_sector_idx = i
                break
        sector_key = st.selectbox(
            "Sector",
            list(sector_labels.keys()),
            format_func=lambda k: sector_labels[k],
            index=_default_sector_idx,
        )
        state_options = list(json.load(open(KB_PATH))["india_state_profiles"].keys())
        _scraped_state = _scraped.get("india_state") or _scraped.get("location", "")
        _default_state = next((s for s in state_options if s in _scraped_state), "Maharashtra")
        state = st.selectbox(
            "State (India)",
            state_options,
            index=state_options.index(_default_state) if _default_state in state_options else 0,
        )

        col_rev, col_ebitda = st.columns(2)
        with col_rev:
            revenue_cr = st.number_input(
                "Revenue (₹ Crore)",
                min_value=0.0, max_value=500.0,
                value=float(_scraped.get("revenue_cr") or (0.32 if demo else 1.0)),
                step=0.1, format="%.2f",
            )
        with col_ebitda:
            ebitda_l = st.number_input(
                "EBITDA (₹ Lakh)",
                min_value=0.0, max_value=5000.0,
                value=float(_scraped.get("ebitda_l") or (22.0 if demo else 10.0)),
                step=1.0,
            )

        business_type = st.selectbox(
            "Business Entity Type",
            ["Private Limited", "Sole Proprietorship / Individual", "Partnership", "LLP", "Unknown"],
            index=0,
        )
        seller_name = st.text_input("Seller Name", value=_scraped.get("seller_handle") or ("Aakash Mehta" if demo else ""))

        col_c1, col_c2, col_c3 = st.columns(3)
        with col_c1:
            is_family = st.checkbox("Family-Run", value=bool(_scraped.get("india_family_run")))
        with col_c2:
            gst_reg = st.checkbox("GST Registered", value=_scraped.get("india_gst_registered") if _scraped.get("india_gst_registered") is not None else True)
        with col_c3:
            udyam = st.checkbox("Udyam Reg.", value=False)
        digital_ready = st.checkbox("Digital-Ready / Cloud Ops", value=True if demo else False)

        _default_risks = "; ".join(_scraped.get("risks", [])[:2]) if _scraped.get("risks") else ("Razorpay PG requires new entity setup" if demo else "")
        special_risk_text = st.text_input(
            "Special Risk Note (optional)",
            value=_default_risks,
            placeholder="e.g. WhatsApp API not transferable",
        )

    if st.button("⚡  RUN AGENT SWARM", use_container_width=False, key="run_deal"):
        if not deal_title.strip():
            st.error("Please enter a business title.")
        else:
            deal_input = {
                "title": deal_title,
                "description": deal_desc,
                "sector": sector_key,
                "state": state,
                "revenue_cr": revenue_cr,
                "ebitda_l": ebitda_l,
                "is_family_run": is_family,
                "gst_registered": gst_reg,
                "udyam_registered": udyam,
                "digital_ready": digital_ready,
                "business_type": business_type,
                "seller_name": seller_name,
                "special_risks": [special_risk_text] if special_risk_text.strip() else [],
                "founder_signals": {
                    "twitter_bio": founder_twitter,
                    "linkedin_activity": founder_linkedin,
                    "reddit_post": founder_reddit,
                },
                "transcript": "",
            }
            with st.spinner("Agent swarm running..."):
                result = run_analysis_pipeline(deal_input, llm=st.session_state.llm)
            st.session_state.analysis_result = result
            st.session_state.demo_prefill = False

    if st.session_state.analysis_result:
        st.markdown("---")
        render_results(st.session_state.analysis_result)

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Voice → Insight
# ════════════════════════════════════════════════════════════════════════════
with t_transcript:
    st.markdown('<div class="panel-header">VOICE CALL TRANSCRIPT → AGENT SWARM</div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:12px;color:#475569;margin-bottom:16px;">'
        'Paste a raw call transcript below. The Psychologist Agent will extract sector, motivation, '
        'family angle, timeline, and revenue signals automatically. Then run the full swarm.</div>',
        unsafe_allow_html=True
    )

    kb_transcript = json.load(open(KB_PATH)).get("sample_voice_transcript", "")

    col_load, _ = st.columns([2, 5])
    with col_load:
        load_sample = st.button("⚡ Load Sample Transcript")

    transcript_text = st.text_area(
        "Paste Call Transcript",
        value=kb_transcript if load_sample else "",
        height=300,
        placeholder="Paste raw call transcript here — no formatting needed..."
    )

    # Quick context fields
    col_ta, col_tb = st.columns(2)
    with col_ta:
        t_title = st.text_input("Deal Label (optional)", placeholder="e.g. Pune Precision Engineering — Call 1")
        t_state = st.selectbox("State (if known)", ["Maharashtra", "Gujarat", "Karnataka", "Tamil Nadu", "Delhi NCR", "Rajasthan", "Telangana", "Unknown"], key="t_state")
    with col_tb:
        t_sector = st.selectbox("Sector (if known)", list(sector_labels.keys()), format_func=lambda k: sector_labels[k], key="t_sector")
        t_revenue = st.number_input("Revenue ₹ Crore (if known)", 0.0, 500.0, 0.0, step=0.5)
        t_ebitda = st.number_input("EBITDA ₹ Lakh (if known)", 0.0, 5000.0, 0.0, step=5.0)

    if st.button("⚡  EXTRACT + ANALYSE TRANSCRIPT", key="run_transcript"):
        if not transcript_text.strip():
            st.error("Please paste a transcript.")
        else:
            # Pre-analysis: extract facts
            psych_pre = FounderPsychologist()
            extracts = psych_pre._extract_transcript_facts(transcript_text)

            # Build deal_input from extracts + user overrides
            deal_input = {
                "title": t_title or f"Transcript Deal — {t_state}",
                "description": transcript_text[:500],
                "sector": t_sector if t_sector else extracts.get("detected_sector", "manufacturing_general"),
                "state": t_state,
                "revenue_cr": t_revenue if t_revenue > 0 else 4.0,
                "ebitda_l": t_ebitda if t_ebitda > 0 else 56.0,
                "is_family_run": extracts.get("has_family_angle", False),
                "gst_registered": True,
                "udyam_registered": False,
                "digital_ready": False,
                "business_type": "Private Limited",
                "seller_name": "Founder",
                "special_risks": [],
                "founder_signals": {},
                "transcript": transcript_text,
            }

            # Show extraction preview
            st.markdown('<div class="panel-header">AUTO-EXTRACTED SIGNALS</div>', unsafe_allow_html=True)
            ex_cols = st.columns(4)
            with ex_cols[0]:
                st.metric("Sector Detected", sector_labels.get(extracts.get("detected_sector", "?"), extracts.get("detected_sector", "?")))
            with ex_cols[1]:
                st.metric("Revenue Mentions", ", ".join(extracts.get("revenue_mentions", ["-"]))[:20])
            with ex_cols[2]:
                st.metric("Timeline Hint", extracts.get("timeline", "N/A"))
            with ex_cols[3]:
                st.metric("Family Angle", "YES ⚠️" if extracts.get("has_family_angle") else "No")

            with st.spinner("Agent swarm processing transcript..."):
                result = run_analysis_pipeline(deal_input, llm=st.session_state.llm)
            st.session_state.analysis_result = result

            st.markdown("---")
            render_results(result)

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — Research Mode
# ════════════════════════════════════════════════════════════════════════════
with t_research:
    st.markdown('<div class="panel-header">KEYWORD RESEARCH MODE</div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:12px;color:#475569;margin-bottom:16px;">'
        'Enter a search query like "Pune CNC shop" or "Bengaluru B2B SaaS exit" and the swarm '
        'generates a sector deep-dive with comps, India risk, and outreach strategy.</div>',
        unsafe_allow_html=True
    )

    col_q, col_params = st.columns([2, 1])
    with col_q:
        research_query = st.text_input(
            "Research Query",
            placeholder="e.g. Pune-based precision CNC shop family exit, Bengaluru SaaS newsletter selling",
        )
    with col_params:
        r_state = st.selectbox("Target State", state_options, key="r_state")
        r_sector = st.selectbox("Sector Focus", list(sector_labels.keys()), format_func=lambda k: sector_labels[k], key="r_sector")

    # Optional: crawl a URL to enrich research
    r_url_col, r_url_btn = st.columns([4, 1])
    with r_url_col:
        research_extra_url = st.text_input(
            "Optional: crawl a URL for additional context",
            placeholder="https://... (news article, company site, sector report)",
            key="research_url_input",
        )
    with r_url_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        r_crawl_btn = st.button("🕷 Crawl", disabled=not _fc_available, key="r_crawl")

    if r_crawl_btn and research_extra_url.strip():
        with st.spinner(f"Crawling {research_extra_url[:50]}..."):
            crawled_md = research_url(research_extra_url.strip())
        if crawled_md:
            st.session_state["research_crawl_context"] = crawled_md
            st.success(f"Crawled {len(crawled_md)} chars of context")
            with st.expander("Crawled content preview"):
                st.text(crawled_md[:800])

    if st.button("🔍  DEEP RESEARCH", key="run_research"):
        if not research_query.strip():
            st.error("Enter a search query.")
        else:
            # Build simulated deal from query + sector defaults
            sector_data = kb_sectors.get(r_sector, {})
            sim_revenue = (sector_data.get("typical_deal_size_cr", [1, 5])[0] + sector_data.get("typical_deal_size_cr", [1, 5])[-1]) / 2
            sim_margin = sector_data.get("typical_ebitda_margin_pct", 12) / 100
            sim_ebitda_l = round(sim_revenue * sim_margin * 100, 0)

            _crawl_ctx = st.session_state.get("research_crawl_context", "")
            deal_input = {
                "title": f"Research: {research_query[:60]}",
                "description": (
                    f"Research query: '{research_query}'. Simulated deal profile for {r_state} "
                    f"{sector_data.get('label', r_sector)} sector. Typical deal size: "
                    f"₹{sim_revenue:.1f}Cr revenue, {sector_data.get('typical_ebitda_margin_pct', 12)}% EBITDA margin. "
                    f"Key risks: {', '.join(sector_data.get('key_risks', [])[:3])}. "
                    f"India focus: motivated seller, sub-₹80L range."
                    + (f"\n\nCRAWLED CONTEXT:\n{_crawl_ctx[:1500]}" if _crawl_ctx else "")
                ),
                "sector": r_sector,
                "state": r_state,
                "revenue_cr": max(0.5, sim_revenue * 0.5),
                "ebitda_l": max(5, sim_ebitda_l * 0.5),
                "is_family_run": "family" in research_query.lower() or "legacy" in research_query.lower(),
                "gst_registered": True,
                "udyam_registered": False,
                "digital_ready": r_sector in ("saas", "automation_ai_tools", "newsletter_media"),
                "business_type": "Private Limited",
                "seller_name": "Founder",
                "special_risks": [],
                "founder_signals": {},
                "transcript": "",
            }

            with st.spinner(f"Deep-diving: {research_query}..."):
                result = run_analysis_pipeline(deal_input, llm=st.session_state.llm)
            st.session_state.analysis_result = result

            # Sector profile
            st.markdown('<div class="panel-header">SECTOR INTELLIGENCE</div>', unsafe_allow_html=True)
            s_cols = st.columns(4)
            with s_cols[0]:
                st.metric("EBITDA Multiple", f"{sector_data['ebitda_multiple_low']}x – {sector_data['ebitda_multiple_high']}x")
            with s_cols[1]:
                st.metric("Revenue Multiple", f"{sector_data['revenue_multiple_low']}x – {sector_data['revenue_multiple_high']}x")
            with s_cols[2]:
                st.metric("Typical Margin", f"{sector_data.get('typical_ebitda_margin_pct', '?')}%")
            with s_cols[3]:
                st.metric("Family Discount", f"-{int(sector_data.get('family_run_discount',0)*100)}%")
            st.markdown(
                f'<div class="ic-panel" style="font-size:12px;color:#94a3b8;">'
                f'<b>Notes:</b> {sector_data.get("notes", "")}<br>'
                f'<b>Key Risks:</b> {", ".join(sector_data.get("key_risks", []))}'
                f'</div>',
                unsafe_allow_html=True
            )

            st.markdown("---")
            render_results(result)

# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — Deal Memory
# ════════════════════════════════════════════════════════════════════════════
with t_memory:
    st.markdown('<div class="panel-header">DEAL MEMORY — VECTOR STORE</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div style="font-size:12px;color:#475569;margin-bottom:16px;">'
        f'Backend: <b style="color:#60a5fa">{MEMORY.get_stats()["backend"].upper()}</b> · '
        f'{MEMORY.get_stats()["total"]} deals indexed. Every analysis adds to this memory, '
        f'enabling smarter comparisons in future runs.</div>',
        unsafe_allow_html=True
    )

    all_deals = MEMORY.get_all_deals()

    if not all_deals:
        st.info("No deals in memory yet. Run an analysis to populate the store.")
    else:
        # Summary metrics
        mem_s = MEMORY.get_stats()
        st.markdown(f"""
<div class="metric-grid">
  <div class="metric-card"><div class="m-label">Total Indexed</div><div class="m-value">{mem_s['total']}</div></div>
  <div class="metric-card"><div class="m-label">🔴 HOT</div><div class="m-value" style="color:#ef4444">{mem_s['hot']}</div></div>
  <div class="metric-card"><div class="m-label">🟡 WARM</div><div class="m-value" style="color:#f59e0b">{mem_s['warm']}</div></div>
  <div class="metric-card"><div class="m-label">⚫ PASSED</div><div class="m-value" style="color:#6b7280">{mem_s['passed']}</div></div>
</div>
""", unsafe_allow_html=True)

        # Table view
        df_rows = []
        for d in all_deals:
            df_rows.append({
                "Title": (d.get("title") or "")[:55],
                "Sector": d.get("sector", "?"),
                "State": d.get("state", "?"),
                "Rev (₹Cr)": d.get("revenue_cr", 0),
                "EBITDA (₹L)": d.get("ebitda_l", 0),
                "Verdict": d.get("verdict", "?"),
                "Conviction": d.get("conviction_score", 0),
                "Burnout": d.get("burnout_score", 0),
                "India Risk": d.get("india_risk", 0),
                "Motivation": d.get("motivation", "?"),
                "Outcome": d.get("deal_outcome", "N/A"),
                "Added": (d.get("added_at") or "")[:10],
            })

        df = pd.DataFrame(df_rows)

        st.dataframe(
            df.style.apply(
                lambda row: [
                    "background-color:#2d0000;color:#ef4444" if row["Verdict"] == "HOT"
                    else "background-color:#2d1f00;color:#f59e0b" if row["Verdict"] == "WARM"
                    else "background-color:#111827;color:#6b7280"
                    if col == "Verdict" else ""
                    for col in df.columns
                ],
                axis=1
            ).format({"Rev (₹Cr)": "{:.2f}", "EBITDA (₹L)": "{:.0f}", "Conviction": "{:.1f}", "Burnout": "{:.1f}", "India Risk": "{:.1f}"}),
            use_container_width=True,
            height=360,
        )

        # Memory similarity test
        st.markdown('<div class="panel-header" style="margin-top:20px;">MEMORY SIMILARITY SEARCH</div>', unsafe_allow_html=True)
        test_query = st.text_input(
            "Test a similarity query",
            placeholder="e.g. Maharashtra family manufacturing burnout exit"
        )
        if test_query:
            similar = MEMORY.find_similar(test_query, k=3)
            if similar:
                for match in similar:
                    st.markdown(_memory_card(match.get("memory_note", ""), match.get("full_deal", {})), unsafe_allow_html=True)
            else:
                st.info("No similar deals found. Try a different query.")

# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — Live Deal Hunt
# ════════════════════════════════════════════════════════════════════════════

with t_hunt:
    from engine.live_deals import hunt_live_deals, score_deal

    st.markdown('<div class="panel-header">LIVE DEAL HUNT — REAL-TIME MULTI-SOURCE ACQUISITION FEED</div>', unsafe_allow_html=True)

    # Left-rail filters + saved searches + source toggles (persistent in session_state)
    _acq_creds = bool(os.environ.get("ACQUIRE_EMAIL"))
    _fc_ok = bool(os.environ.get("FIRECRAWL_API_KEY"))

    # Initialize session defaults for hunt filters
    st.session_state.setdefault("hunt_use_acq_auth", _acq_creds)
    st.session_state.setdefault("hunt_use_side_proj", True)
    st.session_state.setdefault("hunt_use_flippa", _fc_ok)
    st.session_state.setdefault("hunt_use_reddit", True)
    st.session_state.setdefault("hunt_use_ef", _fc_ok)
    st.session_state.setdefault("hunt_use_acq_rss", False)
    st.session_state.setdefault("hunt_use_websrch", _fc_ok)
    st.session_state.setdefault("hunt_max_price", 500000)
    st.session_state.setdefault("hunt_min_score", 20)
    st.session_state.setdefault("hunt_india_only", False)

    left_col, main_col = st.columns([1.6, 4])

    # Left rail: filters + saved searches
    with left_col:
        st.markdown("**Sources**")
        use_acq_auth = st.checkbox("Acquire.com (auth)", value=st.session_state["hunt_use_acq_auth"], help="Authenticated login — real listings", disabled=not _acq_creds, key="hunt_use_acq_auth")
        use_side_proj = st.checkbox("SideProjectors", value=st.session_state["hunt_use_side_proj"], key="hunt_use_side_proj")
        use_flippa = st.checkbox("Flippa (8 pages)", value=st.session_state["hunt_use_flippa"], key="hunt_use_flippa")
        use_reddit = st.checkbox("Reddit (11 subs)", value=st.session_state["hunt_use_reddit"], key="hunt_use_reddit")
        use_ef = st.checkbox("Empire Flippers", value=st.session_state["hunt_use_ef"], key="hunt_use_ef")
        use_acq_rss = st.checkbox("Acquire RSS", value=st.session_state["hunt_use_acq_rss"], key="hunt_use_acq_rss")
        use_websrch = st.checkbox("Web Search", value=st.session_state["hunt_use_websrch"], key="hunt_use_websrch")

        st.markdown("---")
        st.markdown("**Filters**")
        max_price = st.number_input("Max Ask ($)", min_value=0, max_value=5_000_000, value=st.session_state["hunt_max_price"], step=10000, key="hunt_max_price")
        min_score = st.slider("Min Score", 0, 100, st.session_state["hunt_min_score"], key="hunt_min_score")
        india_only = st.checkbox("India only", value=st.session_state["hunt_india_only"], key="hunt_india_only")

        st.markdown("---")
        st.markdown("**Saved Searches**")
        st.session_state.setdefault("saved_searches", {})
        saved_names = ["<none>"] + list(st.session_state["saved_searches"].keys())
        selected_saved = st.selectbox("Apply saved search", saved_names, index=0)
        col_s1, col_s2 = st.columns([2, 1])
        with col_s1:
            save_name = st.text_input("Save current as...", placeholder="e.g. India SaaS <$200k")
        with col_s2:
            if st.button("Save"):
                if save_name:
                    st.session_state["saved_searches"][save_name] = {
                        "use_acq_auth": st.session_state["hunt_use_acq_auth"],
                        "use_side_proj": st.session_state["hunt_use_side_proj"],
                        "use_flippa": st.session_state["hunt_use_flippa"],
                        "use_reddit": st.session_state["hunt_use_reddit"],
                        "use_ef": st.session_state["hunt_use_ef"],
                        "use_acq_rss": st.session_state["hunt_use_acq_rss"],
                        "use_websrch": st.session_state["hunt_use_websrch"],
                        "max_price": st.session_state["hunt_max_price"],
                        "min_score": st.session_state["hunt_min_score"],
                        "india_only": st.session_state["hunt_india_only"],
                    }
                    st.success("Saved")

        # Apply via button or via query param (apply_saved)
        # st.query_params is a dict-like; values are single strings in modern Streamlit
        _apply_saved_param = st.query_params.get("apply_saved", None)
        if _apply_saved_param and st.session_state["saved_searches"]:
            name_to_apply = _apply_saved_param if isinstance(_apply_saved_param, str) else _apply_saved_param[0]
            if name_to_apply in st.session_state["saved_searches"]:
                cfg = st.session_state["saved_searches"][name_to_apply]
                st.session_state["hunt_use_acq_auth"] = cfg["use_acq_auth"]
                st.session_state["hunt_use_side_proj"] = cfg["use_side_proj"]
                st.session_state["hunt_use_flippa"] = cfg["use_flippa"]
                st.session_state["hunt_use_reddit"] = cfg["use_reddit"]
                st.session_state["hunt_use_ef"] = cfg["use_ef"]
                st.session_state["hunt_use_acq_rss"] = cfg["use_acq_rss"]
                st.session_state["hunt_use_websrch"] = cfg["use_websrch"]
                st.session_state["hunt_max_price"] = cfg["max_price"]
                st.session_state["hunt_min_score"] = cfg["min_score"]
                st.session_state["hunt_india_only"] = cfg["india_only"]
                # clear param and rerun
                st.query_params.clear()
                st.rerun()

        if selected_saved and selected_saved != "<none>":
            if st.button("Apply saved"):
                cfg = st.session_state["saved_searches"][selected_saved]
                # apply into session state and rerun
                st.session_state["hunt_use_acq_auth"] = cfg["use_acq_auth"]
                st.session_state["hunt_use_side_proj"] = cfg["use_side_proj"]
                st.session_state["hunt_use_flippa"] = cfg["use_flippa"]
                st.session_state["hunt_use_reddit"] = cfg["use_reddit"]
                st.session_state["hunt_use_ef"] = cfg["use_ef"]
                st.session_state["hunt_use_acq_rss"] = cfg["use_acq_rss"]
                st.session_state["hunt_use_websrch"] = cfg["use_websrch"]
                st.session_state["hunt_max_price"] = cfg["max_price"]
                st.session_state["hunt_min_score"] = cfg["min_score"]
                st.session_state["hunt_india_only"] = cfg["india_only"]
                st.rerun()

        # List saved searches with rename/delete controls
        if st.session_state["saved_searches"]:
            st.markdown("**Manage saved searches**")
            for i, name in enumerate(list(st.session_state["saved_searches"].keys())):
                c1, c2, c3 = st.columns([6, 2, 2])
                with c1:
                    st.write(f"{i+1}. {name}")
                with c2:
                    if st.button("Rename", key=f"rename_{name}"):
                        st.session_state["rename_target"] = name
                with c3:
                    if st.button("Delete", key=f"delete_{name}"):
                        st.session_state["saved_searches"].pop(name, None)
                        st.rerun()

            # Rename UI
            if st.session_state.get("rename_target"):
                target = st.session_state["rename_target"]
                new_name = st.text_input("Rename", value=target, key="rename_input")
                colr1, colr2 = st.columns([3, 1])
                with colr2:
                    if st.button("Confirm Rename"):
                        if new_name and new_name != target:
                            st.session_state["saved_searches"][new_name] = st.session_state["saved_searches"].pop(target)
                        st.session_state["rename_target"] = None
                        st.rerun()
                    if st.button("Cancel Rename"):
                        st.session_state["rename_target"] = None
                        st.rerun()

        if st.button("Clear saved searches"):
            st.session_state["saved_searches"] = {}
            st.success("Cleared saved searches")


        hunt_btn = st.button("🎯  HUNT LIVE DEALS", type="primary", use_container_width=True)

    if hunt_btn:
        source_list = ", ".join(filter(None, [
            "Acquire.com" if use_acq_auth else None,
            "SideProjectors" if use_side_proj else None,
            "Flippa" if use_flippa else None,
            "Reddit" if use_reddit else None,
            "Empire Flippers" if use_ef else None,
            "Web Search" if use_websrch else None,
        ]))
        with st.spinner(f"Hunting live deals across: {source_list}..."):
            try:
                deals = hunt_live_deals(
                    use_acquire_auth=use_acq_auth,
                    use_ef=use_ef,
                    use_flippa=use_flippa,
                    use_reddit=use_reddit,
                    use_acquire_rss=use_acq_rss,
                    use_web_search=use_websrch,
                    use_side_projectors=use_side_proj,
                    max_total=150,
                )
                st.session_state["hunted_deals"] = deals
            except Exception as exc:
                st.error(f"Hunt error: {exc}")
                deals = []

    deals = st.session_state.get("hunted_deals", [])

    if deals:
        # Filter
        filtered = [d for d in deals if d.get("score", 0) >= min_score]
        if india_only:
            filtered = [d for d in filtered if d.get("country") == "India" or
                        re.search(r"\bindia\b|₹|lakh|crore", (d.get("description","") + d.get("title","")).lower())]
        if max_price > 0:
            filtered = [d for d in filtered if
                        d.get("asking_price_usd") is None or d.get("asking_price_usd", 0) <= max_price]

        # Stats bar
        sources = {}
        for d in filtered:
            sources[d["source"]] = sources.get(d["source"], 0) + 1
        src_str = "  ·  ".join(f"{s}: {n}" for s, n in sorted(sources.items(), key=lambda x: -x[1]))

        st.markdown(
            f'<div style="background:#0f172a;border:1px solid #1e293b;border-radius:6px;'
            f'padding:10px 16px;margin-bottom:16px;font-family:JetBrains Mono;font-size:11px;">'
            f'<span style="color:#10b981">✓ {len(filtered)} deals found</span>'
            f'<span style="color:#475569"> (from {len(deals)} total · {src_str})</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Deal cards
        for idx, deal in enumerate(filtered[:40]):
            score = deal.get("score", 0)
            score_color = "#10b981" if score >= 60 else ("#f59e0b" if score >= 35 else "#ef4444")
            ask_str = (f"${deal['asking_price_usd']:,.0f}" if deal.get("asking_price_usd")
                       else (f"₹{deal['asking_price_inr_cr']:.1f}Cr" if deal.get("asking_price_inr_cr") else "—"))
            mrr_str = (f"${deal['monthly_revenue_usd']:,.0f}/mo" if deal.get("monthly_revenue_usd") else "—")
            profit_str = (f"${deal['monthly_profit_usd']:,.0f}/mo profit" if deal.get("monthly_profit_usd") else "")
            mult_str = (f"{deal['multiple']:.1f}x" if deal.get("multiple") else "—")
            country_badge = (f'<span style="background:#1e3a5f;color:#93c5fd;padding:1px 6px;'
                             f'border-radius:3px;font-size:9px;margin-left:6px">{deal["country"]}</span>'
                             if deal.get("country") else "")
            india_flag = "🇮🇳 " if deal.get("country") == "India" else ""

            desc_preview = deal.get("description", "")[:200].replace("\n", " ")
            # pre-render profit HTML to avoid backslashes inside f-string expressions
            profit_html = (f'<span><span style="color:#64748b">PROFIT </span>'
                           f'<span style="color:#34d399">{profit_str}</span></span>' if profit_str else "")
            # pre-render seller handle HTML to avoid backslashes in f-strings
            seller_html = (f'<span style="font-size:10px;color:#475569">{deal.get("seller_handle","")}</span>' 
                           if deal.get("seller_handle") else "")

            # card wrapper with anchor for keyboard nav
            st.markdown(f'<div id="card-{idx}" class="deal-card">', unsafe_allow_html=True)

            st.markdown(
                f'<div style="background:#0d1117;border:1px solid #1e293b;border-radius:8px;'
                f'padding:14px 18px;margin-bottom:10px;">'
                f'<div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:6px;">'
                f'<div style="flex:1;">'
                f'<span style="font-family:JetBrains Mono;font-size:13px;font-weight:700;color:#e2e8f0">'
                f'{india_flag}{deal["title"][:90]}</span>{country_badge}'
                f'<br><span style="font-size:10px;color:#64748b">'
                f'{deal["source"]} · {deal.get("scrape_method","?")} · {deal.get("niche","") or deal.get("monetization","")}'
                f'</span>'
                f'</div>'
                f'<div style="text-align:right;min-width:120px;">'
                f'<span style="font-size:18px;font-weight:700;font-family:JetBrains Mono;color:{score_color}">'
                f'{score:.0f}</span><span style="font-size:9px;color:#475569">/100</span>'
                f'</div>'
                f'</div>'
                f'<div style="display:flex;gap:20px;font-family:JetBrains Mono;font-size:11px;margin-bottom:8px;">'
                f'<span><span style="color:#64748b">ASK </span><span style="color:#f59e0b;font-weight:700">{ask_str}</span></span>'
                f'<span><span style="color:#64748b">MRR </span><span style="color:#10b981">{mrr_str}</span></span>'
                f'<span><span style="color:#64748b">MULT </span><span style="color:#a78bfa">{mult_str}</span></span>'
                f'{profit_html}'
                f'</div>'
                f'<div style="font-size:11px;color:#94a3b8;margin-bottom:8px">{desc_preview}{"..." if len(deal.get("description","")) > 200 else ""}</div>'
                f'<div style="display:flex;gap:10px;align-items:center;">'
                f'<a href="{deal["url"]}" target="_blank" style="font-family:JetBrains Mono;font-size:10px;'
                f'color:#60a5fa;text-decoration:none;background:#1e3a5f;padding:3px 10px;border-radius:4px;">'
                f'→ VIEW LISTING</a>'
                f'{seller_html}'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # hidden select anchor (use query param to select via keyboard JS)
            select_href = f'?select={idx}#card-{idx}'
            st.markdown(f'<a id="select-link-{idx}" href="{select_href}" style="display:none">select</a>', unsafe_allow_html=True)
            # visible select button for mouse users
            if st.button("Select", key=f"select_{deal.get('listing_id') or (deal.get('url') or '')[-12:]}"):
                st.session_state["selected_deal"] = deal
                st.rerun()

            st.markdown('</div>', unsafe_allow_html=True)

            # One-click: load this deal into analysis tab
            if st.button(f"⚡ Analyse", key=f"analyse_{deal['listing_id'] or deal['url'][-12:]}"):
                st.session_state["scraped_deal"] = {
                    "title":       deal["title"],
                    "description": deal.get("description", ""),
                    "url":         deal["url"],
                    "source":      deal["source"],
                    "scrape_source": deal.get("scrape_method", "live-hunt"),
                    "asking_price_usd":  deal.get("asking_price_usd"),
                    "asking_price_inr_cr": deal.get("asking_price_inr_cr"),
                    "mrr_usd":     deal.get("monthly_revenue_usd"),
                    "revenue_cr":  deal.get("revenue_cr", 0.0),
                    "ebitda_l":    deal.get("ebitda_l", 0.0),
                    "reason_for_sale": "",
                    "sector_hint": deal.get("niche", ""),
                    "location":    deal.get("country", ""),
                    "highlights":  [],
                    "risks":       [],
                    "seller_handle": deal.get("seller_handle", ""),
                }
                st.success("Deal loaded into Deal Analysis tab!")
        # Inject keyboard navigation JS after rendering cards
        try:
            import streamlit.components.v1 as components

            js = f'''
            <script>
            (function() {{
              const total = {len(filtered[:40])};
              if (!total) return;
              let idx = 0;
              function highlight() {{
                for (let i=0;i<total;i++) {{
                  const el = document.getElementById('card-'+i);
                  if (!el) continue;
                  if (i===idx) el.style.outline = '2px solid #60a5fa';
                  else el.style.outline = 'none';
                }}
                const target = document.getElementById('card-'+idx);
                if(target) target.scrollIntoView({{behavior:'smooth', block:'center'}});
              }}
              document.addEventListener('keydown', function(e){{
                if (e.key === 'j') {{ idx = Math.min(total-1, idx+1); highlight(); e.preventDefault(); }}
                else if (e.key === 'k') {{ idx = Math.max(0, idx-1); highlight(); e.preventDefault(); }}
                else if (e.key === 's') {{ window.location.href = '?select='+idx + '#card-'+idx; }}
                else if (e.key === 'Enter') {{ window.location.href = '?select='+idx + '#card-'+idx; }}
              }});
              // initial highlight
              setTimeout(highlight, 300);
            }})();
            </script>
            '''
            components.html(js, height=10)
        except Exception:
            pass

    elif not hunt_btn:
        st.markdown(
            '<div style="background:#0f172a;border:1px dashed #1e3a5f;border-radius:8px;'
            'padding:40px;text-align:center;">'
            '<div style="font-family:JetBrains Mono;font-size:14px;color:#475569">Hit HUNT LIVE DEALS to scan</div>'
            '<div style="font-size:11px;color:#374151;margin-top:8px">'
            'Empire Flippers · Flippa · Reddit r/SaaS · Acquire.com RSS · Firecrawl Web Search'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )

# Detail drawer: selected deal shown in the sidebar for quick actions
selected = st.session_state.get("selected_deal")
if selected:
    st.sidebar.markdown("## Selected Deal")
    st.sidebar.markdown(f"**{selected.get('title','Untitled')}**")
    st.sidebar.markdown(f"*Source:* {selected.get('source','')}")
    ask = selected.get("asking_price_usd")
    st.sidebar.markdown(f"*Ask:* {'$%s' % f'{ask:,.0f}' if ask else 'n/a'}")
    st.sidebar.markdown(f"*MRR:* {selected.get('monthly_revenue_usd') or 'n/a'}")
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Description**")
    st.sidebar.write(selected.get("description",""))

    if st.sidebar.button("Save to Pipeline"):
        pipeline = st.session_state.get("pipeline", [])
        pipeline.append(selected)
        st.session_state["pipeline"] = pipeline
        st.sidebar.success("Saved to pipeline")

    # LOI draft (simple)
    loi_default = "Hi,\n\nI'm interested in acquiring {title}. Could we discuss terms?\n\nAsking price: {ask}\n\nRegards,\nPocket Fund\n".format(
        title=selected.get("title"), ask=(f"${ask:,.0f}" if ask else "TBD")
    )
    loi = st.sidebar.text_area("LOI draft", value=st.session_state.get("loi_draft", loi_default), height=200)
    if st.sidebar.button("Save LOI draft"):
        st.session_state["loi_draft"] = loi
        st.sidebar.success("LOI saved")

    if st.sidebar.button("Clear selection"):
        st.session_state.pop("selected_deal", None)
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div style="text-align:center;font-family:JetBrains Mono;font-size:10px;color:#374151;padding:8px;">'
    '⚠️ PROTOTYPE · All regulatory signals are simulated · Public data only · Not financial or legal advice<br>'
    'Pocket Fund Acquisition Intelligence Layer · '
    f'{datetime.utcnow().strftime("%Y-%m-%d")} · streamlit run app.py'
    '</div>',
    unsafe_allow_html=True
)
