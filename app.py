"""
IPL Match Prediction - Streamlit Web Application
Professional Dual-Tone UI with premium glassmorphism, dynamic animations, and high-fidelity layouts.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
from src.predict import IPLPredictor

# ─── Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="IPL Predictor Pro",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── Custom CSS for Premium Dual-Tone UI ──────────────────────────────────────
# Theme: Obsidian Navy (#0A0E17) & Electric Teal (#00F2FE) / Neon Purple (#4FACFE)
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

  /* Global typography & base styling */
  html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
  }

  .stApp {
    background: radial-gradient(circle at 50% 50%, #111827 0%, #030712 100%);
    min-height: 100vh;
  }

  /* Main header layout */
  .main-header {
    text-align: center;
    padding: 3rem 0 1.5rem 0;
  }
  .main-header h1 {
    font-size: 3.5rem;
    font-weight: 800;
    letter-spacing: -2px;
    background: linear-gradient(135deg, #00F2FE 0%, #4FACFE 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
  }
  .main-header p {
    color: #9CA3AF;
    font-size: 1.1rem;
    font-weight: 400;
  }

  /* Glassmorphism containers */
  .glass-panel {
    background: rgba(17, 24, 39, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 24px;
    padding: 2.2rem;
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    margin-bottom: 1.5rem;
  }

  .section-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #F3F4F6;
    letter-spacing: 0.5px;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 0.75rem;
    border-left: 4px solid #00F2FE;
    padding-left: 0.75rem;
  }

  /* Form & Interactive elements */
  .stSelectbox label {
    color: #9CA3AF !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    margin-bottom: 0.5rem !important;
  }

  div[data-baseweb="select"] {
    background-color: rgba(255, 255, 255, 0.03) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 12px !important;
    color: #F3F4F6 !important;
  }

  /* Predict Button styling */
  .stButton > button {
    background: linear-gradient(135deg, #00F2FE 0%, #4FACFE 100%) !important;
    color: #030712 !important;
    font-weight: 700 !important;
    font-size: 1.1rem !important;
    padding: 0.85rem 1.5rem !important;
    border-radius: 14px !important;
    border: none !important;
    box-shadow: 0 8px 25px rgba(0, 242, 254, 0.25);
    transition: all 0.3s ease;
    cursor: pointer;
  }
  .stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 35px rgba(0, 242, 254, 0.45);
  }

  /* Dual-Tone Winner Display Card */
  .winner-card {
    background: linear-gradient(135deg, rgba(0, 242, 254, 0.08) 0%, rgba(79, 172, 254, 0.08) 100%);
    border: 1px solid rgba(0, 242, 254, 0.3);
    border-radius: 20px;
    padding: 2.5rem 2rem;
    text-align: center;
    box-shadow: 0 10px 30px rgba(0, 242, 254, 0.05);
    margin-bottom: 2rem;
  }
  .winner-card .trophy {
    font-size: 3rem;
    margin-bottom: 0.75rem;
  }
  .winner-card .label {
    font-size: 0.8rem;
    font-weight: 700;
    color: #00F2FE;
    text-transform: uppercase;
    letter-spacing: 3px;
    margin-bottom: 0.5rem;
  }
  .winner-card .name {
    font-size: 2.6rem;
    font-weight: 800;
    color: #FFFFFF;
    text-shadow: 0 0 20px rgba(0, 242, 254, 0.4);
  }

  /* Stat details boxes */
  .stat-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin-bottom: 1.5rem;
  }
  .stat-box {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 16px;
    padding: 1.2rem;
    text-align: center;
  }
  .stat-box .val {
    font-size: 1.8rem;
    font-weight: 800;
    color: #00F2FE;
  }
  .stat-box .lbl {
    font-size: 0.75rem;
    color: #9CA3AF;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 0.25rem;
  }

  /* HTML-based Custom Dual-Tone Probability Progress Bar */
  .prob-container {
    margin: 1.5rem 0;
  }
  .prob-labels {
    display: flex;
    justify-content: space-between;
    font-weight: 700;
    font-size: 0.95rem;
    margin-bottom: 0.5rem;
    color: #E5E7EB;
  }
  .prob-bar-bg {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 50px;
    height: 16px;
    overflow: hidden;
    display: flex;
    border: 1px solid rgba(255, 255, 255, 0.08);
  }
  .prob-fill-t1 {
    background: linear-gradient(90deg, #00F2FE, #4FACFE);
    height: 100%;
    border-radius: 50px 0 0 50px;
  }
  .prob-fill-t2 {
    background: linear-gradient(90deg, #EC4899, #F43F5E);
    height: 100%;
    border-radius: 0 50px 50px 0;
  }

  /* Confidence level pill */
  .conf-pill {
    display: inline-block;
    padding: 0.4rem 1.2rem;
    border-radius: 100px;
    font-size: 0.8rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
  }
  .conf-high { background: rgba(16, 185, 129, 0.15); color: #34D399; border: 1px solid rgba(52, 211, 153, 0.3); }
  .conf-med  { background: rgba(245, 158, 11, 0.15); color: #FBBF24; border: 1px solid rgba(251, 191, 36, 0.3); }
  .conf-low  { background: rgba(239, 68, 68, 0.15); color: #FCA5A5; border: 1px solid rgba(252, 165, 165, 0.3); }

  /* Match History Row */
  .history-row {
    display: flex;
    justify-content: space-between;
    padding: 0.8rem 0;
    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    font-size: 0.85rem;
  }
  .history-win { color: #34D399; font-weight: 600; }
  .history-date { color: #6B7280; font-size: 0.75rem; }

  /* Metric cards */
  .stMetric {
    background: rgba(255, 255, 255, 0.02) !important;
    border: 1px solid rgba(255, 255, 255, 0.04) !important;
    padding: 1rem !important;
    border-radius: 16px !important;
  }
</style>
""", unsafe_allow_html=True)

# ─── App Header ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>🏏 IPL PREDICTOR PRO</h1>
  <p>Dual-Tone Premium Sports Predictor driven by machine learning metrics (2008–2026)</p>
</div>
""", unsafe_allow_html=True)

# ─── Load Predictor ───────────────────────────────────────────────────────────
@st.cache_resource
def load_predictor():
    return IPLPredictor()

try:
    predictor = load_predictor()
    venues = predictor.venues
except Exception as e:
    st.error(f"Failed to load predictor: {e}")
    st.stop()

# Active Teams
ACTIVE_TEAMS = [
    "Chennai Super Kings",
    "Delhi Capitals",
    "Gujarat Titans",
    "Kolkata Knight Riders",
    "Lucknow Super Giants",
    "Mumbai Indians",
    "Punjab Kings",
    "Rajasthan Royals",
    "Royal Challengers Bengaluru",
    "Sunrisers Hyderabad",
]

# Layout grid: Inputs on Left, Results on Right
col_left, col_mid, col_right = st.columns([1.2, 0.1, 1.7])

with col_left:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">⚙️ Match Settings</div>', unsafe_allow_html=True)

    team1 = st.selectbox("🔵 Team A (Batting First)", ACTIVE_TEAMS, index=ACTIVE_TEAMS.index("Mumbai Indians"))
    team2_opts = [t for t in ACTIVE_TEAMS if t != team1]
    team2 = st.selectbox("🔴 Team B (Bowling First)", team2_opts, index=team2_opts.index("Chennai Super Kings") if "Chennai Super Kings" in team2_opts else 0)
    
    venue = st.selectbox("🏟️ Venue", venues, index=venues.index("Wankhede Stadium") if "Wankhede Stadium" in venues else 0)
    toss_winner = st.selectbox("🪙 Toss Winner", [team1, team2])
    toss_decision = st.selectbox("🏏 Toss Decision", ["bat", "field"], format_func=lambda x: "Bat First" if x == "bat" else "Field First")

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🚀 Predict Win Probability", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    if predict_btn:
        with st.spinner("Analyzing matchup metrics..."):
            result = predictor.predict(team1, team2, venue, toss_winner, toss_decision)
            h2h = predictor.get_h2h_stats(team1, team2)

        # Main glass panel for outputs
        st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
        
        # Winner Display Card
        st.markdown(f"""
        <div class="winner-card">
          <div class="trophy">🏆</div>
          <div class="label">Predicted Winner</div>
          <div class="name">{result['winner']}</div>
        </div>
        """, unsafe_allow_html=True)

        # Custom HTML Dual-Tone Bar
        t1_prob = result['team1_prob']
        t2_prob = result['team2_prob']
        
        # Determine confidence class
        conf = result['confidence']
        if conf >= 60:
            conf_class, conf_text = "conf-high", "High Confidence"
        elif conf >= 53:
            conf_class, conf_text = "conf-med", "Medium Confidence"
        else:
            conf_class, conf_text = "conf-low", "Low Confidence"

        st.markdown(f"""
        <div class="section-title">📊 Dual-Tone Win Likelihood</div>
        <div class="prob-container">
          <div class="prob-labels">
            <span>{team1[:22]} (Team A)</span>
            <span>{team2[:22]} (Team B)</span>
          </div>
          <div class="prob-bar-bg">
            <div class="prob-fill-t1" style="width: {t1_prob}%;"></div>
            <div class="prob-fill-t2" style="width: {t2_prob}%;"></div>
          </div>
          <div class="prob-labels" style="margin-top: 0.5rem; font-size: 1.1rem; color: #F3F4F6;">
            <span>{t1_prob:.1f}%</span>
            <span>{t2_prob:.1f}%</span>
          </div>
        </div>
        
        <div style="text-align: center; margin-bottom: 2rem;">
          <span class="conf-pill {conf_class}">{conf_text} · {conf:.1f}%</span>
        </div>
        """, unsafe_allow_html=True)

        # Head-To-Head Stats
        st.markdown('<div class="section-title">⚔️ Head-to-Head Stats</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="stat-grid">
          <div class="stat-box">
            <div class="val">{h2h['total']}</div>
            <div class="lbl">Played</div>
          </div>
          <div class="stat-box">
            <div class="val" style="color: #00F2FE;">{h2h['team1_wins']}</div>
            <div class="lbl">{team1[:10]} Wins</div>
          </div>
          <div class="stat-box">
            <div class="val" style="color: #EC4899;">{h2h['team2_wins']}</div>
            <div class="lbl">{team2[:10]} Wins</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Last Encounters list
        if h2h['total'] > 0:
            st.markdown(f"**Last {min(5, h2h['total'])} Match Outcomes:**")
            for match in reversed(h2h['last_5']):
                m_date = str(match['date'])[:10]
                m_winner = match['winner']
                st.markdown(f"""
                <div class="history-row">
                  <span class="history-win">✓ {m_winner}</span>
                  <span class="history-date">{m_date}</span>
                </div>
                """, unsafe_allow_html=True)

        # Footer model details
        st.markdown("<br><hr>", unsafe_allow_html=True)
        st.caption(
            f"Model: **{result['model_name']}** &nbsp;|&nbsp; "
            f"Accuracy: **{result['model_accuracy']}%** &nbsp;|&nbsp; "
            f"Data Scope: **2008–2026 IPL seasons**"
        )
        st.markdown('</div>', unsafe_allow_html=True)

    else:
        # Beautiful Dual-Tone placeholder state
        st.markdown("""
        <div class="glass-panel" style="text-align:center; padding: 5rem 2rem; border-style: dashed; border-color: rgba(255,255,255,0.08);">
          <div style="font-size: 4rem; margin-bottom: 1.5rem; filter: drop-shadow(0 0 20px rgba(0, 242, 254, 0.3));">🏏</div>
          <h3 style="color: #FFFFFF; font-weight: 700; margin-bottom: 0.5rem;">Ready to Simulate Matchup</h3>
          <p style="color: #9CA3AF; font-size: 0.95rem; max-width: 380px; margin: 0 auto 1.5rem auto;">
            Configure the teams, venue, and toss settings on the left panel, then trigger the AI model to calculate win probabilities.
          </p>
        </div>
        """, unsafe_allow_html=True)

# ─── Footer Stats Strip ───────────────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
f1, f2, f3, f4 = st.columns(4)
f1.metric("📅 IPL Seasons", "19")
f2.metric("🏟️ Matches Trained On", "1,234")
f3.metric("🏏 Active Teams", "10")
f4.metric("📍 Match Venues", f"{len(venues)}")
