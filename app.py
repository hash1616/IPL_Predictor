"""
IPL Match Prediction - Streamlit Web Application
Phase 8: Professional UI with win probability bars, H2H stats, and confidence scores.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from src.predict import IPLPredictor

# ─── Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="IPL Match Predictor",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;900&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
  }

  /* Dark gradient background */
  .stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    min-height: 100vh;
  }

  /* Main header */
  .main-header {
    text-align: center;
    padding: 2rem 0 1rem 0;
  }
  .main-header h1 {
    font-size: 3rem;
    font-weight: 900;
    background: linear-gradient(90deg, #f7971e, #ffd200, #f7971e);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -1px;
    margin-bottom: 0.25rem;
  }
  .main-header p {
    color: #a0aec0;
    font-size: 1.05rem;
    font-weight: 400;
    margin-top: 0;
  }

  /* Glassmorphism card */
  .glass-card {
    background: rgba(255,255,255,0.07);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 20px;
    padding: 2rem;
    margin-bottom: 1.5rem;
  }

  /* Section titles */
  .section-title {
    font-size: 1rem;
    font-weight: 600;
    color: #ffd200;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 1.2rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }

  /* Winner banner */
  .winner-banner {
    background: linear-gradient(135deg, #f7971e 0%, #ffd200 100%);
    border-radius: 16px;
    padding: 1.8rem 2rem;
    text-align: center;
    margin: 1rem 0;
  }
  .winner-banner .label {
    font-size: 0.85rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 3px;
    color: rgba(0,0,0,0.6);
    margin-bottom: 0.3rem;
  }
  .winner-banner .name {
    font-size: 2.2rem;
    font-weight: 900;
    color: #1a1a2e;
    letter-spacing: -0.5px;
  }
  .winner-banner .trophy {
    font-size: 2.5rem;
  }

  /* Prob bar container */
  .prob-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.6rem;
  }
  .prob-label {
    font-size: 0.9rem;
    font-weight: 600;
    color: #e2e8f0;
    min-width: 160px;
  }
  .prob-pct {
    font-size: 1.1rem;
    font-weight: 700;
    min-width: 55px;
    text-align: right;
  }

  /* H2H stat boxes */
  .stat-box {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
  }
  .stat-box .stat-value {
    font-size: 2rem;
    font-weight: 800;
    color: #ffd200;
    line-height: 1;
  }
  .stat-box .stat-label {
    font-size: 0.75rem;
    color: #718096;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 0.25rem;
  }

  /* Confidence badge */
  .confidence-badge {
    display: inline-block;
    padding: 0.3rem 0.9rem;
    border-radius: 50px;
    font-weight: 700;
    font-size: 0.85rem;
  }
  .conf-high  { background: #22543d; color: #68d391; border: 1px solid #48bb78; }
  .conf-med   { background: #744210; color: #f6ad55; border: 1px solid #ed8936; }
  .conf-low   { background: #742a2a; color: #fc8181; border: 1px solid #f56565; }

  /* Selectbox styling */
  .stSelectbox > div > div {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 10px !important;
    color: white !important;
  }
  .stSelectbox label {
    color: #a0aec0 !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
  }

  /* Predict button */
  .stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #f7971e 0%, #ffd200 100%) !important;
    color: #1a1a2e !important;
    font-weight: 800 !important;
    font-size: 1.1rem !important;
    padding: 0.75rem 1rem !important;
    border-radius: 12px !important;
    border: none !important;
    letter-spacing: 0.5px;
    transition: all 0.2s;
    box-shadow: 0 4px 20px rgba(247,151,30,0.4);
  }
  .stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(247,151,30,0.6) !important;
  }

  /* Match result row */
  .match-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.6rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    font-size: 0.83rem;
  }
  .match-winner { color: #68d391; font-weight: 600; }
  .match-loser  { color: #718096; }
  .match-date   { color: #4a5568; font-size: 0.75rem; }

  /* Hide Streamlit branding */
  #MainMenu {visibility: hidden;}
  footer {visibility: hidden;}
  header {visibility: hidden;}

  /* Divider */
  hr { border-color: rgba(255,255,255,0.08) !important; }
</style>
""", unsafe_allow_html=True)


# ─── Current Active IPL Teams (2024 Season onwards) ─────────────────────────
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

# ─── Load Predictor (cached) ──────────────────────────────────────────────────
@st.cache_resource
def load_predictor():
    return IPLPredictor()


# ─── Helper: Draw probability bar chart ──────────────────────────────────────
def draw_prob_chart(team1, team2, prob1, prob2):
    fig, ax = plt.subplots(figsize=(6, 1.6))
    fig.patch.set_alpha(0)
    ax.set_facecolor('none')

    colors1 = '#f7971e'
    colors2 = '#4299e1'
    height = 0.42

    ax.barh(1, prob1, height=height, color=colors1, left=0)
    ax.barh(0, prob2, height=height, color=colors2, left=0)

    ax.set_xlim(0, 100)
    ax.set_yticks([0, 1])
    ax.set_yticklabels([team2[:18], team1[:18]], color='white', fontsize=9, fontweight='600')
    ax.tick_params(axis='x', colors='#4a5568', labelsize=8)
    ax.tick_params(axis='y', length=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#2d3748')

    ax.text(prob1 + 1, 1, f"{prob1:.1f}%", va='center', color='#f7971e',
            fontsize=10, fontweight='700')
    ax.text(prob2 + 1, 0, f"{prob2:.1f}%", va='center', color='#4299e1',
            fontsize=10, fontweight='700')

    plt.tight_layout(pad=0.3)
    return fig


# ─── Helper: H2H donut chart ─────────────────────────────────────────────────
def draw_h2h_donut(t1_wins, t2_wins, team1, team2):
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    fig.patch.set_alpha(0)
    ax.set_facecolor('none')

    total = t1_wins + t2_wins
    if total == 0:
        sizes = [1, 1]
        colors = ['#2d3748', '#2d3748']
    else:
        sizes = [t1_wins, t2_wins]
        colors = ['#f7971e', '#4299e1']

    wedges, _ = ax.pie(sizes, colors=colors, startangle=90,
                       wedgeprops=dict(width=0.55, edgecolor='none'))

    # Center text
    ax.text(0, 0.05, str(total), ha='center', va='center',
            fontsize=22, fontweight='900', color='white')
    ax.text(0, -0.22, 'matches', ha='center', va='center',
            fontsize=8, color='#718096')

    ax.legend(
        [mpatches.Patch(color='#f7971e'), mpatches.Patch(color='#4299e1')],
        [f"{team1[:14]} ({t1_wins})", f"{team2[:14]} ({t2_wins})"],
        loc='lower center', bbox_to_anchor=(0.5, -0.18),
        ncol=1, frameon=False,
        labelcolor='white', fontsize=7.5
    )

    plt.tight_layout()
    return fig


# ─── App Header ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>🏏 IPL Match Predictor</h1>
  <p>AI-powered pre-match winner prediction using historical IPL data (2008–2024)</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ─── Load predictor ───────────────────────────────────────────────────────────
try:
    predictor = load_predictor()
    teams  = predictor.teams
    venues = predictor.venues
except Exception as e:
    st.error(f"Failed to load predictor: {e}")
    st.stop()

# ─── Layout: Input | Results ──────────────────────────────────────────────────
col_input, col_spacer, col_result = st.columns([1.1, 0.1, 1.8])

with col_input:
    st.markdown('<div class="section-title">⚙️ Match Setup</div>', unsafe_allow_html=True)

    team1 = st.selectbox("🔵 Team A (Batting First)", ACTIVE_TEAMS,
                         index=ACTIVE_TEAMS.index("Mumbai Indians"))
    team2_options = [t for t in ACTIVE_TEAMS if t != team1]
    team2 = st.selectbox("🔴 Team B (Bowling First)", team2_options,
                         index=team2_options.index("Chennai Super Kings") if "Chennai Super Kings" in team2_options else 0)

    venue = st.selectbox("🏟️ Venue", venues,
                         index=venues.index("Wankhede Stadium") if "Wankhede Stadium" in venues else 0)

    toss_winner = st.selectbox("🪙 Toss Winner", [team1, team2])
    toss_decision = st.selectbox("🏏 Toss Decision", ["bat", "field"],
                                  format_func=lambda x: "Bat First" if x == "bat" else "Field First")

    st.markdown("<br>", unsafe_allow_html=True)
    predict_clicked = st.button("🚀 Predict Winner", use_container_width=True)

with col_result:
    if predict_clicked:
        if team1 == team2:
            st.error("⚠️ Please select two different teams.")
        else:
            with st.spinner("Analysing match conditions..."):
                result = predictor.predict(team1, team2, venue, toss_winner, toss_decision)
                h2h    = predictor.get_h2h_stats(team1, team2)

            # ── Winner Banner ──────────────────────────────────────────────
            st.markdown(f"""
            <div class="winner-banner">
              <div class="trophy">🏆</div>
              <div class="label">Predicted Winner</div>
              <div class="name">{result['winner']}</div>
            </div>
            """, unsafe_allow_html=True)

            # ── Win Probability Chart ──────────────────────────────────────
            st.markdown('<div class="section-title">📊 Win Probabilities</div>', unsafe_allow_html=True)
            chart = draw_prob_chart(team1, team2, result['team1_prob'], result['team2_prob'])
            st.pyplot(chart, width='stretch')
            plt.close()

            # Confidence badge
            conf = result['confidence']
            if conf >= 60:
                badge_class, badge_text = "conf-high", "High Confidence"
            elif conf >= 53:
                badge_class, badge_text = "conf-med", "Moderate Confidence"
            else:
                badge_class, badge_text = "conf-low", "Low Confidence"

            st.markdown(f"""
            <div style="text-align:center; margin: 0.5rem 0 1.2rem 0;">
              <span class="confidence-badge {badge_class}">
                {badge_text} &nbsp;·&nbsp; {conf}%
              </span>
            </div>
            """, unsafe_allow_html=True)

            # ── Head-to-Head ───────────────────────────────────────────────
            st.markdown('<div class="section-title">⚔️ Head-to-Head</div>', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"""
                <div class="stat-box">
                  <div class="stat-value">{h2h['total']}</div>
                  <div class="stat-label">Total Matches</div>
                </div>""", unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div class="stat-box">
                  <div class="stat-value" style="color:#f7971e">{h2h['team1_wins']}</div>
                  <div class="stat-label">{team1[:16]} Wins</div>
                </div>""", unsafe_allow_html=True)
            with c3:
                st.markdown(f"""
                <div class="stat-box">
                  <div class="stat-value" style="color:#4299e1">{h2h['team2_wins']}</div>
                  <div class="stat-label">{team2[:16]} Wins</div>
                </div>""", unsafe_allow_html=True)

            # H2H Donut
            if h2h['total'] > 0:
                st.markdown("<br>", unsafe_allow_html=True)
                donut_col, info_col = st.columns([1, 1.4])
                with donut_col:
                    donut = draw_h2h_donut(h2h['team1_wins'], h2h['team2_wins'], team1, team2)
                    st.pyplot(donut, width='stretch')
                    plt.close()

                with info_col:
                    st.markdown(f"**Last {min(5, h2h['total'])} Encounters**")
                    for match in reversed(h2h['last_5']):
                        date_str = str(match['date'])[:10]
                        win = match['winner']
                        lose = team2 if win == team1 else team1
                        st.markdown(
                            f"<div class='match-row'>"
                            f"<span class='match-winner'>✓ {win[:18]}</span>"
                            f"<span class='match-date'>{date_str}</span>"
                            f"</div>",
                            unsafe_allow_html=True
                        )

            # ── Model Info Footer ──────────────────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            st.caption(
                f"Model: **{result['model_name']}** &nbsp;|&nbsp; "
                f"CV Accuracy: **{result['model_accuracy']}%** &nbsp;|&nbsp; "
                f"Trained on **1,169 IPL matches (2008–2024)**"
            )

    else:
        # Placeholder state
        st.markdown("""
        <div style="text-align:center; padding: 4rem 2rem; color: #4a5568;">
          <div style="font-size: 4rem; margin-bottom: 1rem;">🏏</div>
          <div style="font-size: 1.1rem; font-weight: 600; color: #718096;">
            Select teams and match conditions,<br>then click <strong style="color:#ffd200">Predict Winner</strong>
          </div>
        </div>
        """, unsafe_allow_html=True)

# ─── Footer stats strip ───────────────────────────────────────────────────────
st.markdown("---")
f1, f2, f3, f4 = st.columns(4)
f1.metric("📅 IPL Seasons", "17")
f2.metric("🏟️ Matches Trained On", "1,169")
f3.metric("🏏 Active Teams", "10")
f4.metric("📍 Venues", "37")
