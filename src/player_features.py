"""
Player Feature Engineering for IPL Match Prediction.

Extracts per-player batting and bowling stats from ball-by-ball data,
then aggregates to team-level features for each match.

IMPORTANT: All stats are computed using ONLY matches PRIOR to each match
to prevent data leakage.

Player features added per team:
  Batting:
    - team_avg_batter_avg      : avg runs-per-innings of top 5 batters (all-time prior)
    - team_avg_batter_sr       : avg strike rate of top 5 batters (all-time prior)
    - team_avg_batter_recent_avg: avg runs-per-innings of top 5 batters (last 3 innings)
    - team_avg_batter_recent_sr : avg strike rate in last 3 innings
  Bowling:
    - team_avg_bowler_economy  : avg economy rate of top 4 bowlers (all-time prior)
    - team_avg_bowler_wickets  : avg wickets per match of top 4 bowlers (all-time prior)
    - team_avg_bowler_recent_eco: avg economy in last 3 matches
"""

import os
import pandas as pd
import numpy as np


# ── Utility: Compute per-player batting stats ─────────────────────────────────

def get_player_batting_stats(deliveries, player, match_id_list=None, last_n=None):
    """
    Compute batting stats for a player across given match IDs.
    
    Returns: (avg_runs_per_innings, avg_strike_rate)
    """
    d = deliveries[deliveries['batter'] == player].copy()
    if match_id_list is not None:
        d = d[d['match_id'].isin(match_id_list)]
    if d.empty:
        return 0.0, 0.0

    # Per-innings aggregation
    innings_stats = d.groupby(['match_id', 'innings']).agg(
        runs=('runs_batter', 'sum'),
        balls=('valid_ball', 'sum')
    ).reset_index()

    if last_n is not None:
        innings_stats = innings_stats.tail(last_n)

    if innings_stats.empty:
        return 0.0, 0.0

    if last_n == 3:
        # Momentum-weighted average (last elements are most recent)
        n = len(innings_stats)
        if n == 3:
            weights = np.array([0.2, 0.3, 0.5])
        elif n == 2:
            weights = np.array([0.375, 0.625])
        else:
            weights = np.array([1.0])
            
        avg_runs = np.sum(innings_stats['runs'] * weights)
        w_runs = np.sum(innings_stats['runs'] * weights)
        w_balls = np.sum(innings_stats['balls'] * weights)
        avg_sr = (w_runs / w_balls * 100) if w_balls > 0 else 0.0
    else:
        avg_runs = innings_stats['runs'].mean()
        total_balls = innings_stats['balls'].sum()
        total_runs = innings_stats['runs'].sum()
        avg_sr = (total_runs / total_balls * 100) if total_balls > 0 else 0.0

    return round(avg_runs, 3), round(avg_sr, 3)


def get_player_bowling_stats(deliveries, player, match_id_list=None, last_n=None):
    """
    Compute bowling stats for a player across given match IDs.
    
    Returns: (avg_economy, avg_wickets_per_match)
    """
    d = deliveries[deliveries['bowler'] == player].copy()
    if match_id_list is not None:
        d = d[d['match_id'].isin(match_id_list)]
    if d.empty:
        return 8.0, 0.0  # Default economy = 8 (mediocre)

    match_stats = d.groupby('match_id').agg(
        runs=('runs_bowler', 'sum'),
        balls=('valid_ball', 'sum'),
        wickets=('bowler_wicket', 'sum')
    ).reset_index()

    if last_n is not None:
        match_stats = match_stats.tail(last_n)

    if match_stats.empty:
        return 8.0, 0.0

    if last_n == 3:
        n = len(match_stats)
        if n == 3:
            weights = np.array([0.2, 0.3, 0.5])
        elif n == 2:
            weights = np.array([0.375, 0.625])
        else:
            weights = np.array([1.0])
            
        w_runs = np.sum(match_stats['runs'] * weights)
        w_balls = np.sum(match_stats['balls'] * weights)
        w_wickets = np.sum(match_stats['wickets'] * weights)
        
        w_overs = w_balls / 6
        economy = (w_runs / w_overs) if w_overs > 0 else 8.0
        avg_wickets = w_wickets
    else:
        overs_bowled = match_stats['balls'].sum() / 6
        economy = (match_stats['runs'].sum() / overs_bowled) if overs_bowled > 0 else 8.0
        avg_wickets = match_stats['wickets'].mean()

    return round(economy, 3), round(avg_wickets, 3)


# ── Core function: Compute team batting & bowling features ────────────────────

def compute_team_player_features(team, match_players_batting, match_players_bowling,
                                  prior_match_ids, deliveries, prefix, season_prior_match_ids=None):
    """
    For a given team in a match, aggregate top player stats using only prior match data.
    """
    features = {}

    prior_deliveries = deliveries[deliveries['match_id'].isin(prior_match_ids)]

    # ── BATTING FEATURES ──────────────────────────────────────────
    batter_all_avgs, batter_all_srs = [], []
    batter_recent_avgs, batter_recent_srs = [], []

    # Use top 5 batters (by bat position or appearance order in this match)
    top_batters = match_players_batting[:5]

    for player in top_batters:
        avg, sr = get_player_batting_stats(prior_deliveries, player)
        batter_all_avgs.append(avg)
        batter_all_srs.append(sr)

        # Recent stats: restricted to the current season only!
        r_avg, r_sr = get_player_batting_stats(prior_deliveries, player, match_id_list=season_prior_match_ids, last_n=3)
        batter_recent_avgs.append(r_avg)
        batter_recent_srs.append(r_sr)

    features[f'{prefix}_avg_batter_avg'] = np.mean(batter_all_avgs) if batter_all_avgs else 15.0
    features[f'{prefix}_avg_batter_sr'] = np.mean(batter_all_srs) if batter_all_srs else 120.0
    features[f'{prefix}_avg_batter_recent_avg'] = np.mean(batter_recent_avgs) if batter_recent_avgs else 15.0
    features[f'{prefix}_avg_batter_recent_sr'] = np.mean(batter_recent_srs) if batter_recent_srs else 120.0

    # ── BOWLING FEATURES ──────────────────────────────────────────
    bowler_all_econs, bowler_all_wkts = [], []
    bowler_recent_econs = []

    top_bowlers = match_players_bowling[:4]

    for player in top_bowlers:
        econ, wkts = get_player_bowling_stats(prior_deliveries, player)
        bowler_all_econs.append(econ)
        bowler_all_wkts.append(wkts)

        # Recent stats: restricted to the current season only!
        r_econ, _ = get_player_bowling_stats(prior_deliveries, player, match_id_list=season_prior_match_ids, last_n=3)
        bowler_recent_econs.append(r_econ)

    features[f'{prefix}_avg_bowler_economy'] = np.mean(bowler_all_econs) if bowler_all_econs else 8.0
    features[f'{prefix}_avg_bowler_wickets'] = np.mean(bowler_all_wkts) if bowler_all_wkts else 1.0
    features[f'{prefix}_avg_bowler_recent_eco'] = np.mean(bowler_recent_econs) if bowler_recent_econs else 8.0

    return features


# ── Main: Engineer all features including player stats ────────────────────────

def engineer_features_v2(matches_path, deliveries_path, output_path):
    """
    Full feature engineering pipeline (match-level + player-level).
    Reads matches_clean.csv and deliveries_clean.csv.
    Outputs matches_featured_v2.csv with all features.
    """
    print("Loading datasets...")
    matches = pd.read_csv(matches_path)
    deliveries = pd.read_csv(deliveries_path, low_memory=False)

    matches['date'] = pd.to_datetime(matches['date'])
    matches = matches.sort_values('date').reset_index(drop=True)

    # Pre-build match → (batting players, bowling players) lookup per team
    # Group deliveries by match + innings + team
    print("Pre-computing player rosters per match...")
    
    # Batting roster: unique batters per match per team (in bat order)
    batting_rosters = (
        deliveries.groupby(['match_id', 'batting_team'])['batter']
        .apply(lambda x: list(dict.fromkeys(x.tolist())))  # unique ordered
        .to_dict()
    )
    # Bowling roster: unique bowlers per match per team
    bowling_rosters = (
        deliveries.groupby(['match_id', 'bowling_team'])['bowler']
        .apply(lambda x: list(dict.fromkeys(x.tolist())))
        .to_dict()
    )

    print(f"Processing {len(matches)} matches with player features...")
    print("(This may take 2-4 minutes due to per-player stats computation)")

    all_records = []

    for idx, row in matches.iterrows():
        if idx % 100 == 0:
            print(f"  Processing match {idx+1}/{len(matches)}...")

        match_id = row['match_id']
        team1 = row['team1']
        team2 = row['team2']
        venue = row['venue']
        toss_winner = row['toss_winner']
        toss_decision = row['toss_decision']

        # History: all matches strictly before this one by index (date-sorted)
        prior_matches = matches.iloc[:idx]
        prior_match_ids = set(prior_matches['match_id'].tolist())

        # ── Match-level features (recomputed from scratch for leak safety) ──
        HOME_VENUES = {
            'Kolkata Knight Riders': ['Eden Gardens'],
            'Chennai Super Kings': ['MA Chidambaram Stadium'],
            'Rajasthan Royals': ['Sawai Mansingh Stadium', 'Barsapara Cricket Stadium'],
            'Mumbai Indians': ['Wankhede Stadium', 'Brabourne Stadium', 'Dr DY Patil Sports Academy'],
            'Deccan Chargers': ['Rajiv Gandhi International Stadium', 'Barabati Stadium'],
            'Punjab Kings': [
                'Punjab Cricket Association Stadium', 
                'Maharaja Yadavindra Singh International Cricket Stadium', 
                'Himachal Pradesh Cricket Association Stadium',
                'Punjab Cricket Association IS Bindra Stadium'
            ],
            'Royal Challengers Bengaluru': ['M Chinnaswamy Stadium'],
            'Delhi Capitals': ['Arun Jaitley Stadium'],
            'Kochi Tuskers Kerala': ['Nehru Stadium'],
            'Pune Warriors': ['Maharashtra Cricket Association Stadium'],
            'Sunrisers Hyderabad': ['Rajiv Gandhi International Stadium'],
            'Rising Pune Supergiants': ['Maharashtra Cricket Association Stadium'],
            'Gujarat Lions': ['Saurashtra Cricket Association Stadium', 'Green Park'],
            'Lucknow Super Giants': ['Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium'],
            'Gujarat Titans': ['Narendra Modi Stadium'],
        }

        def get_is_home(team, v):
            if team not in HOME_VENUES:
                return 0
            for hg in HOME_VENUES[team]:
                if hg in v or v in hg:
                    return 1
            return 0

        target_date = pd.to_datetime(row['date'])

        def win_rate_decay(team, hist, t_date, lambda_decay=0.0005):
            tm = hist[(hist['team1'] == team) | (hist['team2'] == team)]
            if len(tm) == 0:
                return 0.5
            days_diff = (t_date - pd.to_datetime(tm['date'])).dt.days
            weights = np.exp(-lambda_decay * days_diff)
            wins = (tm['winner'] == team).astype(int)
            return np.sum(weights * wins) / np.sum(weights)

        def recent_form_decay(team, hist, t_date, current_season, n=5, lambda_decay=0.004):
            season_hist = hist[hist['season'] == current_season]
            tm = season_hist[(season_hist['team1'] == team) | (season_hist['team2'] == team)].tail(n)
            if len(tm) == 0:
                return 0.5
            days_diff = (t_date - pd.to_datetime(tm['date'])).dt.days
            weights = np.exp(-lambda_decay * days_diff)
            wins = (tm['winner'] == team).astype(int)
            return np.sum(weights * wins) / np.sum(weights)

        def h2h_decay(t1, t2, hist, t_date, lambda_decay=0.0005):
            m = hist[((hist['team1'] == t1) & (hist['team2'] == t2)) |
                     ((hist['team1'] == t2) & (hist['team2'] == t1))]
            if len(m) == 0:
                return 0.5
            days_diff = (t_date - pd.to_datetime(m['date'])).dt.days
            weights = np.exp(-lambda_decay * days_diff)
            wins = (m['winner'] == t1).astype(int)
            return np.sum(weights * wins) / np.sum(weights)

        def venue_wr_decay(team, v, hist, t_date, lambda_decay=0.0005):
            m = hist[(hist['venue'] == v) &
                     ((hist['team1'] == team) | (hist['team2'] == team))]
            if len(m) == 0:
                return 0.5
            days_diff = (t_date - pd.to_datetime(m['date'])).dt.days
            weights = np.exp(-lambda_decay * days_diff)
            wins = (m['winner'] == team).astype(int)
            return np.sum(weights * wins) / np.sum(weights)

        def toss_adv_decay(v, hist, t_date, lambda_decay=0.0005):
            m = hist[hist['venue'] == v]
            if len(m) == 0:
                return 0.5
            days_diff = (t_date - pd.to_datetime(m['date'])).dt.days
            weights = np.exp(-lambda_decay * days_diff)
            toss_wins = (m['toss_winner'] == m['winner']).astype(int)
            return np.sum(weights * toss_wins) / np.sum(weights)

        def season_form_rating(team, hist, current_season, hist_wr, alpha=3.0):
            season_hist = hist[hist['season'] == current_season]
            tm = season_hist[(season_hist['team1'] == team) | (season_hist['team2'] == team)]
            if len(tm) == 0:
                return hist_wr
            wins = (tm['winner'] == team).sum()
            return (wins + alpha * hist_wr) / (len(tm) + alpha)

        t1_wr = win_rate_decay(team1, prior_matches, target_date)
        t2_wr = win_rate_decay(team2, prior_matches, target_date)
        t1_rf = recent_form_decay(team1, prior_matches, target_date, row['season'])
        t2_rf = recent_form_decay(team2, prior_matches, target_date, row['season'])
        h2h_wr = h2h_decay(team1, team2, prior_matches, target_date)
        t1_vwr = venue_wr_decay(team1, venue, prior_matches, target_date)
        t2_vwr = venue_wr_decay(team2, venue, prior_matches, target_date)
        tva = toss_adv_decay(venue, prior_matches, target_date)
        
        t1_is_home = get_is_home(team1, venue)
        t2_is_home = get_is_home(team2, venue)
        
        t1_season_form = season_form_rating(team1, prior_matches, row['season'], t1_wr)
        t2_season_form = season_form_rating(team2, prior_matches, row['season'], t2_wr)
        
        t1_str = 0.40 * t1_wr + 0.35 * t1_rf + 0.25 * t1_vwr
        t2_str = 0.40 * t2_wr + 0.35 * t2_rf + 0.25 * t2_vwr
        toss_is_t1 = 1 if toss_winner == team1 else 0
        toss_dec = 1 if toss_decision == 'bat' else 0

        # ── Player-level features ──────────────────────────────────────────
        # Who actually played for each team in THIS match
        t1_batters = batting_rosters.get((match_id, team1), [])
        t2_batters = batting_rosters.get((match_id, team2), [])
        # Bowling team = the OTHER team's bowlers
        t1_bowlers = bowling_rosters.get((match_id, team2), [])  # team1 fields against team2 bowling
        t2_bowlers = bowling_rosters.get((match_id, team1), [])

        season_prior_match_ids = set(matches[(matches['date'] < target_date) & (matches['season'] == row['season'])]['match_id'])

        t1_player_feats = compute_team_player_features(
            team1, t1_batters, t1_bowlers, prior_match_ids, deliveries, 'team1', season_prior_match_ids
        )
        t2_player_feats = compute_team_player_features(
            team2, t2_batters, t2_bowlers, prior_match_ids, deliveries, 'team2', season_prior_match_ids
        )

        record = {
            'match_id': match_id,
            'date': row['date'],
            'season': row['season'],
            'year': row['year'],
            'team1': team1,
            'team2': team2,
            'venue': venue,
            'city': row['city'],
            'toss_winner': toss_winner,
            'toss_decision': toss_decision,
            'winner': row['winner'],
            # Match-level features
            'team1_win_rate': t1_wr,
            'team2_win_rate': t2_wr,
            'team1_recent_form': t1_rf,
            'team2_recent_form': t2_rf,
            'h2h_team1_win_rate': h2h_wr,
            'team1_venue_win_rate': t1_vwr,
            'team2_venue_win_rate': t2_vwr,
            'toss_venue_advantage': tva,
            'team1_strength': t1_str,
            'team2_strength': t2_str,
            'toss_winner_is_team1': toss_is_t1,
            'toss_decision_encoded': toss_dec,
            'team1_is_home': t1_is_home,
            'team2_is_home': t2_is_home,
            'team1_season_form': t1_season_form,
            'team2_season_form': t2_season_form,
            'target': 1 if row['winner'] == team1 else 0,
        }
        record.update(t1_player_feats)
        record.update(t2_player_feats)
        all_records.append(record)

    df_out = pd.DataFrame(all_records)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_out.to_csv(output_path, index=False)

    print(f"\nFeatured v2 dataset saved: {output_path}")
    print(f"Shape: {df_out.shape}")
    player_cols = [c for c in df_out.columns if 'batter' in c or 'bowler' in c]
    print(f"\nPlayer feature columns ({len(player_cols)}):")
    for c in player_cols:
        print(f"  {c:40s}  mean={df_out[c].mean():.2f}")

    return df_out


if __name__ == '__main__':
    engineer_features_v2(
        matches_path='data/processed/matches_clean.csv',
        deliveries_path='data/processed/deliveries_clean.csv',
        output_path='data/processed/matches_featured_v2.csv'
    )
