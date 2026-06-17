"""
Prediction Pipeline V2 for IPL Match Winner.
Handles both match-level and player-level features.

Usage:
    from src.predict import IPLPredictor
    predictor = IPLPredictor()
    result = predictor.predict("Mumbai Indians", "Chennai Super Kings",
                                "Wankhede Stadium", "Mumbai Indians", "bat")
"""

import os
import pickle
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class IPLPredictor:
    """End-to-end prediction pipeline for IPL match outcomes."""

    def __init__(self):
        models_dir = os.path.join(BASE_DIR, 'models')
        data_dir   = os.path.join(BASE_DIR, 'data', 'processed')

        with open(os.path.join(models_dir, 'best_model.pkl'),       'rb') as f:
            self.model = pickle.load(f)
        with open(os.path.join(models_dir, 'label_encoders.pkl'),   'rb') as f:
            self.label_encoders = pickle.load(f)
        with open(os.path.join(models_dir, 'feature_list.pkl'),     'rb') as f:
            self.feature_list = pickle.load(f)
        with open(os.path.join(models_dir, 'model_metadata.pkl'),   'rb') as f:
            self.metadata = pickle.load(f)

        # Match-level history
        self.history_df = pd.read_csv(os.path.join(data_dir, 'matches_clean.csv'))
        self.history_df['date'] = pd.to_datetime(self.history_df['date'])
        self.history_df = self.history_df.sort_values('date').reset_index(drop=True)

        # Deliveries for player stats
        self.deliveries = pd.read_csv(
            os.path.join(data_dir, 'deliveries_clean.csv'), low_memory=False
        )

        self.teams  = sorted(self.label_encoders['team1'].classes_.tolist())
        self.venues = sorted(self.label_encoders['venue'].classes_.tolist())
        self.has_player_features = self.metadata.get('has_player_features', False)

        print(f"IPLPredictor loaded. Model: {self.metadata['model_name']}  "
              f"CV Accuracy: {self.metadata['cv_accuracy']:.1%}  "
              f"Player features: {self.has_player_features}")

    # ── Match-level helpers ───────────────────────────────────────────────────

    def _win_rate(self, team):
        tm = self.history_df[(self.history_df['team1'] == team) | (self.history_df['team2'] == team)]
        return (tm['winner'] == team).sum() / len(tm) if len(tm) > 0 else 0.5

    def _recent_form(self, team, n=5):
        tm = self.history_df[(self.history_df['team1'] == team) | (self.history_df['team2'] == team)].tail(n)
        return (tm['winner'] == team).sum() / len(tm) if len(tm) > 0 else 0.5

    def _h2h(self, team1, team2):
        m = self.history_df[
            ((self.history_df['team1'] == team1) & (self.history_df['team2'] == team2)) |
            ((self.history_df['team1'] == team2) & (self.history_df['team2'] == team1))
        ]
        if len(m) == 0:
            return 0.5, 0, 0, 0
        t1w = int((m['winner'] == team1).sum())
        t2w = int((m['winner'] == team2).sum())
        return t1w / len(m), len(m), t1w, t2w

    def _venue_wr(self, team, venue):
        m = self.history_df[
            (self.history_df['venue'] == venue) &
            ((self.history_df['team1'] == team) | (self.history_df['team2'] == team))
        ]
        return (m['winner'] == team).sum() / len(m) if len(m) > 0 else 0.5

    def _toss_venue_adv(self, venue):
        m = self.history_df[self.history_df['venue'] == venue]
        return (m['toss_winner'] == m['winner']).sum() / len(m) if len(m) > 0 else 0.5

    # ── Player-level helpers ──────────────────────────────────────────────────

    def _team_recent_batters(self, team, prior_match_ids, n_matches=5):
        """Get the most recent N matches' worth of batters for a team using only prior matches."""
        team_matches = self.history_df[
            self.history_df['match_id'].isin(prior_match_ids) &
            ((self.history_df['team1'] == team) | (self.history_df['team2'] == team))
        ].tail(n_matches)
        team_match_ids = team_matches['match_id'].tolist()
        batters = (
            self.deliveries[
                (self.deliveries['match_id'].isin(team_match_ids)) &
                (self.deliveries['batting_team'] == team)
            ]['batter'].value_counts().head(5).index.tolist()
        )
        return batters

    def _team_recent_bowlers(self, team, prior_match_ids, n_matches=5):
        """Get the most active bowlers for a team in last N matches using only prior matches."""
        team_matches = self.history_df[
            self.history_df['match_id'].isin(prior_match_ids) &
            ((self.history_df['team1'] == team) | (self.history_df['team2'] == team))
        ].tail(n_matches)
        team_match_ids = team_matches['match_id'].tolist()
        bowlers = (
            self.deliveries[
                (self.deliveries['match_id'].isin(team_match_ids)) &
                (self.deliveries['bowling_team'] == team)
            ]['bowler'].value_counts().head(4).index.tolist()
        )
        return bowlers

    def _batter_stats(self, player, allowed_match_ids, last_n=None):
        """Returns (avg_runs, avg_sr) for a batter, restricted to allowed_match_ids."""
        d = self.deliveries[
            (self.deliveries['batter'] == player) &
            (self.deliveries['match_id'].isin(allowed_match_ids))
        ]
        if d.empty:
            return 15.0, 120.0
        stats = d.groupby(['match_id', 'innings']).agg(
            runs=('runs_batter', 'sum'), balls=('valid_ball', 'sum')
        )
        if last_n:
            stats = stats.tail(last_n)
        if stats.empty:
            return 15.0, 120.0
        
        if last_n == 3:
            n = len(stats)
            if n == 3:
                weights = np.array([0.2, 0.3, 0.5])
            elif n == 2:
                weights = np.array([0.375, 0.625])
            else:
                weights = np.array([1.0])
            w_runs = np.sum(stats['runs'] * weights)
            w_balls = np.sum(stats['balls'] * weights)
            sr = (w_runs / w_balls * 100) if w_balls > 0 else 120.0
            avg_r = np.sum(stats['runs'] * weights)
        else:
            avg_r = stats['runs'].mean()
            sr    = (stats['runs'].sum() / stats['balls'].sum() * 100) if stats['balls'].sum() > 0 else 120.0
            
        return round(avg_r, 2), round(sr, 2)

    def _bowler_stats(self, player, allowed_match_ids, last_n=None):
        """Returns (economy, avg_wickets) for a bowler, restricted to allowed_match_ids."""
        d = self.deliveries[
            (self.deliveries['bowler'] == player) &
            (self.deliveries['match_id'].isin(allowed_match_ids))
        ]
        if d.empty:
            return 8.0, 1.0
        stats = d.groupby('match_id').agg(
            runs=('runs_bowler', 'sum'), balls=('valid_ball', 'sum'),
            wkts=('bowler_wicket', 'sum')
        )
        if last_n:
            stats = stats.tail(last_n)
        if stats.empty:
            return 8.0, 1.0
            
        if last_n == 3:
            n = len(stats)
            if n == 3:
                weights = np.array([0.2, 0.3, 0.5])
            elif n == 2:
                weights = np.array([0.375, 0.625])
            else:
                weights = np.array([1.0])
            w_runs = np.sum(stats['runs'] * weights)
            w_balls = np.sum(stats['balls'] * weights)
            w_wkts = np.sum(stats['wkts'] * weights)
            overs = w_balls / 6
            econ = w_runs / overs if overs > 0 else 8.0
            avg_wk = w_wkts
        else:
            overs  = stats['balls'].sum() / 6
            econ   = stats['runs'].sum() / overs if overs > 0 else 8.0
            avg_wk = stats['wkts'].mean()
            
        return round(econ, 2), round(avg_wk, 2)

    def _compute_player_features(self, team, prefix, prior_match_ids, season_prior_match_ids):
        """Aggregate batting & bowling features for a team, restricting recent stats to current season."""
        batters = self._team_recent_batters(team, prior_match_ids)
        bowlers = self._team_recent_bowlers(team, prior_match_ids)

        b_avgs, b_srs, b_ravgs, b_rsrs = [], [], [], []
        for p in batters:
            a, s   = self._batter_stats(p, prior_match_ids)
            ra, rs = self._batter_stats(p, season_prior_match_ids, last_n=3)
            b_avgs.append(a); b_srs.append(s)
            b_ravgs.append(ra); b_rsrs.append(rs)

        e_all, wk_all, e_rec = [], [], []
        for p in bowlers:
            e, w  = self._bowler_stats(p, prior_match_ids)
            re, _ = self._bowler_stats(p, season_prior_match_ids, last_n=3)
            e_all.append(e); wk_all.append(w); e_rec.append(re)

        def safe_mean(lst, default):
            return np.mean(lst) if lst else default

        return {
            f'{prefix}_avg_batter_avg':        safe_mean(b_avgs,  15.0),
            f'{prefix}_avg_batter_sr':         safe_mean(b_srs,  120.0),
            f'{prefix}_avg_batter_recent_avg': safe_mean(b_ravgs, 15.0),
            f'{prefix}_avg_batter_recent_sr':  safe_mean(b_rsrs, 120.0),
            f'{prefix}_avg_bowler_economy':    safe_mean(e_all,    8.0),
            f'{prefix}_avg_bowler_wickets':    safe_mean(wk_all,   1.0),
            f'{prefix}_avg_bowler_recent_eco': safe_mean(e_rec,    8.0),
        }

    # ── Public API ────────────────────────────────────────────────────────────

    def get_h2h_stats(self, team1, team2):
        m = self.history_df[
            ((self.history_df['team1'] == team1) & (self.history_df['team2'] == team2)) |
            ((self.history_df['team1'] == team2) & (self.history_df['team2'] == team1))
        ]
        total = len(m)
        if total == 0:
            return {'total': 0, 'team1_wins': 0, 'team2_wins': 0,
                    'last_5': [], 'team1_name': team1, 'team2_name': team2}
        t1w = int((m['winner'] == team1).sum())
        t2w = int((m['winner'] == team2).sum())
        last_5 = m.tail(5)[['date', 'team1', 'team2', 'winner', 'venue']].to_dict('records')
        return {'total': total, 'team1_wins': t1w, 'team2_wins': t2w,
                'last_5': last_5, 'team1_name': team1, 'team2_name': team2}

    def predict(self, team1, team2, venue, toss_winner, toss_decision, date=None):
        if date is None:
            target_date = pd.Timestamp.now()
        else:
            target_date = pd.to_datetime(date)
            
        # Filter history for matches strictly prior to target_date
        hist = self.history_df[self.history_df['date'] < target_date]

        # Deduce season
        match_season = str(target_date.year)
        # Check if the season is in our history (sometimes season is e.g. "2007/08")
        seasons = self.history_df['season'].unique()
        matched_season = match_season
        for s in seasons:
            if match_season in str(s):
                matched_season = s
                break

        # ── Match-level features ──────────────────────────────────
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

        def win_rate_decay(team, prior_hist, lambda_decay=0.0005):
            tm = prior_hist[(prior_hist['team1'] == team) | (prior_hist['team2'] == team)]
            if len(tm) == 0:
                return 0.5
            days_diff = (target_date - pd.to_datetime(tm['date'])).dt.days
            weights = np.exp(-lambda_decay * days_diff)
            wins = (tm['winner'] == team).astype(int)
            return np.sum(weights * wins) / np.sum(weights)

        def recent_form_decay(team, prior_hist, current_season, n=5, lambda_decay=0.004):
            # Restrict to current season matches only!
            season_hist = prior_hist[prior_hist['season'] == current_season]
            tm = season_hist[(season_hist['team1'] == team) | (season_hist['team2'] == team)].tail(n)
            if len(tm) == 0:
                return 0.5
            days_diff = (target_date - pd.to_datetime(tm['date'])).dt.days
            weights = np.exp(-lambda_decay * days_diff)
            wins = (prior_hist.loc[tm.index, 'winner'] == team).astype(int)
            return np.sum(weights * wins) / np.sum(weights)

        def h2h_decay(t1, t2, prior_hist, lambda_decay=0.0005):
            m = prior_hist[((prior_hist['team1'] == t1) & (prior_hist['team2'] == t2)) |
                           ((prior_hist['team1'] == t2) & (prior_hist['team2'] == t1))]
            if len(m) == 0:
                return 0.5
            days_diff = (target_date - pd.to_datetime(m['date'])).dt.days
            weights = np.exp(-lambda_decay * days_diff)
            wins = (m['winner'] == t1).astype(int)
            return np.sum(weights * wins) / np.sum(weights)

        def venue_wr_decay(team, v, prior_hist, lambda_decay=0.0005):
            m = prior_hist[(prior_hist['venue'] == v) &
                           ((prior_hist['team1'] == team) | (prior_hist['team2'] == team))]
            if len(m) == 0:
                return 0.5
            days_diff = (target_date - pd.to_datetime(m['date'])).dt.days
            weights = np.exp(-lambda_decay * days_diff)
            wins = (m['winner'] == team).astype(int)
            return np.sum(weights * wins) / np.sum(weights)

        def toss_adv_decay(v, prior_hist, lambda_decay=0.0005):
            m = prior_hist[prior_hist['venue'] == v]
            if len(m) == 0:
                return 0.5
            days_diff = (target_date - pd.to_datetime(m['date'])).dt.days
            weights = np.exp(-lambda_decay * days_diff)
            toss_wins = (m['toss_winner'] == m['winner']).astype(int)
            return np.sum(weights * toss_wins) / np.sum(weights)

        def season_form_rating(team, prior_hist, current_season, hist_wr, alpha=3.0):
            season_hist = prior_hist[prior_hist['season'] == current_season]
            tm = season_hist[(season_hist['team1'] == team) | (season_hist['team2'] == team)]
            if len(tm) == 0:
                return hist_wr
            wins = (tm['winner'] == team).sum()
            return (wins + alpha * hist_wr) / (len(tm) + alpha)

        t1_wr  = win_rate_decay(team1, hist)
        t2_wr  = win_rate_decay(team2, hist)
        t1_rf  = recent_form_decay(team1, hist, matched_season)
        t2_rf  = recent_form_decay(team2, hist, matched_season)
        h2h_wr = h2h_decay(team1, team2, hist)
        t1_vwr = venue_wr_decay(team1, venue, hist)
        t2_vwr = venue_wr_decay(team2, venue, hist)
        tva    = toss_adv_decay(venue, hist)
        
        t1_season_form = season_form_rating(team1, hist, matched_season, t1_wr)
        t2_season_form = season_form_rating(team2, hist, matched_season, t2_wr)
        
        t1_is_home = get_is_home(team1, venue)
        t2_is_home = get_is_home(team2, venue)

        t1_str = 0.40 * t1_wr + 0.35 * t1_rf + 0.25 * t1_vwr
        t2_str = 0.40 * t2_wr + 0.35 * t2_rf + 0.25 * t2_vwr
        t1_is_toss = 1 if toss_winner == team1 else 0
        toss_dec   = 1 if toss_decision == 'bat' else 0

        feature_vals = {
            'team1_win_rate': t1_wr, 'team2_win_rate': t2_wr,
            'team1_recent_form': t1_rf, 'team2_recent_form': t2_rf,
            'h2h_team1_win_rate': h2h_wr,
            'team1_venue_win_rate': t1_vwr, 'team2_venue_win_rate': t2_vwr,
            'toss_venue_advantage': tva,
            'team1_strength': t1_str, 'team2_strength': t2_str,
            'toss_winner_is_team1': t1_is_toss, 'toss_decision_encoded': toss_dec,
            'team1_is_home': t1_is_home, 'team2_is_home': t2_is_home,
            'team1_season_form': t1_season_form, 'team2_season_form': t2_season_form,
        }

        # ── Player-level features ─────────────────────────────────
        if self.has_player_features:
            prior_match_ids = set(hist['match_id'])
            season_prior_match_ids = set(hist[hist['season'] == matched_season]['match_id'])
            feature_vals.update(self._compute_player_features(team1, 'team1', prior_match_ids, season_prior_match_ids))
            feature_vals.update(self._compute_player_features(team2, 'team2', prior_match_ids, season_prior_match_ids))

        # ── Encode categoricals ───────────────────────────────────
        def safe_encode(col, val):
            le = self.label_encoders[col]
            return le.transform([val])[0] if val in le.classes_ else 0

        feature_vals['team1'] = safe_encode('team1', team1)
        feature_vals['team2'] = safe_encode('team2', team2)
        feature_vals['venue'] = safe_encode('venue', venue)

        # Build vector in exact model feature order
        vec = np.array([[feature_vals.get(f, 0.0) for f in self.feature_list]])

        prediction   = self.model.predict(vec)[0]
        probabilities = self.model.predict_proba(vec)[0]

        team1_prob = probabilities[1]
        team2_prob = probabilities[0]
        winner     = team1 if prediction == 1 else team2
        confidence = max(team1_prob, team2_prob)

        return {
            'winner':         winner,
            'team1':          team1,
            'team2':          team2,
            'team1_prob':     round(team1_prob * 100, 1),
            'team2_prob':     round(team2_prob * 100, 1),
            'confidence':     round(confidence * 100, 1),
            'model_name':     self.metadata['model_name'],
            'model_accuracy': round(self.metadata['cv_accuracy'] * 100, 1),
        }


if __name__ == '__main__':
    predictor = IPLPredictor()

    print("\n" + "=" * 60)
    print("  SAMPLE PREDICTION (V2 with Player Features)")
    print("=" * 60)

    result = predictor.predict(
        team1="Mumbai Indians",
        team2="Chennai Super Kings",
        venue="Wankhede Stadium",
        toss_winner="Mumbai Indians",
        toss_decision="field"
    )

    print(f"  {result['team1']} vs {result['team2']}")
    print(f"  🏆 Predicted Winner: {result['winner']}")
    print(f"  {result['team1']}: {result['team1_prob']}%")
    print(f"  {result['team2']}: {result['team2_prob']}%")
    print(f"  Confidence: {result['confidence']}%")
    print("=" * 60)
