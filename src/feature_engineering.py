"""
Feature Engineering for IPL Match Prediction.

All features are computed using ONLY historical data prior to each match
to prevent data leakage. For a match on date D, we only use matches before D.
"""

import os
import pandas as pd
import numpy as np


def compute_win_rate(history_df, team):
    """Calculate win rate of a team from a dataframe of past matches."""
    if history_df.empty:
        return 0.5  # Default when no history exists
    team_matches = history_df[
        (history_df['team1'] == team) | (history_df['team2'] == team)
    ]
    if len(team_matches) == 0:
        return 0.5
    wins = (team_matches['winner'] == team).sum()
    return wins / len(team_matches)


def compute_recent_form(history_df, team, n=5):
    """Calculate win rate in the last N matches for a team."""
    if history_df.empty:
        return 0.5
    team_matches = history_df[
        (history_df['team1'] == team) | (history_df['team2'] == team)
    ].tail(n)
    if len(team_matches) == 0:
        return 0.5
    wins = (team_matches['winner'] == team).sum()
    return wins / len(team_matches)


def compute_h2h_win_rate(history_df, team1, team2):
    """Head-to-head win rate of team1 against team2."""
    if history_df.empty:
        return 0.5
    h2h = history_df[
        ((history_df['team1'] == team1) & (history_df['team2'] == team2)) |
        ((history_df['team1'] == team2) & (history_df['team2'] == team1))
    ]
    if len(h2h) == 0:
        return 0.5
    wins = (h2h['winner'] == team1).sum()
    return wins / len(h2h)


def compute_venue_win_rate(history_df, team, venue):
    """Win rate of a team at a specific venue."""
    if history_df.empty:
        return 0.5
    venue_matches = history_df[
        (history_df['venue'] == venue) &
        ((history_df['team1'] == team) | (history_df['team2'] == team))
    ]
    if len(venue_matches) == 0:
        return 0.5
    wins = (venue_matches['winner'] == team).sum()
    return wins / len(venue_matches)


def compute_toss_venue_advantage(history_df, venue):
    """
    How often does the toss winner also win the match at this venue?
    A value > 0.5 means winning the toss is advantageous here.
    """
    if history_df.empty:
        return 0.5
    venue_matches = history_df[history_df['venue'] == venue]
    if len(venue_matches) == 0:
        return 0.5
    toss_wins = (venue_matches['toss_winner'] == venue_matches['winner']).sum()
    return toss_wins / len(venue_matches)


def engineer_features(matches_path, output_path):
    """
    Main function: loads clean matches, sorts chronologically,
    and computes rolling features for each match.
    """
    print("Loading cleaned match data...")
    df = pd.read_csv(matches_path)

    # Sort by date so we can iterate chronologically
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    print(f"Total matches (sorted by date): {len(df)}")

    # Pre-allocate feature columns
    features = {
        'team1_win_rate': [],
        'team2_win_rate': [],
        'team1_recent_form': [],
        'team2_recent_form': [],
        'h2h_team1_win_rate': [],
        'team1_venue_win_rate': [],
        'team2_venue_win_rate': [],
        'toss_venue_advantage': [],
        'team1_strength': [],
        'team2_strength': [],
        'toss_winner_is_team1': [],
        'toss_decision_encoded': [],
    }

    for idx in range(len(df)):
        row = df.iloc[idx]
        team1 = row['team1']
        team2 = row['team2']
        venue = row['venue']
        toss_winner = row['toss_winner']
        toss_decision = row['toss_decision']

        # Historical data: all matches BEFORE this one (by index, already date-sorted)
        history = df.iloc[:idx]

        # ── Feature 1: Overall win rates ──
        t1_wr = compute_win_rate(history, team1)
        t2_wr = compute_win_rate(history, team2)
        features['team1_win_rate'].append(t1_wr)
        features['team2_win_rate'].append(t2_wr)

        # ── Feature 2: Recent form (last 5 matches) ──
        t1_rf = compute_recent_form(history, team1, n=5)
        t2_rf = compute_recent_form(history, team2, n=5)
        features['team1_recent_form'].append(t1_rf)
        features['team2_recent_form'].append(t2_rf)

        # ── Feature 3: Head-to-head ──
        h2h = compute_h2h_win_rate(history, team1, team2)
        features['h2h_team1_win_rate'].append(h2h)

        # ── Feature 4: Venue win rates ──
        t1_vwr = compute_venue_win_rate(history, team1, venue)
        t2_vwr = compute_venue_win_rate(history, team2, venue)
        features['team1_venue_win_rate'].append(t1_vwr)
        features['team2_venue_win_rate'].append(t2_vwr)

        # ── Feature 5: Toss venue advantage ──
        tva = compute_toss_venue_advantage(history, venue)
        features['toss_venue_advantage'].append(tva)

        # ── Feature 6: Composite team strength ──
        # Weighted combination: 40% overall + 35% recent form + 25% venue
        t1_strength = 0.40 * t1_wr + 0.35 * t1_rf + 0.25 * t1_vwr
        t2_strength = 0.40 * t2_wr + 0.35 * t2_rf + 0.25 * t2_vwr
        features['team1_strength'].append(t1_strength)
        features['team2_strength'].append(t2_strength)

        # ── Toss context features ──
        features['toss_winner_is_team1'].append(1 if toss_winner == team1 else 0)
        features['toss_decision_encoded'].append(1 if toss_decision == 'bat' else 0)

    # Add all features to the dataframe
    for col, values in features.items():
        df[col] = values

    # Create the binary target
    df['target'] = (df['winner'] == df['team1']).astype(int)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nFeatured dataset saved to: {output_path}")
    print(f"Shape: {df.shape}")
    print(f"\nNew feature columns:")
    for col in features.keys():
        print(f"  {col:30s}  mean={df[col].mean():.3f}  min={df[col].min():.3f}  max={df[col].max():.3f}")

    return df


if __name__ == '__main__':
    engineer_features(
        matches_path='data/processed/matches_clean.csv',
        output_path='data/processed/matches_featured.csv'
    )
