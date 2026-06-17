"""
IPL 2026 Backtesting Script.

For each IPL 2026 match:
  - Feed actual match conditions (team1, team2, venue, toss_winner, toss_decision)
  - Get the predicted probability of the ACTUAL winner
  - Aggregate Mean Winner Probability Score

This is more meaningful than binary accuracy — it rewards confident correct
predictions and penalises confident wrong ones.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

import pandas as pd
from src.predict import IPLPredictor

def run_backtest():
    print("Loading predictor...")
    predictor = IPLPredictor()

    print("Loading IPL 2026 matches...")
    df = pd.read_csv('data/processed/matches_clean.csv')
    matches_2026 = df[df['year'] == 2026].reset_index(drop=True)
    print(f"Total 2026 matches: {len(matches_2026)}\n")

    results = []
    binary_correct = 0

    print(f"{'#':<4} {'Match':<50} {'Predicted':>22} {'Actual':>26} {'Winner Prob':>12} {'Correct':>8}")
    print("-" * 128)

    for i, row in matches_2026.iterrows():
        team1        = row['team1']
        team2        = row['team2']
        venue        = row['venue']
        toss_winner  = row['toss_winner']
        toss_decision= row['toss_decision']
        actual_winner= row['winner']

        # Run prediction
        pred = predictor.predict(team1, team2, venue, toss_winner, toss_decision, date=row['date'])

        predicted_winner = pred['winner']

        # Probability assigned to the ACTUAL winner
        if actual_winner == team1:
            winner_prob = pred['team1_prob']
        else:
            winner_prob = pred['team2_prob']

        is_correct = (predicted_winner == actual_winner)
        if is_correct:
            binary_correct += 1

        match_str   = f"{team1[:20]} vs {team2[:20]}"
        correct_str = "✅" if is_correct else "❌"

        print(f"{i+1:<4} {match_str:<50} {predicted_winner[:22]:>22} {actual_winner[:26]:>26} {winner_prob:>11.1f}% {correct_str:>8}")

        results.append({
            'match': f"{team1} vs {team2}",
            'venue': venue,
            'toss_winner': toss_winner,
            'toss_decision': toss_decision,
            'actual_winner': actual_winner,
            'predicted_winner': predicted_winner,
            'winner_prob': winner_prob,
            'correct': is_correct,
        })

    # ── Summary metrics ───────────────────────────────────────────
    total     = len(results)
    mean_prob = sum(r['winner_prob'] for r in results) / total
    bin_acc   = binary_correct / total * 100

    print("-" * 128)
    print(f"\n{'='*60}")
    print(f"  IPL 2026 BACKTEST RESULTS")
    print(f"{'='*60}")
    print(f"  Total Matches Tested         : {total}")
    print(f"  Binary Correct Predictions   : {binary_correct}/{total}  ({bin_acc:.1f}%)")
    print(f"  Mean Winner Probability Score: {mean_prob:.1f}%")
    print(f"    → This is the avg probability assigned to the actual winner")
    print(f"    → A random model scores 50%, perfect model scores 100%")
    print(f"{'='*60}")

    # ── Per-team breakdown ────────────────────────────────────────
    print("\nPer-Team Breakdown:")
    team_stats = {}
    for r in results:
        for team_key in ['actual_winner']:
            t = r[team_key]
            if t not in team_stats:
                team_stats[t] = {'wins': 0, 'predicted_correctly': 0}
            team_stats[t]['wins'] += 1
            if r['correct']:
                team_stats[t]['predicted_correctly'] += 1

    print(f"  {'Team':<32} {'Actual Wins':>12} {'Pred Correct':>14}")
    print(f"  {'-'*32} {'-'*12} {'-'*14}")
    for team, s in sorted(team_stats.items(), key=lambda x: -x[1]['wins']):
        print(f"  {team:<32} {s['wins']:>12} {s['predicted_correctly']:>14}")

    return mean_prob, bin_acc

if __name__ == '__main__':
    run_backtest()
