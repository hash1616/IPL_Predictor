# Detailed RCB Match & Feature Analysis (IPL 2026)

**Overall RCB Match Prediction Accuracy**: 7/16 (43.8%)

## 1. Feature Weights (XGBoost)

| Feature Name | Weight (%) | Description |
|---|---|---|
| `team2_avg_batter_avg` | 3.881% | |
| `team2_is_home` | 3.814% | |
| `toss_winner_is_team1` | 3.778% | |
| `team1_is_home` | 3.774% | |
| `team1_recent_form` | 3.446% | |
| `team2_avg_batter_recent_avg` | 3.425% | |
| `team1_avg_batter_sr` | 3.417% | |
| `toss_venue_advantage` | 3.311% | |
| `team2_venue_win_rate` | 3.301% | |
| `team1_avg_batter_recent_sr` | 3.234% | |
| `team2_avg_batter_sr` | 3.225% | |
| `team2_win_rate` | 3.202% | |
| `team2_avg_batter_recent_sr` | 3.145% | |
| `team2_avg_bowler_wickets` | 3.136% | |
| `team1` | 3.121% | |
| `team2_strength` | 3.113% | |
| `team1_avg_batter_avg` | 3.050% | |
| `team1_venue_win_rate` | 3.044% | |
| `team1_avg_bowler_recent_eco` | 3.011% | |
| `team2_recent_form` | 2.963% | |
| `team1_avg_batter_recent_avg` | 2.929% | |
| `team1_avg_bowler_wickets` | 2.904% | |
| `team2_avg_bowler_recent_eco` | 2.890% | |
| `team1_strength` | 2.870% | |
| `team2` | 2.850% | |
| `h2h_team1_win_rate` | 2.822% | |
| `team2_avg_bowler_economy` | 2.799% | |
| `team1_win_rate` | 2.755% | |
| `team2_season_form` | 2.754% | |
| `team1_avg_bowler_economy` | 2.707% | |
| `venue` | 2.691% | |
| `team1_season_form` | 2.637% | |
| `toss_decision_encoded` | 0.000% | |


## 2. Match-by-Match Breakdown of RCB in 2026

| Date | Opponent | Winner | Predicted | Correct? | RCB Win Prob | Opponent Win Prob | RCB Bat Avg | Opp Bat Avg | RCB Bow Eco | Opp Bow Eco |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-03-28 | Sunrisers Hyderabad | Royal Challengers Bengaluru | Royal Challengers Bengaluru | ✅ | 68.6% | 31.4% | 26.8 | 28.6 | 8.48 | 9.33 |
| 2026-04-05 | Chennai Super Kings | Royal Challengers Bengaluru | Royal Challengers Bengaluru | ✅ | 67.3% | 32.7% | 26.8 | 27.7 | 8.48 | 9.24 |
| 2026-04-10 | Rajasthan Royals | Rajasthan Royals | Rajasthan Royals | ✅ | 15.7% | 84.3% | 26.8 | 27.7 | 8.48 | 9.27 |
| 2026-04-12 | Mumbai Indians | Royal Challengers Bengaluru | Royal Challengers Bengaluru | ✅ | 65.1% | 34.9% | 26.8 | 27.3 | 8.48 | 8.71 |
| 2026-04-15 | Lucknow Super Giants | Royal Challengers Bengaluru | Royal Challengers Bengaluru | ✅ | 78.7% | 21.3% | 26.8 | 26.5 | 8.48 | 9.26 |
| 2026-04-18 | Delhi Capitals | Delhi Capitals | Royal Challengers Bengaluru | ❌ | 84.6% | 15.4% | 26.8 | 24.2 | 8.48 | 8.33 |
| 2026-04-24 | Gujarat Titans | Royal Challengers Bengaluru | Gujarat Titans | ❌ | 49.0% | 51.0% | 26.8 | 28.6 | 8.48 | 8.36 |
| 2026-04-27 | Delhi Capitals | Royal Challengers Bengaluru | Delhi Capitals | ❌ | 27.9% | 72.1% | 26.8 | 24.2 | 8.48 | 8.33 |
| 2026-04-30 | Gujarat Titans | Gujarat Titans | Gujarat Titans | ✅ | 44.1% | 55.9% | 26.8 | 28.6 | 8.48 | 8.36 |
| 2026-05-07 | Lucknow Super Giants | Lucknow Super Giants | Royal Challengers Bengaluru | ❌ | 59.8% | 40.2% | 26.8 | 26.5 | 8.48 | 9.26 |
| 2026-05-10 | Mumbai Indians | Royal Challengers Bengaluru | Mumbai Indians | ❌ | 47.3% | 52.7% | 26.8 | 27.3 | 8.48 | 8.71 |
| 2026-05-13 | Kolkata Knight Riders | Royal Challengers Bengaluru | Royal Challengers Bengaluru | ✅ | 57.2% | 42.8% | 26.8 | 27.1 | 8.48 | 8.18 |
| 2026-05-17 | Punjab Kings | Royal Challengers Bengaluru | Punjab Kings | ❌ | 41.8% | 58.2% | 26.8 | 27.5 | 8.48 | 9.13 |
| 2026-05-22 | Sunrisers Hyderabad | Sunrisers Hyderabad | Royal Challengers Bengaluru | ❌ | 76.0% | 24.0% | 26.8 | 28.6 | 8.48 | 9.33 |
| 2026-05-26 | Gujarat Titans | Royal Challengers Bengaluru | Gujarat Titans | ❌ | 36.9% | 63.1% | 26.8 | 28.6 | 8.48 | 8.36 |
| 2026-05-31 | Gujarat Titans | Royal Challengers Bengaluru | Gujarat Titans | ❌ | 36.1% | 63.9% | 26.8 | 28.6 | 8.48 | 8.36 |