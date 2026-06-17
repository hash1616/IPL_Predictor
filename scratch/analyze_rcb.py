import pickle
import pandas as pd
import numpy as np

def analyze_rcb():
    # Load model and features
    with open('models/best_model.pkl', 'rb') as f:
        model = f.load() if hasattr(f, 'load') else pickle.load(f)
    with open('models/feature_list.pkl', 'rb') as f:
        feature_list = pickle.load(f)
    with open('models/model_metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
        
    # Feature importances
    importances = model.feature_importances_
    feats_df = pd.DataFrame({
        'Feature': feature_list,
        'Weight_%': importances * 100
    }).sort_values(by='Weight_%', ascending=False)
    
    # Load processed match clean
    matches = pd.read_csv('data/processed/matches_clean.csv')
    matches['date'] = pd.to_datetime(matches['date'])
    
    # Filter 2026 matches involving RCB
    rcb_matches = matches[
        (matches['year'] == 2026) & 
        ((matches['team1'] == 'Royal Challengers Bengaluru') | (matches['team2'] == 'Royal Challengers Bengaluru'))
    ].copy()
    
    # Let's import predictor to run the predictions on these matches
    from src.predict import IPLPredictor
    predictor = IPLPredictor()
    
    records = []
    correct_count = 0
    total_matches = len(rcb_matches)
    
    for idx, row in rcb_matches.iterrows():
        t1, t2 = row['team1'], row['team2']
        venue = row['venue']
        toss_w = row['toss_winner']
        toss_d = row['toss_decision']
        winner = row['winner']
        date = row['date']
        
        pred = predictor.predict(t1, t2, venue, toss_w, toss_d, date=date)
        pred_winner = pred['winner']
        is_correct = (pred_winner == winner)
        if is_correct:
            correct_count += 1
            
        # Extract features for both teams
        hist = predictor.history_df[predictor.history_df['date'] < date]
        
        def wr_decay(t):
            tm = hist[(hist['team1'] == t) | (hist['team2'] == t)]
            if len(tm) == 0: return 0.5
            diff = (date - pd.to_datetime(tm['date'])).dt.days
            w = np.exp(-0.0005 * diff)
            return np.sum(w * (tm['winner'] == t).astype(int)) / np.sum(w)
            
        def rf_decay(t):
            tm = hist[(hist['team1'] == t) | (hist['team2'] == t)].tail(5)
            if len(tm) == 0: return 0.5
            diff = (date - pd.to_datetime(tm['date'])).dt.days
            w = np.exp(-0.004 * diff)
            return np.sum(w * (hist.loc[tm.index, 'winner'] == t).astype(int)) / np.sum(w)
            
        t1_wr, t2_wr = wr_decay(t1), wr_decay(t2)
        t1_rf, t2_rf = rf_decay(t1), rf_decay(t2)
        
        p_feats = predictor._compute_player_features(t1, 'team1')
        p_feats.update(predictor._compute_player_features(t2, 'team2'))
        
        if t1 == 'Royal Challengers Bengaluru':
            rcb_bat_avg = p_feats['team1_avg_batter_avg']
            opp_bat_avg = p_feats['team2_avg_batter_avg']
            rcb_bow_eco = p_feats['team1_avg_bowler_economy']
            opp_bow_eco = p_feats['team2_avg_bowler_economy']
            rcb_wr, opp_wr = t1_wr, t2_wr
            rcb_rf, opp_rf = t1_rf, t2_rf
            rcb_prob = pred['team1_prob']
            opp_prob = pred['team2_prob']
            opp_name = t2
        else:
            rcb_bat_avg = p_feats['team2_avg_batter_avg']
            opp_bat_avg = p_feats['team1_avg_batter_avg']
            rcb_bow_eco = p_feats['team2_avg_bowler_economy']
            opp_bow_eco = p_feats['team1_avg_bowler_economy']
            rcb_wr, opp_wr = t2_wr, t1_wr
            rcb_rf, opp_rf = t2_rf, t1_rf
            rcb_prob = pred['team2_prob']
            opp_prob = pred['team1_prob']
            opp_name = t1
            
        records.append({
            'date': date.strftime('%Y-%m-%d'),
            'opponent': opp_name,
            'winner': winner,
            'predicted': pred_winner,
            'correct': is_correct,
            'rcb_prob': rcb_prob,
            'opp_prob': opp_prob,
            'rcb_wr': rcb_wr,
            'opp_wr': opp_wr,
            'rcb_rf': rcb_rf,
            'opp_rf': opp_rf,
            'rcb_bat_avg': rcb_bat_avg,
            'opp_bat_avg': opp_bat_avg,
            'rcb_bow_eco': rcb_bow_eco,
            'opp_bow_eco': opp_bow_eco
        })
        
    records_df = pd.DataFrame(records)
    
    # Save a detailed markdown analysis file
    md_content = []
    md_content.append("# Detailed RCB Match & Feature Analysis (IPL 2026)\n")
    md_content.append(f"**Overall RCB Match Prediction Accuracy**: {correct_count}/{total_matches} ({correct_count/total_matches*100:.1f}%)\n")
    
    md_content.append("## 1. Feature Weights (XGBoost)\n")
    md_content.append("| Feature Name | Weight (%) | Description |")
    md_content.append("|---|---|---|")
    for _, r in feats_df.iterrows():
        md_content.append(f"| `{r['Feature']}` | {r['Weight_%']:.3f}% | |")
    md_content.append("\n")
    
    md_content.append("## 2. Match-by-Match Breakdown of RCB in 2026\n")
    md_content.append("| Date | Opponent | Winner | Predicted | Correct? | RCB Win Prob | Opponent Win Prob | RCB Bat Avg | Opp Bat Avg | RCB Bow Eco | Opp Bow Eco |")
    md_content.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in records_df.iterrows():
        chk = "✅" if r['correct'] else "❌"
        md_content.append(f"| {r['date']} | {r['opponent']} | {r['winner']} | {r['predicted']} | {chk} | {r['rcb_prob']:.1f}% | {r['opp_prob']:.1f}% | {r['rcb_bat_avg']:.1f} | {r['opp_bat_avg']:.1f} | {r['rcb_bow_eco']:.2f} | {r['opp_bow_eco']:.2f} |")
        
    # Write analysis to a markdown file
    with open('rcb_deep_analysis_results.md', 'w') as f:
        f.write("\n".join(md_content))
    print("Successfully wrote analysis to rcb_deep_analysis_results.md")

if __name__ == '__main__':
    analyze_rcb()
