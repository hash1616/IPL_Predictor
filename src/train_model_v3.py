"""
Train Model V3 — Random Forest + XGBoost on full feature set.
Uses match-level features + player batting & bowling stats.
Auto-selects best model and saves as production model.
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score, classification_report)
from xgboost import XGBClassifier
from catboost import CatBoostClassifier


def train_model_v3(featured_path, models_dir):
    # ── 1. Load Data ──────────────────────────────────────────────
    print("Loading featured v2 match data...")
    df = pd.read_csv(featured_path)
    print(f"Total matches: {len(df)}")

    # ── 2. Feature Columns ────────────────────────────────────────
    numeric_features = [
        # Match-level
        'team1_win_rate', 'team2_win_rate',
        'team1_recent_form', 'team2_recent_form',
        'h2h_team1_win_rate',
        'team1_venue_win_rate', 'team2_venue_win_rate',
        'toss_venue_advantage',
        'team1_strength', 'team2_strength',
        'toss_winner_is_team1', 'toss_decision_encoded',
        'team1_is_home', 'team2_is_home',
        'team1_season_form', 'team2_season_form',
        # Player batting features
        'team1_avg_batter_avg', 'team1_avg_batter_sr',
        'team1_avg_batter_recent_avg', 'team1_avg_batter_recent_sr',
        'team2_avg_batter_avg', 'team2_avg_batter_sr',
        'team2_avg_batter_recent_avg', 'team2_avg_batter_recent_sr',
        # Player bowling features
        'team1_avg_bowler_economy', 'team1_avg_bowler_wickets',
        'team1_avg_bowler_recent_eco',
        'team2_avg_bowler_economy', 'team2_avg_bowler_wickets',
        'team2_avg_bowler_recent_eco',
    ]

    categorical_features = ['team1', 'team2', 'venue']

    # Encode categoricals
    label_encoders = {}
    df_enc = df.copy()
    for col in categorical_features:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_enc[col].astype(str))
        label_encoders[col] = le

    all_features = numeric_features + categorical_features

    # Drop rows with NaN
    df_enc = df_enc.dropna(subset=all_features + ['target'])
    print(f"Matches after cleanup: {len(df_enc)}")

    X = df_enc[all_features].values
    y = df_enc['target'].values
    print(f"Feature matrix: {X.shape}  |  Team1 wins: {y.sum()}, Team2 wins: {len(y)-y.sum()}")

    # ── 3. Train/Test Split ───────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {X_train.shape[0]}  |  Test: {X_test.shape[0]}\n")

    # ── 4. Models ─────────────────────────────────────────────────
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=4,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        ),
        'XGBoost': XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.05,
            reg_lambda=1.0,
            min_child_weight=3,
            random_state=42,
            eval_metric='logloss',
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.8,
            random_state=42,
        ),
        'CatBoost': CatBoostClassifier(
            iterations=300,
            depth=6,
            learning_rate=0.06,
            verbose=0,
            random_state=42
        ),
    }

    # ── 5. Train & Evaluate ───────────────────────────────────────
    results = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc   = accuracy_score(y_test, y_pred)
        prec  = precision_score(y_test, y_pred, average='weighted')
        rec   = recall_score(y_test, y_pred, average='weighted')
        f1    = f1_score(y_test, y_pred, average='weighted')
        cv_sc = cross_val_score(model, X, y, cv=cv, scoring='accuracy')

        results[name] = dict(model=model, accuracy=acc, precision=prec,
                              recall=rec, f1=f1,
                              cv_mean=cv_sc.mean(), cv_std=cv_sc.std())
        print(f"  Test Acc: {acc:.4f}  |  CV: {cv_sc.mean():.4f} ± {cv_sc.std():.4f}\n")

    # ── 6. Comparison Table ───────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  MODEL COMPARISON  (V3 — with player features)")
    print(f"{'='*72}")
    print(f"  {'Model':<22} {'Test Acc':>9} {'Precision':>10} {'Recall':>8} {'F1':>8} {'CV Acc':>9}")
    print(f"  {'-'*22} {'-'*9} {'-'*10} {'-'*8} {'-'*8} {'-'*9}")
    for name, r in results.items():
        print(f"  {name:<22} {r['accuracy']:>9.4f} {r['precision']:>10.4f} "
              f"{r['recall']:>8.4f} {r['f1']:>8.4f} {r['cv_mean']:>9.4f}")
    print(f"{'='*72}")

    # ── 7. Auto-select best model ─────────────────────────────────
    best_name = max(results, key=lambda k: results[k]['cv_mean'])
    best = results[best_name]
    best_model = best['model']

    print(f"\n  🏆 BEST MODEL: {best_name}")
    print(f"     CV Accuracy : {best['cv_mean']:.4f} ± {best['cv_std']:.4f}")
    print(f"     Test Accuracy: {best['accuracy']:.4f}")

    y_pred_best = best_model.predict(X_test)
    print(f"\nClassification Report ({best_name}):")
    print(classification_report(y_test, y_pred_best,
                                 target_names=['Team 2 Wins', 'Team 1 Wins']))

    # Feature importances
    print("Feature Importances (Top 15):")
    importances = sorted(zip(all_features, best_model.feature_importances_),
                         key=lambda x: -x[1])[:15]
    for col, imp in importances:
        bar = '█' * int(imp * 60)
        print(f"  {col:38s} {imp:.4f}  {bar}")

    # ── 8. Save production model ──────────────────────────────────
    os.makedirs(models_dir, exist_ok=True)

    paths = {
        'model':     os.path.join(models_dir, 'best_model.pkl'),
        'encoders':  os.path.join(models_dir, 'label_encoders.pkl'),
        'features':  os.path.join(models_dir, 'feature_list.pkl'),
        'metadata':  os.path.join(models_dir, 'model_metadata.pkl'),
    }

    metadata = {
        'model_name':          best_name,
        'accuracy':            best['accuracy'],
        'cv_accuracy':         best['cv_mean'],
        'precision':           best['precision'],
        'recall':              best['recall'],
        'f1':                  best['f1'],
        'features':            all_features,
        'numeric_features':    numeric_features,
        'categorical_features': categorical_features,
        'has_player_features': True,
    }

    for key, path in paths.items():
        obj = (best_model if key == 'model'
               else label_encoders if key == 'encoders'
               else all_features if key == 'features'
               else metadata)
        with open(path, 'wb') as f:
            pickle.dump(obj, f)
        print(f"Saved {key} → {path}")

    print("\n✅ Model V3 training complete! Production model updated.")
    return best_model, label_encoders, metadata


if __name__ == '__main__':
    train_model_v3(
        featured_path='data/processed/matches_featured_v2.csv',
        models_dir='models/'
    )
