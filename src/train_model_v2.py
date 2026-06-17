"""
Train Model V2 — Random Forest on engineered features.
Uses the enriched dataset from feature_engineering.py.
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report


def train_model_v2(featured_path, models_dir):
    # ── 1. Load Featured Data ─────────────────────────────────────
    print("Loading featured match data...")
    df = pd.read_csv(featured_path)
    print(f"Total matches: {len(df)}")

    # ── 2. Define Features ────────────────────────────────────────
    # Numeric engineered features (no leakage — computed from history only)
    numeric_features = [
        'team1_win_rate',
        'team2_win_rate',
        'team1_recent_form',
        'team2_recent_form',
        'h2h_team1_win_rate',
        'team1_venue_win_rate',
        'team2_venue_win_rate',
        'toss_venue_advantage',
        'team1_strength',
        'team2_strength',
        'toss_winner_is_team1',
        'toss_decision_encoded',
    ]

    # Categorical features we still want to include (encoded)
    categorical_features = ['team1', 'team2', 'venue']

    # Encode categoricals
    label_encoders = {}
    df_encoded = df.copy()
    for col in categorical_features:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        label_encoders[col] = le

    all_features = numeric_features + categorical_features

    # Drop rows with NaN in features or target
    df_encoded = df_encoded.dropna(subset=all_features + ['target'])
    print(f"Matches after cleanup: {len(df_encoded)}")

    X = df_encoded[all_features].values
    y = df_encoded['target'].values

    print(f"Feature matrix shape: {X.shape}")
    print(f"Target distribution: team1 wins = {y.sum()}, team2 wins = {len(y) - y.sum()}")

    # ── 3. Train/Test Split ───────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain set: {X_train.shape[0]} | Test set: {X_test.shape[0]}")

    # ── 4. Train Random Forest V2 ─────────────────────────────────
    print("\nTraining Random Forest V2 (200 estimators, enriched features)...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # ── 5. Evaluate ───────────────────────────────────────────────
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n{'='*55}")
    print(f"  MODEL V2 EVALUATION RESULTS")
    print(f"{'='*55}")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"  Baseline was: 50.0%  →  Improvement: +{(accuracy - 0.50)*100:.1f}%")
    print(f"{'='*55}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Team 2 Wins', 'Team 1 Wins']))

    # Feature importances
    print("Feature Importances (Top → Bottom):")
    importances = sorted(
        zip(all_features, model.feature_importances_),
        key=lambda x: -x[1]
    )
    for col, imp in importances:
        bar = '█' * int(imp * 50)
        print(f"  {col:30s} {imp:.4f}  {bar}")

    # ── 6. Save Model & Encoders ──────────────────────────────────
    os.makedirs(models_dir, exist_ok=True)

    model_path = os.path.join(models_dir, 'random_forest_v2.pkl')
    encoders_path = os.path.join(models_dir, 'label_encoders_v2.pkl')
    feature_list_path = os.path.join(models_dir, 'feature_list_v2.pkl')

    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"\nModel saved to: {model_path}")

    with open(encoders_path, 'wb') as f:
        pickle.dump(label_encoders, f)
    print(f"Label encoders saved to: {encoders_path}")

    with open(feature_list_path, 'wb') as f:
        pickle.dump(all_features, f)
    print(f"Feature list saved to: {feature_list_path}")

    print("\nPhase 5 complete!")
    return model, label_encoders, accuracy


if __name__ == '__main__':
    train_model_v2(
        featured_path='data/processed/matches_featured.csv',
        models_dir='models/'
    )
