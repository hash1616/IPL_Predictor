"""
Train Model Final — Compare Random Forest vs XGBoost.
Auto-selects the best model and saves it as the production model.
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from xgboost import XGBClassifier


def train_and_compare(featured_path, models_dir):
    # ── 1. Load Data ──────────────────────────────────────────────
    print("Loading featured match data...")
    df = pd.read_csv(featured_path)
    print(f"Total matches: {len(df)}")

    # ── 2. Define Features ────────────────────────────────────────
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

    categorical_features = ['team1', 'team2', 'venue']

    # Encode categoricals
    label_encoders = {}
    df_encoded = df.copy()
    for col in categorical_features:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        label_encoders[col] = le

    all_features = numeric_features + categorical_features
    df_encoded = df_encoded.dropna(subset=all_features + ['target'])

    X = df_encoded[all_features].values
    y = df_encoded['target'].values

    # ── 3. Train/Test Split ───────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}\n")

    # ── 4. Define Models ──────────────────────────────────────────
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1
        ),
        'XGBoost': XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            eval_metric='logloss',
            use_label_encoder=False
        ),
    }

    # ── 5. Train & Evaluate Both ──────────────────────────────────
    results = {}

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted')
        rec = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        # Cross-validation score (5-fold) for more robust comparison
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

        results[name] = {
            'model': model,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
        }

        print(f"  Test Accuracy:  {acc:.4f}")
        print(f"  CV Accuracy:    {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print()

    # ── 6. Comparison Table ───────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  MODEL COMPARISON")
    print(f"{'='*70}")
    print(f"  {'Model':<20s} {'Accuracy':>10s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s} {'CV Acc':>10s}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    for name, r in results.items():
        print(f"  {name:<20s} {r['accuracy']:>10.4f} {r['precision']:>10.4f} {r['recall']:>10.4f} {r['f1']:>10.4f} {r['cv_mean']:>10.4f}")

    print(f"{'='*70}")

    # ── 7. Auto-Select Best Model ─────────────────────────────────
    # Use CV accuracy as the selection metric (more robust than single test split)
    best_name = max(results, key=lambda k: results[k]['cv_mean'])
    best_result = results[best_name]
    best_model = best_result['model']

    print(f"\n  🏆 BEST MODEL: {best_name}")
    print(f"     CV Accuracy: {best_result['cv_mean']:.4f} ± {best_result['cv_std']:.4f}")
    print(f"     Test Accuracy: {best_result['accuracy']:.4f}")

    # Print detailed classification report for the best model
    y_pred_best = best_model.predict(X_test)
    print(f"\nDetailed Classification Report ({best_name}):")
    print(classification_report(y_test, y_pred_best, target_names=['Team 2 Wins', 'Team 1 Wins']))

    # Feature importances for best model
    print(f"Feature Importances ({best_name}):")
    importances = sorted(
        zip(all_features, best_model.feature_importances_),
        key=lambda x: -x[1]
    )
    for col, imp in importances:
        bar = '█' * int(imp * 50)
        print(f"  {col:30s} {imp:.4f}  {bar}")

    # ── 8. Save Production Model ──────────────────────────────────
    os.makedirs(models_dir, exist_ok=True)

    # Save as the "production" model
    model_path = os.path.join(models_dir, 'best_model.pkl')
    encoders_path = os.path.join(models_dir, 'label_encoders.pkl')
    feature_list_path = os.path.join(models_dir, 'feature_list.pkl')
    metadata_path = os.path.join(models_dir, 'model_metadata.pkl')

    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)
    print(f"\nBest model saved to: {model_path}")

    with open(encoders_path, 'wb') as f:
        pickle.dump(label_encoders, f)
    print(f"Label encoders saved to: {encoders_path}")

    with open(feature_list_path, 'wb') as f:
        pickle.dump(all_features, f)
    print(f"Feature list saved to: {feature_list_path}")

    metadata = {
        'model_name': best_name,
        'accuracy': best_result['accuracy'],
        'cv_accuracy': best_result['cv_mean'],
        'precision': best_result['precision'],
        'recall': best_result['recall'],
        'f1': best_result['f1'],
        'features': all_features,
        'numeric_features': numeric_features,
        'categorical_features': categorical_features,
    }
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"Model metadata saved to: {metadata_path}")

    print("\n✅ Phase 6 complete! Production model ready.")
    return best_model, label_encoders, metadata


if __name__ == '__main__':
    train_and_compare(
        featured_path='data/processed/matches_featured.csv',
        models_dir='models/'
    )
