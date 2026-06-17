import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

def train_random_forest(matches_path, models_dir):
    """
    Train a Random Forest model for IPL match winner prediction.
    
    Features: team1, team2, toss_winner, toss_decision, venue
    Target: winner (binary — 0 = team1 wins, 1 = team2 wins)
    """
    
    # ── 1. Load Data ──────────────────────────────────────────────
    print("Loading cleaned match data...")
    df = pd.read_csv(matches_path)
    print(f"Total matches loaded: {len(df)}")
    
    # ── 2. Select Features & Target ───────────────────────────────
    feature_cols = ['team1', 'team2', 'toss_winner', 'toss_decision', 'venue']
    
    # Drop rows with any missing values in our feature/target columns
    df = df.dropna(subset=feature_cols + ['winner'])
    print(f"Matches after dropping NAs: {len(df)}")
    
    # Create binary target: 1 if team1 wins, 0 if team2 wins
    df['target'] = (df['winner'] == df['team1']).astype(int)
    
    # ── 3. Encode Categorical Features ────────────────────────────
    label_encoders = {}
    df_encoded = df.copy()
    
    for col in feature_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        label_encoders[col] = le
        print(f"  Encoded '{col}': {len(le.classes_)} unique values")
    
    X = df_encoded[feature_cols].values
    y = df_encoded['target'].values
    
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Target distribution: team1 wins = {y.sum()}, team2 wins = {len(y) - y.sum()}")
    
    # ── 4. Train/Test Split ───────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain set: {X_train.shape[0]} samples")
    print(f"Test set:  {X_test.shape[0]} samples")
    
    # ── 5. Train Model ────────────────────────────────────────────
    print("\nTraining Random Forest (100 estimators)...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    print("Training complete!")
    
    # ── 6. Evaluate ───────────────────────────────────────────────
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n{'='*50}")
    print(f"  MODEL EVALUATION RESULTS")
    print(f"{'='*50}")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"{'='*50}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Team 2 Wins', 'Team 1 Wins']))
    
    # Feature importances
    print("Feature Importances:")
    for col, imp in sorted(zip(feature_cols, model.feature_importances_), key=lambda x: -x[1]):
        print(f"  {col:20s} → {imp:.4f}")
    
    # ── 7. Save Model & Encoders ──────────────────────────────────
    os.makedirs(models_dir, exist_ok=True)
    
    model_path = os.path.join(models_dir, 'random_forest_v1.pkl')
    encoders_path = os.path.join(models_dir, 'label_encoders_v1.pkl')
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"\nModel saved to: {model_path}")
    
    with open(encoders_path, 'wb') as f:
        pickle.dump(label_encoders, f)
    print(f"Label encoders saved to: {encoders_path}")
    
    # ── 8. Quick Sanity Check ─────────────────────────────────────
    # Reload and verify the model works
    with open(model_path, 'rb') as f:
        loaded_model = pickle.load(f)
    
    sanity_pred = loaded_model.predict(X_test[:5])
    print(f"\nSanity check — predictions on 5 test samples: {sanity_pred}")
    print("Phase 4 complete!")
    
    return model, label_encoders, accuracy

if __name__ == '__main__':
    train_random_forest(
        matches_path='data/processed/matches_clean.csv',
        models_dir='models/'
    )
