import os, joblib, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
APP_DIR = os.path.join(BASE_DIR, "app")
MODELS_DIR = os.path.join(APP_DIR, "models")

FEATURES = ["N","P","K","temperature","humidity","ph","rainfall"]

SYNTH_CSV = os.path.join(DATA_DIR, "synthetic_data.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
LE_PATH = os.path.join(MODELS_DIR, "label_encoder.pkl")

def main():
    if not os.path.exists(SYNTH_CSV):
        raise FileNotFoundError("Run generate_synthetic.py first (synthetic_data.csv missing).")

    synth = pd.read_csv(SYNTH_CSV)
    X_train = synth[FEATURES].copy()
    y_train = synth["label"].copy()

    # Ensure labels are encoded consistent with real encoder (if exists)
    if os.path.exists(LE_PATH):
        le = joblib.load(LE_PATH)
        if y_train.dtype.kind not in "iu":
            y_train = le.transform(y_train)
    else:
        # Create new encoder if needed
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder().fit(y_train)
        y_train = le.transform(y_train)
        joblib.dump(le, LE_PATH)

    # Train RF with small grid
    param_grid = {
        "n_estimators": [300],
        "max_depth": [None, 16, 24],
        "min_samples_split": [2, 4],
        "min_samples_leaf": [1, 2]
    }
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    gs = GridSearchCV(rf, param_grid, scoring="accuracy", cv=3, n_jobs=-1, verbose=0)
    gs.fit(X_train, y_train)
    best = gs.best_estimator_
    joblib.dump(best, os.path.join(MODELS_DIR, "model_synth.pkl"))
    print("Saved:", os.path.join(MODELS_DIR, "model_synth.pkl"))

    # Optional: quick eval on real test split if present
    if os.path.exists(TEST_CSV):
        test = pd.read_csv(TEST_CSV)
        X_test = test[FEATURES].copy()
        y_test = test["label"].copy()
        y_pred = best.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        print(f"Synth model accuracy on REAL test: {acc:.4f}, F1: {f1:.4f}")

if __name__ == "__main__":
    main()
