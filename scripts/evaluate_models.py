import os, joblib, numpy as np, pandas as pd
from sklearn.metrics import accuracy_score, f1_score

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
APP_DIR = os.path.join(BASE_DIR, "app")
MODELS_DIR = os.path.join(APP_DIR, "models")

FEATURES = ["N","P","K","temperature","humidity","ph","rainfall"]

p_real = os.path.join(MODELS_DIR, "model_real.pkl")
p_synth = os.path.join(MODELS_DIR, "model_synth.pkl")
p_le = os.path.join(MODELS_DIR, "label_encoder.pkl")
p_test = os.path.join(DATA_DIR, "test.csv")

real = joblib.load(p_real) if os.path.exists(p_real) else None
synth = joblib.load(p_synth) if os.path.exists(p_synth) else None

if not os.path.exists(p_test):
    raise FileNotFoundError("No test.csv found. Run train_real.py first.")

test = pd.read_csv(p_test)
X = test[FEATURES].copy()
y = test["label"].copy()

def metrics(model):
    if model is None:
        return None, None
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, average="weighted")
    return float(acc), float(f1)

acc_r, f1_r = metrics(real)
acc_s, f1_s = metrics(synth)

print("Real Model  -> Acc:", acc_r, "F1:", f1_r)
print("Synth Model -> Acc:", acc_s, "F1:", f1_s)
