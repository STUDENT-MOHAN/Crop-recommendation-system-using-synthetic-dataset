import os, joblib, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
APP_DIR = os.path.join(BASE_DIR, "app")
MODELS_DIR = os.path.join(APP_DIR, "models")

FEATURES = ["N","P","K","temperature","humidity","ph","rainfall"]

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

csv_path = os.path.join(DATA_DIR, "crop_recommendation.csv")

def load_or_make_toy():
    # Tiny toy dataset for demo if Kaggle CSV not found
    np.random.seed(42)
    rows = []
    crops = ["rice","maize","chickpea","kidneybeans","pigeonpeas","mothbeans","mungbean","blackgram","lentil","pomegranate","banana","mango","grapes","watermelon","muskmelon","apple","orange","papaya","coconut","cotton","jute","coffee"]
    for crop in crops[:8]:  # fewer to keep it tiny
        for _ in range(40):
            N = np.random.randint(0,141)
            P = np.random.randint(5,146)
            K = np.random.randint(5,206)
            temperature = np.random.uniform(10,40)
            humidity = np.random.uniform(30,95)
            ph = np.random.uniform(4.5,8.5)
            rainfall = np.random.uniform(30,250)
            rows.append([N,P,K,temperature,humidity,ph,rainfall,crop])
    df = pd.DataFrame(rows, columns=FEATURES+["label"])
    return df

if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    # Kaggle dataset often has 'label' as target
    # Ensure expected columns
    expected = set(FEATURES + ["label"])
    if not expected.issubset(set(df.columns)):
        raise ValueError(f"CSV missing required columns. Found: {df.columns}")
else:
    print("WARNING: Kaggle CSV not found. Using a tiny toy dataset for demo.")
    df = load_or_make_toy()

X = df[FEATURES].copy()
y = df["label"].copy()

# Encode labels consistently and save encoder
le = LabelEncoder()
y_enc = le.fit_transform(y)
joblib.dump(le, os.path.join(MODELS_DIR, "label_encoder.pkl"))

# Split and persist to data/train.csv and data/test.csv for consistent evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)
train = X_train.copy()
train["label"] = y_train
test = X_test.copy()
test["label"] = y_test
train.to_csv(os.path.join(DATA_DIR, "train.csv"), index=False)
test.to_csv(os.path.join(DATA_DIR, "test.csv"), index=False)

# Baseline RandomForest + small grid search
param_grid = {
    "n_estimators": [200],
    "max_depth": [None, 12, 20],
    "min_samples_split": [2, 4],
    "min_samples_leaf": [1, 2]
}
rf = RandomForestClassifier(random_state=42, n_jobs=-1)
gs = GridSearchCV(rf, param_grid, scoring="accuracy", cv=3, n_jobs=-1, verbose=0)
gs.fit(X_train, y_train)
best = gs.best_estimator_

# Save real-data model
joblib.dump(best, os.path.join(MODELS_DIR, "model_real.pkl"))

# Report
y_pred = best.predict(X_test)
acc = accuracy_score(y_test, y_pred)
from sklearn.metrics import f1_score
f1 = f1_score(y_test, y_pred, average="weighted")
print(f"Real model accuracy: {acc:.4f}, F1: {f1:.4f}")
print("Saved:", os.path.join(MODELS_DIR, "model_real.pkl"))
