import os, joblib, numpy as np, pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import LabelEncoder

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
APP_DIR = os.path.join(BASE_DIR, "app")
MODELS_DIR = os.path.join(APP_DIR, "models")

FEATURES = ["N","P","K","temperature","humidity","ph","rainfall"]
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
REAL_CSV = os.path.join(DATA_DIR, "crop_recommendation.csv")

SYNTH_FACTOR = 1.2  # multiplier per class (tune to >1.0 for more synthetic samples)

def make_bounds(df):
    bounds = {}
    for col in FEATURES:
        bounds[col] = (df[col].min(), df[col].max())
    return bounds

def sample_mvnormal(mean, cov, n):
    return np.random.multivariate_normal(mean, cov, size=n)

def main():
    if os.path.exists(TRAIN_CSV):
        train = pd.read_csv(TRAIN_CSV)
    else:
        # Fallback: build from real CSV if available; else create small toy dataset like train_real.py
        if os.path.exists(REAL_CSV):
            real = pd.read_csv(REAL_CSV)
            train = real.sample(frac=0.8, random_state=42)
            # Assume label as 'label'
            if "label" not in train.columns:
                raise ValueError("Expected 'label' column in CSV.")
        else:
            print("No train.csv or real CSV found. Run train_real.py first.")
            return

    # Determine if labels are encoded ints; if not, encode to group easily
    if train["label"].dtype.kind in "iu":
        y = train["label"].values
        le = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl")) if os.path.exists(os.path.join(MODELS_DIR, "label_encoder.pkl")) else None
        label_names = le.inverse_transform(np.unique(y)) if le is not None else np.unique(y).astype(str)
    else:
        # string labels
        le = LabelEncoder()
        y = le.fit_transform(train["label"])
        label_names = le.inverse_transform(np.unique(y))
        # Save encoder so synth model matches real encoder
        joblib.dump(le, os.path.join(MODELS_DIR, "label_encoder.pkl"))

    X = train[FEATURES].copy().reset_index(drop=True)
    y = pd.Series(y)

    bounds = make_bounds(train)

    synth_rows = []
    for cls in np.unique(y):
        Xc = X[y==cls]
        n = Xc.shape[0]
        # target synth size
        m = int(max(1, SYNTH_FACTOR * n))
        # Fit Ledoit-Wolf covariance to be stable
        lw = LedoitWolf().fit(Xc.values)
        mean = lw.location_
        cov = lw.covariance_
        S = sample_mvnormal(mean, cov, m)
        S = pd.DataFrame(S, columns=FEATURES)
        # clip to observed bounds to keep realism
        for col in FEATURES:
            lo, hi = bounds[col]
            S[col] = S[col].clip(lo, hi)
        lbl = np.array([cls]*m)
        synth_rows.append(pd.concat([S, pd.Series(lbl, name="label")], axis=1))

    synth = pd.concat(synth_rows, ignore_index=True)
    # If we had string labels originally, map back to names
    if train["label"].dtype.kind not in "iu" and 'le' in locals():
        synth["label"] = le.inverse_transform(synth["label"].astype(int))
    synth.to_csv(os.path.join(DATA_DIR, "synthetic_data.csv"), index=False)
    print("Wrote synthetic data ->", os.path.join(DATA_DIR, "synthetic_data.csv"), "rows:", len(synth))

if __name__ == "__main__":
    main()
