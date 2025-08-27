# Crop Recommendation System — Real vs Synthetic

A full Flask project that:
- Trains a **real-data model** on Kaggle's Crop Recommendation dataset.
- Generates **synthetic data** (per-class multivariate Gaussian with shrinkage covariance) and trains a **synthetic-data model**.
- Serves a **beautiful web UI** where you can:
  - Enter inputs with **strict value ranges**.
  - See **predictions from both models** with confidence.
  - **Compare accuracy & F1** of real vs synthetic models with a chart.

> ✅ If the Kaggle CSV is missing, the training scripts auto-create a tiny toy dataset so you can still run the app end‑to‑end.  
> ⚠️ The toy dataset is only for demo—use the real CSV for meaningful results.

---

## 1) Project Structure

```
crop-reco-project/
├── app/
│   ├── app.py
│   ├── models/
│   ├── static/
│   │   └── styles.css
│   └── templates/
│       ├── base.html
│       ├── index.html
│       ├── result.html
│       └── compare.html
├── data/
│   └── crop_recommendation.csv   # ← Put Kaggle file here
├── scripts/
│   ├── train_real.py
│   ├── generate_synthetic.py
│   ├── train_synth.py
│   └── evaluate_models.py
└── requirements.txt
```

Get the Kaggle dataset: **Crop Recommendation Dataset** (features: N, P, K, temperature, humidity, ph, rainfall; label: crop). Save it as:
```
data/crop_recommendation.csv
```

---

## 2) Quickstart

```bash
# 1) Create a virtual env
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

# 2) Install deps
pip install -r requirements.txt

# 3) (Recommended) Place Kaggle CSV
#    data/crop_recommendation.csv

# 4) Train real-data model + save train/test splits
python scripts/train_real.py

# 5) Generate synthetic data from the real train split
python scripts/generate_synthetic.py

# 6) Train synthetic-data model
python scripts/train_synth.py

# 7) (Optional) Evaluate both models (also done on-demand by the web app)
python scripts/evaluate_models.py

# 8) Run the web app
python app/app.py
```

The app runs at **http://127.0.0.1:5000**.

---

## 3) Notes on Ranges & Validation

The UI enforces realistic ranges (min/max) and server validates too:

- N (0–140)
- P (5–145)
- K (5–205)
- temperature in °C (8–43)
- humidity % (20–100)
- ph (3.5–9.5)
- rainfall in mm (20–300)

Adjust these in `app/app.py` -> `RANGES` if needed.

---

## 4) Goal: Synthetic ≥ Real?

It’s **not guaranteed** that a model trained on synthetic data will beat one trained on real data when evaluated on **real** test data. This project includes tips:
- Better base model / hyperparameters (RandomForest grid already included).
- Tune synthetic sample multiplier in `generate_synthetic.py` (e.g., 1.5× per class).
- Ensure per-class covariance is well-conditioned (we use Ledoit-Wolf shrinkage).
- Try mixing real + synthetic for training the “synthetic” model.

Use the **Compare** page to see unbiased metrics on a held-out real test set.

---

## 5) Troubleshooting

- If you see a `scikit-learn InconsistentVersionWarning`, re-train models with your current sklearn version by re-running the training scripts.
- If CSV is missing, scripts fall back to a tiny built-in dataset to let you proceed.
- If class imbalance causes issues, increase `SYNTH_FACTOR` in `generate_synthetic.py`.
