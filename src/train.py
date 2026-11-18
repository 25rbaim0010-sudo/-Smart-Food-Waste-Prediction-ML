"""
Smart Food Waste Prediction and Management
Reproducible academic demonstration using synthetic daily canteen data.

Run:
    python -m pip install -r requirements.txt
    python src/train.py
    streamlit run app.py
"""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib

# Use a non-interactive backend for saving charts in command-line training.
# This prevents Tkinter GUI-thread cleanup warnings on Windows.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR, OUT_DIR, MODEL_DIR = ROOT / "data", ROOT / "outputs", ROOT / "models"
DATA_DIR.mkdir(exist_ok=True); OUT_DIR.mkdir(exist_ok=True); MODEL_DIR.mkdir(exist_ok=True)

FEATURES = [
    "expected_customers", "day_of_week", "weekend", "temperature_c",
    "special_event", "holiday", "previous_day_consumption"
]
TARGET = "meals_consumed"
NUMERIC = FEATURES

def generate_dataset(n=1200, seed=42):
    """Create reproducible synthetic records; not real canteen measurements."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    dow = dates.dayofweek.to_numpy()
    weekend = (dow >= 5).astype(int)
    holiday = (rng.random(n) < 0.045).astype(int)
    special_event = (rng.random(n) < 0.08).astype(int)
    temperature = np.clip(27 + 5*np.sin(np.arange(n)*2*np.pi/365) + rng.normal(0, 3, n), 16, 42)
    customers = (430 - 115*weekend - 220*holiday + 55*special_event + rng.normal(0, 48, n)).clip(80, 650).round().astype(int)
    meals = np.zeros(n, dtype=float)
    for i in range(n):
        lag = meals[i-1] if i else customers[i]*0.92
        noise = rng.normal(0, 17)
        meals[i] = max(40, 0.79*customers[i] + 0.13*lag + 9*special_event[i] - 7*holiday[i] - 0.35*abs(temperature[i]-27) + noise)
    previous = np.r_[meals[0], meals[:-1]]
    # operational quantities are descriptive only and are excluded from model features
    prepared = np.ceil(meals + rng.normal(22, 8, n)).clip(0)
    df = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "expected_customers": customers,
        "day_of_week": dow,
        "weekend": weekend,
        "temperature_c": temperature.round(1),
        "special_event": special_event,
        "holiday": holiday,
        "previous_day_consumption": previous.round(1),
        "meals_consumed": meals.round(1),
        "food_prepared": prepared.astype(int),
    })
    df["leftover_meals"] = (df["food_prepared"] - df["meals_consumed"]).clip(lower=0).round(1)
    return df

def build_models():
    lr = Pipeline([("scale", StandardScaler()), ("model", LinearRegression())])
    rf = RandomForestRegressor(n_estimators=250, max_depth=12, min_samples_leaf=3,
                               random_state=42, n_jobs=-1)
    return {"Linear Regression": lr, "Random Forest Regression": rf}

def train_and_evaluate():
    df = generate_dataset()
    df.to_csv(DATA_DIR/"canteen_data.csv", index=False)
    # Keep the time order: first 80% train, final 20% test.
    split = int(len(df)*0.8)
    train, test = df.iloc[:split], df.iloc[split:]
    X_train, y_train = train[FEATURES], train[TARGET]
    X_test, y_test = test[FEATURES], test[TARGET]
    results, fitted = [], {}
    for name, model in build_models().items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        results.append({
            "Model": name,
            "MAE (meals)": round(mean_absolute_error(y_test, pred), 3),
            "RMSE (meals)": round(mean_squared_error(y_test, pred)**0.5, 3),
            "R2": round(r2_score(y_test, pred), 4)
        })
        fitted[name] = model
        joblib.dump(model, MODEL_DIR/(name.lower().replace(" ", "_")+".joblib"))
    metrics = pd.DataFrame(results)
    metrics.to_csv(OUT_DIR/"model_metrics.csv", index=False)
    (OUT_DIR/"metrics.json").write_text(metrics.to_json(orient="records", indent=2), encoding="utf-8")

    # Save predictions for transparent review.
    best_name = metrics.sort_values("RMSE (meals)").iloc[0]["Model"]
    best_pred = fitted[best_name].predict(X_test)
    pred_df = test[["date", TARGET]].copy()
    pred_df["predicted_meals"] = np.round(best_pred, 1)
    pred_df.to_csv(OUT_DIR/"test_predictions.csv", index=False)

    # Actual vs predicted plot.
    plt.figure(figsize=(8, 5))
    plt.scatter(y_test, best_pred, alpha=.65)
    lo, hi = min(y_test.min(), best_pred.min()), max(y_test.max(), best_pred.max())
    plt.plot([lo, hi], [lo, hi], linestyle="--")
    plt.xlabel("Actual meals"); plt.ylabel("Predicted meals")
    plt.title(f"Actual vs Predicted — {best_name}")
    plt.tight_layout(); plt.savefig(OUT_DIR/"actual_vs_predicted.png", dpi=160); plt.close()

    # Model RMSE comparison.
    plt.figure(figsize=(7, 4))
    plt.bar(metrics["Model"], metrics["RMSE (meals)"])
    plt.ylabel("RMSE (meals)"); plt.title("Model RMSE Comparison")
    plt.xticks(rotation=10); plt.tight_layout()
    plt.savefig(OUT_DIR/"rmse_comparison.png", dpi=160); plt.close()

    sample = pd.DataFrame([{
        "expected_customers": 470, "day_of_week": 2, "weekend": 0,
        "temperature_c": 30.0, "special_event": 0, "holiday": 0,
        "previous_day_consumption": 425.0
    }])
    sample_result = {name: round(float(model.predict(sample[FEATURES])[0]), 1)
                     for name, model in fitted.items()}
    (OUT_DIR/"sample_prediction.json").write_text(json.dumps({
        "input": sample.iloc[0].to_dict(), "predictions": sample_result,
        "note": "Illustrative estimates from the reproducible synthetic demonstration."
    }, indent=2), encoding="utf-8")
    print("\nMODEL EVALUATION")
    print(metrics.to_string(index=False))
    print("\nSAMPLE PREDICTION")
    print(json.dumps(sample_result, indent=2))
    print(f"\nSaved dataset, models, metrics and plots under: {ROOT}")
    return metrics

if __name__ == "__main__":
    train_and_evaluate()
