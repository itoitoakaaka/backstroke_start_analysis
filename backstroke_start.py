import sys

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold, cross_validate
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURES = [
    "Hands-off phase relative time (%)",
    "Take-off phase relative time (%)",
    "Flight phase relative time (%)",
    "Entry phase relative time (%)",
    "Resultant take-off velocity (m·s-1)",
    "Resultant flight velocity (m·s-1)",
    "Resultant entry velocity (m·s-1)",
    "Wrist entry angle (o)",
    "Shoulder entry angle (o)",
    "Hip entry angle (o)",
    "Back arc angle (o)",
    "Upper limb force at starting position (N/N)",
    "Maximal upper limb force and time (N/N; %)",
    "Upper limb horizontal and vertical impulse",
    "Lower limbs force at starting position (N/N)",
    "1st maximal lower limb force and time (N/N; %)",
    "Intermediate lower limb force and time (N/N; %)",
    "2nd maximal lower limb force and time (N/N; %)",
    "Lower limb horizontal、 vertical and medio-lateral impulse",
]

TARGET = "5 m start time (s)"


def evaluate_model(name, model, X, y, cv):
    """Evaluate one regression model with repeated cross-validation."""
    scoring = {
        "mae": "neg_mean_absolute_error",
        "mse": "neg_mean_squared_error",
        "r2": "r2",
    }
    scores = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=None)

    mae = -scores["test_mae"]
    rmse = np.sqrt(-scores["test_mse"])
    r2 = scores["test_r2"]

    print(f"\n{name}")
    print(f"  MAE : {mae.mean():.4f} ± {mae.std():.4f} s")
    print(f"  RMSE: {rmse.mean():.4f} ± {rmse.std():.4f} s")
    print(f"  R²  : {r2.mean():.4f} ± {r2.std():.4f}")


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "your_data.csv"
    df = pd.read_csv(csv_path)

    required = FEATURES + [TARGET]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    clean = df[required].apply(pd.to_numeric, errors="coerce").dropna()
    if len(clean) < 10:
        raise ValueError("Too few complete rows for a meaningful cross-validation demo.")

    X = clean[FEATURES]
    y = clean[TARGET]

    n_splits = min(5, max(2, len(clean) // 4))
    cv = RepeatedKFold(n_splits=n_splits, n_repeats=5, random_state=42)

    linear_model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", LinearRegression()),
        ]
    )

    mlp_model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                MLPRegressor(
                    hidden_layer_sizes=(64, 32),
                    activation="relu",
                    alpha=1e-3,
                    early_stopping=True,
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    )

    print(f"Rows used: {len(clean)}")
    print(f"Repeated CV: {n_splits} folds × 5 repeats")
    evaluate_model("Linear regression", linear_model, X, y, cv)
    evaluate_model("MLP regression", mlp_model, X, y, cv)


if __name__ == "__main__":
    main()
