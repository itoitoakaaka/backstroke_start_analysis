# backstroke_start_analysis

Regression demo for modeling 5 m backstroke-start performance from biomechanical features.

## Why this project exists

This repository connects swimming biomechanics with reproducible machine-learning evaluation. It compares a simple linear baseline with a nonlinear multilayer perceptron while keeping preprocessing inside each cross-validation fold.

## Inputs

The model expects biomechanical predictors such as:

- phase timing
- take-off / flight / entry velocity
- entry angles
- trunk/back-arc angle
- upper- and lower-limb force / impulse variables

Target:

- 5 m start time (s)

## Models

- Linear regression
- MLPRegressor with two hidden layers

Both models use StandardScaler inside a scikit-learn Pipeline so the scaler is fitted only on training folds.

## Evaluation

The script uses repeated K-fold cross-validation and reports:

- MAE
- RMSE
- R²

This is intended as a methodological comparison, not evidence that a neural network necessarily outperforms a linear model. For small biomechanical datasets, uncertainty and sample size should be considered carefully.

## Setup

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

## Usage

    python backstroke_start.py biomechanics_data.csv

The input CSV must contain all feature columns listed in backstroke_start.py and the target column "5 m start time (s)".
