# backstroke_start_analysis

Modeling and predicting backstroke start performance using linear regression and artificial neural networks based on biomechanical parameters.

## Overview

This project investigates the relationship between biomechanical parameters during the backstroke start and the 5m start time. It implements both Linear Regression and a feedforward Artificial Neural Network (ANN) to predict start performance based on kinematic and kinetic features.

## Features

- **Linear Regression Model**: Baseline prediction of 5m start time from biomechanical parameters using scikit-learn.
- **ANN Model**: Deep learning approach using a TensorFlow/Keras Sequential model with dense layers and dropout regularization.
- **Biomechanical Features**: Includes phase timings (hands-off, take-off, flight, entry), velocities, entry angles, arc angles, and force parameters.

## Biomechanical Parameters

The model uses 18 features including:
- Phase relative times (hands-off, take-off, flight, entry)
- Resultant velocities (take-off, flight, entry)
- Entry angles (wrist, shoulder, hip)
- Back arc angle
- Upper/lower limb forces and impulses

## Requirements

- Python 3.8+
- pandas
- scikit-learn
- TensorFlow (optional, for ANN model)

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python backstroke_start.py <csv_data_file>
# Example:
python backstroke_start.py biomechanics_data.csv
```

> **Note**: The CSV file must contain all 18 biomechanical parameter columns and a `5 m start time (s)` target column. If TensorFlow is not installed, only the Linear Regression model will run.
