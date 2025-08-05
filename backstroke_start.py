import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Assuming df is your DataFrame containing the biomechanical parameters
# Define your features (columns you want to use for prediction)
features = [
    'Hands-off phase relative time (%)','Take-off phase relative time (%)',
    'Flight phase relative time (%)', 'Entry phase relative time (%)',
    'Resultant take-off velocity (m·s-1)', 'Resultant flight velocity (m·s-1)',
    'Resultant entry velocity (m·s-1)', 'Wrist entry angle (o)',
    'Shoulder entry angle (o)'    'Hip entry angle (o)',  'Back arc angle (o)',
    'Upper limb force at starting position (N/N)',   'Maximal upper limb force and time (N/N; %)',
    'Upper limb horizontal and vertical impulse',   'Lower limbs force at starting position (N/N)',
    '1st maximal lower limb force and time (N/N; %)',   'Intermediate lower limb force and time (N/N; %)',
    '2nd maximal lower limb force and time (N/N; %)', 'Lower limb horizontal、 vertical and medio-lateral impulse'
]

# Load your data into df (if not already loaded)
　 df = pd.read_csv('your_data.csv')

X = df[features].astype(float)
y = df['5 m start time (s)'].astype(float)  # Assuming this column exists in your DataFrame

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)
y_pred_lr = model_lr.predict(X_test)

mse_lr = mean_squared_error(y_test, y_pred_lr)
print(f"Linear Regression Mean Squared Error: {mse_lr}")

# print("Coefficients:", model_lr.coef_)
# print("Intercept:", model_lr.intercept_)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import StandardScaler

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define ANN architecture
model_ann = Sequential([
    Dense(64, activation='relu', input_shape=(X.shape[1],)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)  # Output layer (1 neuron for regression)
])

# Compile the model
model_ann.compile(optimizer='adam', loss='mean_squared_error')

# Split data into training and testing sets (if not already split)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the model
model_ann.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)

# Evaluate the model on test data
mse_ann = model_ann.evaluate(X_test, y_test)
print(f"ANN Mean Squared Error: {mse_ann}")
