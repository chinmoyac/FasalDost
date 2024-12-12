import joblib
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

# Load the model
model = joblib.load('arhar_model.pkl')  # Example for "arhar"

# Test data (replace with real test data)
X_test = np.array([[1.0, 2023, 30.5]])  # Example test input
y_test = np.array([1000])  # Example ground truth for evaluation

# Predict and evaluate
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Predictions: {predictions}")
print(f"Mean Squared Error: {mse}")
print(f"R² Score: {r2}")
