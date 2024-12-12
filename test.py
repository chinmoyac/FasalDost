import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, r2_score

# Load the trained model
model = joblib.load('arhar_model.pkl')

# Load test data
X_test = pd.read_csv('X_test.csv').values
Y_test = pd.read_csv('Y_test.csv').values

# Predict and evaluate
predictions = model.predict(X_test)
mse = mean_squared_error(Y_test, predictions)
r2 = r2_score(Y_test, predictions)

print(f"Mean Squared Error: {mse}")
print(f"R² Score: {r2}")
