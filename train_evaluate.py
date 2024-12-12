import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt

# Step 1: Load the data
dataset = pd.read_csv('static/Arhar.csv')  # Update to correct path
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 3].values

# Step 2: Split the data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=0)

# Step 3: Train the model
model = DecisionTreeRegressor(max_depth=10)
model.fit(X_train, Y_train)
joblib.dump(model, 'arhar_model.pkl')
print("Model trained and saved as 'arhar_model.pkl'")

# Step 4: Evaluate the model
predictions = model.predict(X_test)
mse = mean_squared_error(Y_test, predictions)
r2 = r2_score(Y_test, predictions)
print(f"Mean Squared Error: {mse}")
print(f"R² Score: {r2}")

# Step 5: Visualize the results
plt.scatter(Y_test, predictions, color='blue', alpha=0.6)
plt.plot([min(Y_test), max(Y_test)], [min(Y_test), max(Y_test)], color='red', linestyle='--')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.show()
