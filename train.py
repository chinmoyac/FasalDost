import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import joblib

# Load the dataset
dataset = pd.read_csv('static/Arhar.csv')  # Update the file path
X = dataset.iloc[:, :-1].values  # Features: All columns except the last
Y = dataset.iloc[:, -1].values   # Target: The last column

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=0)

# Train the model
model = DecisionTreeRegressor(max_depth=10)
model.fit(X_train, Y_train)

# Save the trained model
joblib.dump(model, 'arhar_model.pkl')
print("Model trained and saved as 'arhar_model.pkl'")

# Save test data for evaluation
pd.DataFrame(X_test).to_csv('X_test.csv', index=False)
pd.DataFrame(Y_test).to_csv('Y_test.csv', index=False)
print("Test data saved as 'X_test.csv' and 'Y_test.csv'")
