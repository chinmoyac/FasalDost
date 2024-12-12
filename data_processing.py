import pandas as pd

# Load the dataset
dataset = pd.read_csv('static/Arhar.csv')  # Update to the actual file path

# Separate features (X) and target (Y)
X = dataset.iloc[:, :-1].values  # Features: All columns except the last
Y = dataset.iloc[:, 3].values    # Target: Replace 3 with the correct index for your target variable
