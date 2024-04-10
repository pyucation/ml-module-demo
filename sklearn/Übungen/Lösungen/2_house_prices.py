"""
Gegeben ist ein berühmter Datensatz: California Housing Dataset
Ihre Aufgabe ist es, ein Regressionsmodell zu trainieren, welches zu den gegebenen Features
den Preis (bzw. Median des Hauswertes) vorhersagt.
"""
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

# Load the California housing dataset
california = fetch_california_housing()
X = pd.DataFrame(california.data, columns=california.feature_names)
y = pd.Series(california.target, name='MedianHouseValue')  # Median house value in 100,000's

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Linear Regression model
model = LinearRegression()

# Train the model on the training set
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
print(f"Mean Squared Error: {mse:.2f}")
print(f"R^2 Score: {r2:.2f}")

# Example: Predicting the median house value for a new data point
new_data = [[8.3252, 41, 6.984126, 1.023810, 322, 2.555556, 37.88, -122.23]]  # Sample features
new_value_pred = model.predict(new_data)
print(f"Predicted Median House Value: ${new_value_pred[0]*100_000:.2f}")

