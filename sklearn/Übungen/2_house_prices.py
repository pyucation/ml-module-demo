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
