"""
Aufgabe: Gegeben ist ein synthetischer Datensatz, der das Kaufverhalten von Kunden auf einer Website
abbildet.

Trainieren Sie ein logistisches Regressionsmodell, um zu entscheiden, ob neue Kunden ein
Produkt kaufen basieren auf den gegeben Features.
"""
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report


# Generate a synthetic binary classification dataset
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, n_clusters_per_class=1, random_state=42)
