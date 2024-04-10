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

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline that includes scaling and logistic regression
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Feature scaling
    ('log_reg', LogisticRegression())  # Logistic regression model
])

# Train the model
pipeline.fit(X_train, y_train)

# Make predictions on the test set
predictions = pipeline.predict(X_test)

# Evaluate the model
print(f"Accuracy: {accuracy_score(y_test, predictions):.2f}")
print(classification_report(y_test, predictions))