from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.pipeline import make_pipeline

# Datensatz laden
data = load_iris()
# Train-Test-Split
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# Pipeline mit StandardScaler als Preprocessing-Schritt
# Model = SVC (Support Vector Classifier)
pipeline = make_pipeline(StandardScaler(), SVC())

# Training
pipeline.fit(X_train, y_train)

# Testing
print("Test score: {:.3f}".format(pipeline.score(X_test, y_test)))
