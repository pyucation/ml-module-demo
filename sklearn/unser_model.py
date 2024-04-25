import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt


# Daten aus eigener Excel-Tabelle einlesen
df = pd.read_excel("sklearn/unser_datensatz.xlsx")
# print(df.head()) # gibt uns die Daten aus

# Dataframe teilen in y (Target-Variable : das was wir vorhersagen wollen)
# und X (unsere Datenmatrix mit den Trainingsdaten)
y_train = df["Preis"]
X_train = df.drop(columns=["Preis"])

# Pipeline inklusive Skalierung und Model bauen
pipeline = make_pipeline(StandardScaler(), LinearRegression())

# Training der Pipeline ausführen
# .to_numpy() müssen wir machen, weil sklearn mit numpy arrays und
# nihct mit DataFrames arbeitet
pipeline.fit(X_train.to_numpy(), y_train.to_numpy())

# ausgedachte Testdaten
# z.B. 2 Schlafzimmer, 110 m^2 und 28 €/MOnat Stromkosten
# und 4 Schlafzimmer, 400 m^2 und 130 €/Monat Stromkosten
X_test = np.array([
    [2, 110, 28],
    [4, 400, 130]
])
# Vorhersage des Models aus der Pipeline für unsere ausgedachten Testdaten
y_hat = pipeline.predict(X_test)
# Vorhersagen ausgeben
print(y_hat)

