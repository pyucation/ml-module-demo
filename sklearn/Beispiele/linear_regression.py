import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


np.random.seed(42)

# synthetische Dtaen
x = 10 * np.random.rand(100, 1)
y = 3 * x + 2 + np.random.randn(100, 1) * 2.5

model = LinearRegression()

# Training
model.fit(x, y)

# Vorhersagen
x_new = np.linspace(0, 10, 100).reshape(100, 1)
y_pred = model.predict(x_new)

# Plotting
plt.figure(figsize=(8, 6))
plt.scatter(x, y, color='blue', label='Data Points')
plt.plot(x_new, y_pred, color='red', linewidth=2,
         label='Fit: $y = {:.2f}x + {:.2f}$'.format(model.coef_[0][0], model.intercept_[0]))
plt.xlabel('Feature X')
plt.ylabel('Zielvariable Y')
plt.title('Linear Regression Fit')
plt.legend()
plt.show()

# Koeffizienten (Anstieg und Schnittpunkt mit y-Achse)
print("Model slope: {:.2f}, Model intercept: {:.2f}".format(model.coef_[0][0], model.intercept_[0]))
