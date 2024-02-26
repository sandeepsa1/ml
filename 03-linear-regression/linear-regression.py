#Step 1: Import packages and classes
import numpy as np
from sklearn.linear_model import LinearRegression

#Step 2: Provide data
x = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))
y = np.array([5, 20, 14, 32, 22, 38])

#Step 3: Create a model and fit it
model = LinearRegression()
model.fit(x, y)

#Step 4: Get results
r_sq = model.score(x, y)
print(f"coefficient of determination: {r_sq}")
print(f"intercept: {model.intercept_}")
print(f"slope: {model.coef_}")

#Step 5: Predict response
y_pred = model.predict(x)
print(f"predicted response:\n{y_pred}")
