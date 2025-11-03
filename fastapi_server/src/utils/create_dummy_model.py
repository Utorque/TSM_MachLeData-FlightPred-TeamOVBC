import joblib
import numpy as np
from sklearn.linear_model import LinearRegression

X = np.array([[1, 100], [2, 200], [3, 300], [4, 400]])
y = np.array([120, 230, 310, 410])

model = LinearRegression()
model.fit(X, y)

joblib.dump(model, "model/dummy_model.pkl")
print("Dummy model saved in model/dummy_model.pkl")