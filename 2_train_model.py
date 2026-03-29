import numpy as np
from sklearn.linear_model import LinearRegression
import joblib

data = np.load('split_data.npz')
X_train = data['X_train']
y_train = data['y_train']

model = LinearRegression()
model.fit(X_train, y_train)

# ============================
# OUTPUT LEBIH JELAS
# ============================
print(f"Koefisien NDVI (X1): {model.coef_[0]:.4f}")
print(f"Koefisien NDVI² (X2): {model.coef_[1]:.4f}")
print(f"Intercept: {model.intercept_:.4f}")

joblib.dump(model, 'model_regresi.pkl')
print("Model tersimpan sebagai 'model_regresi.pkl'")