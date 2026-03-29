import numpy as np
import matplotlib.pyplot as plt
import joblib

# Load data
data = np.load('split_data.npz')
X_test = data['X_test']
y_test = data['y_test']

model = joblib.load('model_regresi.pkl')

# Prediksi
y_pred = model.predict(X_test)

# ============================
# SORT BIAR KURVA RAPI
# ============================
sorted_idx = X_test[:,0].argsort()
X_sorted = X_test[sorted_idx]
y_pred_sorted = y_pred[sorted_idx]

# ============================
# SCATTER + CURVE
# ============================
plt.figure(figsize=(8,6))

# scatter
plt.scatter(X_test[:,0], y_test, alpha=0.3)

# kurva polinomial
plt.plot(X_sorted[:,0], y_pred_sorted, linewidth=2)

plt.xlabel("NDVI")
plt.ylabel("AGB")
plt.title("Regresi Polinomial NDVI vs AGB")

plt.tight_layout()
plt.savefig("polynomial_plot.png", dpi=150)
plt.show()

print("Plot tersimpan sebagai polynomial_plot.png")