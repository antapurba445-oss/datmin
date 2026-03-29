import numpy as np
from sklearn.metrics import r2_score

# Load hasil prediksi
data = np.load('prediksi.npz')
y_test = data['y_test']
y_pred = data['y_pred']

# Hitung R²
r2 = r2_score(y_test, y_pred)
print(f"Koefisien Determinasi (R²): {r2:.4f}")

# Simpan nilai R² ke file teks
with open('hasil_evaluasi.txt', 'w') as f:
    f.write(f"R² = {r2:.4f}\n")