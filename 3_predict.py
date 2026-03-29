import numpy as np
import joblib

# Load data testing dan model
data = np.load('split_data.npz')
X_test = data['X_test']
y_test = data['y_test']

model = joblib.load('model_regresi.pkl')

# Lakukan prediksi
y_pred = model.predict(X_test)

# Simpan hasil prediksi (opsional)
np.savez('prediksi.npz', y_test=y_test, y_pred=y_pred)
print("Prediksi selesai. Hasil tersimpan di 'prediksi.npz'")