import numpy as np
from sklearn.model_selection import train_test_split

# Load data dari file .npz (opsional, bisa juga langsung dari df)
data = np.load('data_agb_ndvi.npz')
X = data['X']
y = data['y']

# Bagi dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set: {X_train.shape[0]:,} baris")
print(f"Testing set : {X_test.shape[0]:,} baris")

# Simpan hasil pembagian (opsional)
np.savez('split_data.npz', 
         X_train=X_train, X_test=X_test, 
         y_train=y_train, y_test=y_test)