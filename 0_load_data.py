import pandas as pd
import numpy as np

# Load data
file_csv = r"C:\Project SINTA 5\output\data_agb_ndvi_full.csv"
df = pd.read_csv(file_csv)
print(f"Total data: {len(df):,} baris")

# ============================
# TAMBAHAN: NDVI KUADRAT
# ============================
df['NDVI2'] = df['NDVI'] ** 2

# Pisahkan fitur (X) dan target (y)
X = df[['NDVI', 'NDVI2']]   # ← DIUBAH
y = df['AGB']

# Simpan
np.savez('data_agb_ndvi.npz', X=X.values, y=y.values)
print("Data tersimpan sebagai 'data_agb_ndvi.npz'")