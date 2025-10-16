import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("kelulusan.csv")
print(df.info())
print(df.head())

# Cek missing values dan hapus duplikat
print(df.isnull().sum())
df = df.drop_duplicates()

# Visualisasi boxplot untuk IPK
sns.boxplot(x=df['IPK'])
plt.title("Boxplot IPK")
plt.show()

# Statistik deskriptif
print(df.describe())

# Histogram distribusi IPK
sns.histplot(df['IPK'], bins=10, kde=True)
plt.title("Distribusi IPK")
plt.show()

# Scatterplot IPK vs Waktu Belajar
sns.scatterplot(x='IPK', y='Waktu_Belajar_Jam', data=df, hue='Lulus')
plt.title("IPK vs Waktu Belajar Jam")
plt.show()

# Korelasi
# Pastikan hanya kolom numerik untuk heatmap
df_corr = df.select_dtypes(include=['float64', 'int64'])
sns.heatmap(df_corr.corr(), annot=True, cmap="coolwarm")
plt.title("Heatmap Korelasi")
plt.show()

# Fitur turunan
df['Rasio_Absensi'] = df['Jumlah_Absensi'] / 14
df['IPK_x_Study'] = df['IPK'] * df['Waktu_Belajar_Jam']

# Simpan data yang sudah diproses
df.to_csv("processed_kelulusan.csv", index=False)

# ================================
# Pembagian Dataset
# ================================

# Cek jumlah data per kelas label
print("Distribusi label:\n", df['Lulus'].value_counts())

# Siapkan fitur dan label
X = df.drop('Lulus', axis=1)
y = df['Lulus']

# Stratified split hanya bisa dilakukan jika setiap kelas punya >=2 data
label_counts = y.value_counts()
if (label_counts < 2).any():
    print("⚠️ Ada label dengan jumlah kurang dari 2. Menghapus label langka...")
    rare_labels = label_counts[label_counts < 2].index
    df = df[~df['Lulus'].isin(rare_labels)]
    X = df.drop('Lulus', axis=1)
    y = df['Lulus']

# Split menjadi train dan sisa
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42)

# Split sisa menjadi val dan test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

print("Shapes:")
print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)
