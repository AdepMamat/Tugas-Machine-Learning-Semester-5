import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os

# --- 1️⃣ Cek apakah file ada ---
if not os.path.exists("kelulusan.csv"):
    print("File 'kelulusan.csv' tidak ditemukan. Membuat file baru...")

    data = {
        "IPK": [3.8,2.4,3.5,2.7,3.9,2.3,3.6,2.8,3.4,2.1,3.7,2.5,3.2,2.6,3.9,2.2,3.3,2.9,3.6,2.4,3.8,2.7,3.5,2.0,3.9,2.6,3.4,2.8,3.7,2.3],
        "Jumlah_Absensi": [3,10,5,8,2,11,4,7,5,12,4,9,6,8,2,10,5,7,3,9,4,8,5,12,3,10,5,7,4,11],
        "Waktu_Belajar_Jam": [12,3,9,5,14,2,10,6,8,3,11,4,9,5,13,3,8,6,12,4,11,5,9,2,13,4,8,5,10,3],
        "Lulus": [1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0]
    }

    df = pd.DataFrame(data)
    df.to_csv("kelulusan.csv", index=False)
    print("File 'kelulusan.csv' berhasil dibuat!\n")

# --- 2️⃣ Baca data ---
df = pd.read_csv("kelulusan.csv")
df.info()
print(df.head())

# --- 3️⃣ Cleaning data ---
print(df.isnull().sum())
df = df.drop_duplicates()

# --- 4️⃣ Exploratory Data Analysis (EDA) ---
sns.boxplot(x=df['IPK'])
plt.title("Boxplot IPK")
plt.show()

print(df.describe())

sns.histplot(df['IPK'], bins=10, kde=True)
plt.title("Distribusi IPK")
plt.show()

sns.scatterplot(x='IPK', y='Waktu_Belajar_Jam', data=df, hue='Lulus')
plt.title("Hubungan IPK vs Waktu Belajar")
plt.show()

sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Korelasi antar Fitur")
plt.show()

# --- 5️⃣ Feature Engineering ---
df['Rasio_Absensi'] = df['Jumlah_Absensi'] / 14
df['IPK_x_Study'] = df['IPK'] * df['Waktu_Belajar_Jam']
df.to_csv("processed_kelulusan.csv", index=False)
print("File 'processed_kelulusan.csv' berhasil disimpan!\n")

# --- 6️⃣ Splitting Dataset ---
X = df.drop('Lulus', axis=1)
y = df['Lulus']

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

print("Ukuran data:")
print("Train :", X_train.shape)
print("Valid :", X_val.shape)
print("Test  :", X_test.shape)
