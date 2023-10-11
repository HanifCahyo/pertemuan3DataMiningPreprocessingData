# Numpy merupakan library yang digunakan untuk operasi matriks dan array
import numpy as np

# Matplotlib merupakan library yang digunakan untuk visualisasi data
import matplotlib.pyplot as plt

# Pandas merupakan library yang digunakan untuk manipulasi data
import pandas as pd

# Scikit-learn merupakan library yang digunakan untuk machine learning
import sklearn as sk


dataset = pd.read_csv("Data - Copy.csv")  # Membaca file csv
x = dataset.iloc[:, :-1].values  # Mengambil semua data kecuali kolom terakhir
y = dataset.iloc[:, -1].values  # Mengambil kolom terakhir
# print(x)
# print(y)


from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])
# print(x)

from sklearn.compose import (
    ColumnTransformer,
)  # ColumnTransformer untuk mengubah data kategorikal menjadi numerikal
from sklearn.preprocessing import (
    OneHotEncoder,
)  # OneHotEncoder untuk mengubah data kategorikal menjadi numerikal

ct = ColumnTransformer(
    transformers=[("encoder", OneHotEncoder(), [0])], remainder="passthrough"
)  # OneHotEncoder untuk mengubah data kategorikal menjadi numerikal

x = np.array(ct.fit_transform(x))  # Mengubah data kategorikal menjadi numerikal
# print(x)

from sklearn.preprocessing import (
    LabelEncoder,
)  # LabelEncoder untuk mengubah data kategorikal menjadi numerikal

le = LabelEncoder()  # LabelEncoder untuk mengubah data kategorikal menjadi numerikal
y = le.fit_transform(y)  # Mengubah data kategorikal menjadi numerikal
# print(y)  # Menampilkan data y

from sklearn.model_selection import (
    train_test_split,
)  # train_test_split untuk membagi data menjadi data training dan data testing

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=1
)  # Memisahkan data menjadi data training dan data testing
# print(x_train)  # Menampilkan data x_train
# print(x_test)  # Menampilkan data x_test
# print(y_train)  # Menampilkan data y_train
# print(y_test)  # Menampilkan data y_test

from sklearn.preprocessing import (
    StandardScaler,
)  # StandardScaler untuk mengubah data menjadi data yang berdistribusi normal

sc = (
    StandardScaler()
)  # StandardScaler untuk mengubah data menjadi data yang berdistribusi normal
x_train[:, 3:] = sc.fit_transform(
    x_train[:, 3:]
)  # Mengubah data menjadi data yang berdistribusi normal
x_test[:, 3:] = sc.transform(
    x_test[:, 3:]
)  # Mengubah data menjadi data yang berdistribusi normal
# print(x_train)  # Menampilkan data x_train
# print(x_test)  # Menampilkan data x_test
# print(y_train)  # Menampilkan data y_train
print(y_test)  # Menampilkan data y_test
