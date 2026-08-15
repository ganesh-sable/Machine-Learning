import pandas as pd
import numpy as np

df = pd.read_csv("data.csv")
print(df.head())

print("\nColumns: ")
print(df.columns)

df = df.drop(columns=["id", "Unnamed: 32"])

print("\nNull Values Count: ")
print(df.isnull().sum())

print("\nInfo: ")
print(df.info())

print("\n")

X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

from sklearn.decomposition import PCA

pca = PCA(n_components=10)

X_pca = pca.fit_transform(X_scaled)
print(X_pca)

print("\nExplained Varience Ratio: ")
print(pca.explained_variance_ratio_)

print("\nTotal Variance:")
print(pca.explained_variance_ratio_.sum())

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))

plt.plot(
    range(1, len(pca.explained_variance_ratio_) + 1),
    pca.explained_variance_ratio_,
    marker="o"
)

plt.xlabel("Principal Component")
plt.ylabel("Explained Variance Ratio")
plt.title("PCA Scree Plot")

plt.xticks(range(1, 31))

plt.grid()

plt.show()
