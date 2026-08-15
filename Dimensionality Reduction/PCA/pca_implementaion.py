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
pca_df = pd.DataFrame(
    X_pca,
    columns= ['PC1', 'PC2']
)

print("\nDataframe: ")
print(pca_df)


plt.figure(figsize=(8, 6))

for diagnosis in ["M", "B"]:

    mask = y == diagnosis

    plt.scatter(
        pca_df.loc[mask, "PC1"],
        pca_df.loc[mask, "PC2"],
        label=diagnosis
    )


plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA - Breast Cancer Dataset")
plt.legend()

plt.show()
