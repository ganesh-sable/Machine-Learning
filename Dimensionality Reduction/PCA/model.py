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

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    random_state= 42,
    stratify= y
)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(X_scaled, y_train)

from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test_scaled)
print("\nAccuracy: ", accuracy_score(y_test, y_pred))


print("\n")
print("\n")
print("\n")
#-------------PCA--------------------------
from sklearn.decomposition import PCA

pca = PCA(n_components=10)

X_pca = pca.fit_transform(X_scaled)
X_pca_test = pca.transform(X_test_scaled)

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

pca_model = LogisticRegression()
pca_model.fit(X_pca, y_train)

y_pred_pca = pca_model.predict(X_pca_test)
print("\nPCA Accuracy: ", accuracy_score(y_test, y_pred_pca))

