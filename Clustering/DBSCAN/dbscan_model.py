import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("wine-clustering.csv")
print(df.head())
print(df.isnull().sum())
print(df.info())

X = df

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

from sklearn.neighbors import NearestNeighbors

neighbor = NearestNeighbors(n_neighbors = 5)
neighbor.fit(X_scaled)
distance, indices=neighbor.kneighbors(X_scaled)
k_distance = np.sort(distance[:, 4])
plt.plot(k_distance)

plt.xlabel("Data Points")
plt.ylabel("5th Nearest Neighbor Distance")
plt.title("K-Distance Graph")
plt.savefig("Elbow_curve.png")
plt.show()

from sklearn.cluster import DBSCAN

dbscan = DBSCAN(
    eps = 0.5,
    min_samples= 5
)
df["labels"] = dbscan.fit_predict(X_scaled)

print(df.head())

print(pd.Series(df["labels"]).value_counts().sort_index())

plt.figure(figsize=(10, 6))

plt.scatter(
    X_scaled[:, 0],
    X_scaled[:, 1],
    c=df["labels"]
)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("DBSCAN Clustering")
plt.savefig("DBSCAN_model.png")
plt.show()