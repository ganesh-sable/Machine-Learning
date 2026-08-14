import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("dataset/Clustering_gmm.csv")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

X = df

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

from sklearn.neighbors import NearestNeighbors

neighbor = NearestNeighbors(n_neighbors= 5)
neighbor.fit(X_scaled)

distance, indices = neighbor.kneighbors(X_scaled)

k_distance = np.sort(distance[:, 4])

plt.plot(k_distance)

plt.xlabel("Data Points")
plt.ylabel("5th Nearest Neighbor Distance")
plt.title("K-Distance Graph")
plt.savefig("Elbow_curve_gmm.png")
plt.show()

from sklearn.cluster import DBSCAN
dbscan = DBSCAN(
    eps = 0.04,
    min_samples= 5
)

df["Cluster"] = dbscan.fit_predict(X_scaled)
print(df.head())

print(pd.Series(df["Cluster"]).value_counts().sort_index())

plt.figure(figsize=(10, 6))

plt.scatter(
    df["Height"],
    df["Weight"],
    c=df["Cluster"]
)

plt.xlabel("Height")
plt.ylabel("Weight")
plt.title("DBSCAN Clustering")
plt.savefig("DBSCAN Clustering.png")
plt.show()