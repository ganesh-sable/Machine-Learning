import numpy as np
import pandas as pd

df = pd.read_csv("dataset/driver-data.csv")

print(df.head())

print("\n",df.isnull().sum())

print("\n", df.info())

X = df.drop(columns= "id")

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

from sklearn.cluster import KMeans

# Elbow Curve
inertia = []
k_range = range(1,11)
for k in k_range:
    kmeans = KMeans(
        n_clusters = k,
        random_state = 42,
        n_init = 10
    )

    kmeans.fit(X_scaled)

    inertia.append(kmeans.inertia_)

import matplotlib.pyplot as plt
plt.plot(
    k_range,
    inertia,
    marker = 'o'
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Curve")
plt.xticks(k_range)

plt.show()

# Silhouette Score

from sklearn.metrics import silhouette_score
silhouette_scores = []

k_ranges = range(2,11)

for k in k_ranges:
    kmeans = KMeans(
        n_clusters=k,
        random_state= 42,
        n_init= 10
    )

    labels = kmeans.fit_predict(X_scaled)

    score = silhouette_score(
        X_scaled,
        labels
    )
    silhouette_scores.append(score)

plt.plot(
    k_ranges,
    silhouette_scores,
    marker ="o"
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score")
plt.xticks(k_ranges)

plt.show()


best_k = k_ranges[np.argmax(silhouette_scores)]


final_kmeans = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)

df["Cluster"] = final_kmeans.fit_predict(X_scaled)

print(df.head())

df.to_csv("Clusterd_csv.csv", index= False)

cluster_summary = df.groupby("Cluster")[
    X.columns
].mean()

print(cluster_summary)

plt.figure(figsize=(8, 6))

plt.scatter(
    df["mean_dist_day"],
    df["mean_over_speed_perc"],
    c=df["Cluster"],
    s=50
)

plt.xlabel("Mean Distance Per Day")
plt.ylabel("Mean Over Speed Percentage")
plt.title("Driver Clusters using K-Means")

plt.show()