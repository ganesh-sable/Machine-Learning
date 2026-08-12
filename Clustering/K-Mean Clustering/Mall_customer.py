import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("dataset/Mall_Customers.csv")

print(df.head())
print("\n", df.isnull().sum())

X = df.drop(columns = ["CustomerID","Gender"])

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

from sklearn.cluster import KMeans
inertia = []
K_range = range(1,11)

for k in K_range:
    kmeans = KMeans(
        n_clusters= k,
        random_state= 42,
        n_init= 10
    )

    kmeans.fit(X_scaled)

    inertia.append(kmeans.inertia_)

plt.plot(
    K_range,
    inertia,
    marker ="o"
)

plt.xlabel("K range")
plt.ylabel("Inertia")
plt.title("Elbow Curve")
plt.xticks(K_range)
plt.savefig('Mall_elbow_curve.png')
plt.show()


from sklearn.metrics import silhouette_score

silhouette_scores = []

k_ranges = range(2,11)

for k in k_ranges:
    kmeans = KMeans(
        n_clusters= k,
        random_state= 42,
        n_init = 10
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

plt.xlabel("K range")
plt.ylabel("silhouette_scores")
plt.title("silhouette_score")
plt.xticks(k_ranges)
plt.savefig('Mall_silhouette_scores.png')
plt.show()

best_k = k_ranges[np.argmax(silhouette_scores)]

final_kmeans = KMeans(
    n_clusters= best_k,
    random_state= 42,
    n_init= 10
)

df["Cluster"] = final_kmeans.fit_predict(X_scaled)
print(df.head())

df.to_csv("Mall_Clusterd.csv", index= False)

cluster_summary = df.groupby("Cluster")[
    X.columns
].mean()

print(cluster_summary)


plt.figure(figsize=(8, 6))

plt.scatter(
    df["Annual Income (k$)"],
    df["Spending Score (1-100)"],
    c=df["Cluster"],
    s=50
)

plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Mall Customer Segmentation using K-Means")
plt.savefig("Mall_sammary.png")
plt.show()