import pandas as pd
import numpy as np

df = pd.read_csv("dataset/Mall_Customers.csv")

print(df.head())
df = df.drop("CustomerID", axis = 1)
cat_cols = df.select_dtypes(include= ["object"]).columns
num_cols = df.select_dtypes(include= "number").columns


X = df

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans

column = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ]
)

X_scaled = column.fit_transform(X)


inertia = []
k_range = range(1,11)

for k in k_range:
    kmeans = KMeans(
        n_clusters= k,
        random_state= 42,
        n_init= 10
    )

    kmeans.fit(X_scaled)

    inertia.append(kmeans.inertia_)

import matplotlib.pyplot as plt

plt.plot(
    k_range,
    inertia,
    marker = "o"
)

plt.xlabel("K range")
plt.ylabel("Inertia")
plt.xticks(k_range)
plt.show()

from sklearn.metrics  import silhouette_score

silhouette_scores = []
k_ranges = range(2,11)

for k in k_ranges:
    kmeans = KMeans(
        n_clusters= k,
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
    marker = "o"
)

plt.xlabel("K range")
plt.ylabel("silhouette_scores")
plt.xticks(k_ranges)
plt.show()

best_k = k_ranges[np.argmax(silhouette_scores)]
print("\nBest K: ", best_k)

final_kmean = KMeans(
    n_clusters= best_k,
    random_state= 42,
    n_init= 10
)

df["Cluster"] = final_kmean.fit_predict(X_scaled)

print(df)

cluster_summary = df.groupby("Cluster")[
    X.columns
].mean()

print(cluster_summary)

plt.scatter(
    df["Annual Income (k$)"],
    df["Spending Score (1-100)"],
    c=df["Cluster"],
    s=50
)

plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Mall Customer Segmentation using K-Means")
plt.show()