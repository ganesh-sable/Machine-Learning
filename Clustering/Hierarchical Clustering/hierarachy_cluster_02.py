import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("dataset/Clustering_gmm.csv")
print(df.head())

print("\nMissing Values: ")
print(df.isnull().sum())

print("\nInfo: ")
print(df.info())

X = df

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

from scipy.cluster.hierarchy import dendrogram, linkage

linked = linkage(
    X_scaled,
    method = "ward"
)

dendrogram(linked, truncate_mode= "lastp", p=30)

plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Customers")
plt.ylabel("Distance")
plt.savefig("Dendrogram_gmm.png")
plt.show()

from sklearn.cluster import AgglomerativeClustering

hc = AgglomerativeClustering(
    n_clusters= 2,
    linkage = "ward"
)

df["Cluster"] = hc.fit_predict(X_scaled)

print(df.head())

print(pd.Series(df["Cluster"]).value_counts().sort_index())

cluster_profile = df.groupby("Cluster")[["Weight", "Height"]].mean()

print(cluster_profile)

plt.figure(figsize=(10, 6))

plt.scatter(
    df["Weight"],
    df["Height"],
    c=df["Cluster"]
)

plt.xlabel("Credit Limit")
plt.ylabel("Total Transaction Amount")
plt.title("Customer Clusters")
plt.savefig("Scatter_plot_gmm.png")
plt.show()