import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("dataset/Bank_Churners.csv")
print(df.head())

print(df.isnull().sum())

cat_cols = df.select_dtypes(include= ["object"]).columns

X = df.drop(columns=list(cat_cols) + ["Attrition_Flag"])

from sklearn.preprocessing import MinMaxScaler 
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

from scipy.cluster.hierarchy import dendrogram, linkage

linked = linkage(X_scaled, method= "ward")

dendrogram(linked, truncate_mode="lastp", p=30)

plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Customers")
plt.ylabel("Distance")
plt.savefig("Dendrogram_Bank.png")
plt.show()


from sklearn.cluster import AgglomerativeClustering

hc = AgglomerativeClustering(
    n_clusters= 3,
    linkage= "ward"
)

df["Cluster"] = hc.fit_predict(X_scaled)

print(df)

print("\nCluster Counts:")
print(pd.Series(df["Cluster"]).value_counts().sort_index())

cluster_profile = df.groupby("Cluster")[
    [
        "Customer_Age",
        "Credit_Limit",
        "Total_Trans_Amt",
        "Total_Trans_Ct",
        "Total_Revolving_Bal",
        "Avg_Utilization_Ratio"
    ]
].mean()

print(cluster_profile)

plt.figure(figsize=(10, 6))

plt.scatter(
    df["Credit_Limit"],
    df["Total_Trans_Amt"],
    c=df["Cluster"]
)

plt.xlabel("Credit Limit")
plt.ylabel("Total Transaction Amount")
plt.title("Customer Clusters")
plt.savefig("Scatter_plot.png")
plt.show()