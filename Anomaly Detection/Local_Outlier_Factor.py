import pandas as pd
import numpy as np

df = pd.read_csv("EU_energy_data.csv")
print(df.head())

print("\nColumns: ")
print(df.columns)

print("\nNull Value Count: ")
print(df.isnull().sum())

print("\nInfo: ")
print(df.info())

df = df.drop(columns=["Unnamed: 0", "fecha", "fecha_actualizacion", "sistema"])

X = df.copy()

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

from sklearn.neighbors import LocalOutlierFactor
model =LocalOutlierFactor(
    n_neighbors= 20,
    contamination=0.05
)


df["anomaly"] = model.fit_predict(X_scaled)
print("\n")
print(df)
print("\n")

df["score"] = model.negative_outlier_factor_

print(df)

print("\nResult:")
print(df.head(20))

anomalies = df[df["anomaly"] == -1]
print(anomalies.head(20))

print("\nAnomaly count: ")
print(df["anomaly"].value_counts())

print("\nAnomaly Percentage:")
print(df["anomaly"].value_counts(normalize=True) * 100)