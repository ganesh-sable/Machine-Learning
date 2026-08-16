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

print(df["sistema"].unique())
print(df["sistema"].nunique())

print(df["tipo_moneda"].unique())
print(df["origen_dato"].unique())

df = df.drop(columns=["Unnamed: 0", "fecha", "fecha_actualizacion", "sistema"])

X = df.copy()

from sklearn.ensemble import IsolationForest

model = IsolationForest(
    contamination=0.05,
    random_state=42
)

model.fit(X)
df["anomaly"] = model.predict(X)

print(df)

print("\nAnomaly Count:")
print(df["anomaly"].value_counts())

print("\nAnomaly Percentage:")
print(df["anomaly"].value_counts(normalize=True) * 100)

anomalies = df[df["anomaly"] == -1]

print(anomalies.head(20))

scores = model.decision_function(X)

df["score"] = scores

print(
    df[["precio", "anomaly", "score"]]
    .sort_values("score")
    .head(20)
)