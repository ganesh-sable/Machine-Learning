import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = sns.load_dataset("iris")
df.to_csv("iris.csv")
print(df.head())
print(df.isnull().sum())

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

le = LabelEncoder()
df["species"] = le.fit_transform(df["species"])

X = df.drop("species", axis = 1)
y = df["species"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression()

scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')
print(scores)