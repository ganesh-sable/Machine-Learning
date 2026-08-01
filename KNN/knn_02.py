import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("dataset/KNNAlgorithmDataset.csv")

print(df.head())

print()
print(df.isnull().sum())
print(df.columns)

data = df.copy()
data = data.drop(columns=["Unnamed: 32", "id"], errors="ignore")

X = data.drop("diagnosis", axis =1)
y = data["diagnosis"]

X_train, X_test, y_train, y_test = train_test_split(
    X,y,
    test_size=0.2,
    random_state = 42,
    stratify=y
)

scaler = StandardScaler()
X_train_scaler = scaler.fit_transform(X_train)
X_test_scaler = scaler.transform(X_test)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train_scaler, y_train)

y_pred = model.predict(X_test_scaler)
print("\nAccuracy Score: ")
print(accuracy_score(y_test, y_pred))

print("\nConfusion Matrix: ")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report: ")
print(classification_report(y_test, y_pred))