import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


df = pd.read_csv("Social_Network_Ads.csv")
print(df.head())
print(df.isnull().sum())
print(df.info())

le = LabelEncoder()
df["Gender_encoded"] = le.fit_transform(df["Gender"])

df = df.drop(["User ID","Gender"], axis=1)
X = df.drop("Purchased", axis=1)
y = df["Purchased"]


X_train, X_test, y_train, y_test = train_test_split(
    X,y,
    test_size= 0.2,
    random_state= 42,
    stratify= y
)


scaler = StandardScaler()
X_train_scaler = scaler.fit_transform(X_train)
X_test_scaler = scaler.transform(X_test)

model = DecisionTreeClassifier(random_state=42)

model.fit(X_train_scaler, y_train)

y_pred = model.predict(X_test_scaler)

accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy score: ", accuracy*100, "%")

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))


print("\nClassification Report")
print(classification_report(y_test, y_pred))