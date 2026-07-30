import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,  accuracy_score, classification_report

df = sns.load_dataset("dataset/titanic")

print(df.head())
print()
df = df.drop(columns=['alive', 'class', 'who', 'embarked', 'deck'])

print(df.head())
print()
print(df.isnull().sum())

df["age"] = df["age"].fillna(df["age"].median())
df["embark_town"] = df["embark_town"].fillna(df["embark_town"].mode()[0])
print(df.isnull().sum())

print(df.info())
print(df.columns)


cat_col = ['sex', 'embark_town', 'alone']

df_encoded = pd.get_dummies(df, columns=cat_col, drop_first=True)
bool_cols = df_encoded.select_dtypes(include='bool').columns
df_encoded[bool_cols] = df_encoded[bool_cols].astype(int)
print(df_encoded.head())

X = df_encoded.drop("survived", axis =1)
y = df_encoded["survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy score: ", accuracy)

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))


print("\nClassification Report")
print(classification_report(y_test, y_pred))

df_encoded.to_csv("Prediction.csv")