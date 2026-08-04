import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('dataset/survey.csv')
#print(df.head())
print(df.info())

print()
print(df.isnull().sum())

df = df.drop(["Timestamp","comments"], axis=1)
print(df.head())

num_cols = df.select_dtypes(include=["int64", "float64"]).columns
cat_cols = df.select_dtypes(include=["object"]).columns.drop("treatment")


df[num_cols] = df[num_cols].fillna(df[num_cols].median())

for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

X = df_encoded.drop("treatment", axis=1)
y = df_encoded["treatment"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size = 0.2,
    random_state = 42
)

scaler = StandardScaler()

X_train_scaler = scaler.fit_transform(X_train)
X_test_scaler = scaler.transform(X_test)

model = SVC(probability=True)

model.fit(X_train_scaler, y_train)

y_pred = model.predict(X_test_scaler)

print("\nAccuracy: ", accuracy_score(y_test, y_pred))

print("\nConfusion matix: ")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("\nClassification report: ")
print(classification_report(y_test, y_pred))