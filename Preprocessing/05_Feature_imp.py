import numpy as np
import pandas as pd

df = pd.read_csv("dataset/bank-full.csv", sep=";")
print(df.head())

df.replace("unknown", np.nan, inplace=True)

print(df.isnull().sum())

missing_per = df.isnull().mean() * 100
drop_cols = missing_per[missing_per > 80].index
print("\n", drop_cols)

df = df.drop(columns= drop_cols, axis = 1)

print(df.columns)

num_cols = df.select_dtypes(include= "number").columns
cat_cols = df.select_dtypes(include= ["object"]).columns

for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

print("\n", df.isnull().sum())


X = df.drop('y', axis= 1)
y = df["y"]

num_cols = X.select_dtypes(include="number").columns
cat_cols = X.select_dtypes(include=["object"]).columns

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size = 0.2,
    random_state= 42,
    stratify = y
)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
preprocessing = ColumnTransformer(
    transformers =[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(), cat_cols)
    ]
)

X_train = preprocessing.fit_transform(X_train)
X_test = preprocessing.transform(X_test)

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state= 42)

model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

y_pred = model.predict(X_test)
print("\nAccuracy: ", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix: ")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report: ")
print(classification_report(y_test, y_pred))


# Feature importance
feature_names = preprocessing.get_feature_names_out()

feature_importance = pd.DataFrame({
    "Feature" : feature_names,
    "Importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values( by="Importance",
    ascending=False)

print(feature_importance)
