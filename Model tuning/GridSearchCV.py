import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("dataset/data.csv")
print(df.head())

print()
print(df.isnull().sum())
print()
print(df.info())
print()
print(df.columns)
print()

print(df["diagnosis"].unique())
'''
cat_col = ["diagnosis"]
df_encoded = pd.get_dummies(df, columns = cat_col, drop_first=True)'''

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df["diagnosis"] = le.fit_transform(df["diagnosis"])
X = df.drop(["id","diagnosis","Unnamed: 32"], axis = 1)
y = df["diagnosis"]


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestClassifier()

classifier = GridSearchCV(estimator = model, param_grid={
    "n_estimators": [3,5,7,9,13], 
    "max_depth": [3,5,10],
    "min_samples_split": [2,3,5]
}, cv = 5, return_train_score=False)

classifier.fit(X_scaled, y)

print(classifier.best_estimator_)
print(classifier.best_score_)




