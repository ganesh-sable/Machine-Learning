import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv('dataset/healthcare-dataset-stroke-data.csv')
print(df.head())

print()
print(df.info())
print()
print(df.isnull().sum())
print()
print(df.columns)

data = df.copy()
data = data.drop("id", axis=1)
print(data.columns)

data['bmi'] = data['bmi'].fillna(data['bmi'].median())
print(data.info())

print()
corr_col = ['age', 'hypertension', 'heart_disease','avg_glucose_level', 'bmi']
cat_col = ['gender','ever_married', 'work_type', 'Residence_type','smoking_status']

for i in cat_col:
    data[i] = data[i].astype('category')
print(data.info())
print()

data_encoded = pd.get_dummies(data, columns= cat_col, drop_first=True)
print(data_encoded)

X = data_encoded.drop('stroke', axis =1)
y = data_encoded['stroke']

X_train, X_test, y_train, y_test = train_test_split(
    X,y,
    test_size = 0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()
X_train_scaler = scaler.fit_transform(X_train)
X_test_scaler = scaler.transform(X_test)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train_scaler, y_train)

y_pred = model.predict(X_test_scaler)
print("\nAccuracy: ",accuracy_score(y_test, y_pred))

print("\nConfusion Matrix: ")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report: ")
print(classification_report(y_test, y_pred))
