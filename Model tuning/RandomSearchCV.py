import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

print(df.columns)
print(df.head())
print(df.isnull().sum())
print(df.info())

X = df.drop("target", axis = 1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,
    stratify= y
)


model = RandomForestClassifier(random_state= 42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("\nAccuracy:",accuracy_score(y_test, y_pred))


parameter = {
    'n_estimators': [100, 200, 300],      
    'max_depth': [None, 10, 20],          
    'min_samples_split': [2, 5],          
    'min_samples_leaf': [1, 2],           
    'max_features': ['sqrt', 'log2']      
}

from sklearn.model_selection import GridSearchCV

grid = GridSearchCV(estimator= model, param_grid= parameter, cv = 5, return_train_score= False)

grid.fit(X_train, y_train)

best_grid = grid.best_estimator_
print(best_grid)

print("\n")

y_pred_grid = best_grid.predict(X_test)
print("\nAccuracy:",accuracy_score(y_test, y_pred_grid))


from sklearn.model_selection import RandomizedSearchCV

random = RandomizedSearchCV(
    estimator=model,
    param_distributions=parameter,
    n_iter=10,
    cv=5,
    random_state=42,
    return_train_score=False
)

random.fit(X_train, y_train)

best_random = random.best_estimator_
print(best_random)

print("\n")

y_pred_random = best_random.predict(X_test)
print("\nAccuracy:",accuracy_score(y_test, y_pred_random))
