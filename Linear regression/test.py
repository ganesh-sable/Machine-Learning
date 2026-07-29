from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

data ={
    "hours": [1,2,3],
    "Marks": [89,70,87]
}

df = pd.DataFrame(data)

X = df[["hours"]]
y = df["Marks"]

X_train, X_test, y_train, y_test = train_test_split(

    X, y,
    test_size=0.2,
    random_state=42
)

print(X_train)
print(X_test)

print(y_train)
print(y_test)