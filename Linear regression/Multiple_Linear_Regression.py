from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

import pandas as pd

df = pd.read_csv("dataset/ice-cream.csv")

X = df[["Temperature", "Rainfall"]]
y = df["IceCreamsSold"]

X_train, X_test, y_train, y_test = train_test_split(
    X,y,
    test_size = 0.3,
    random_state = 42
)

model = LinearRegression()
model.fit(X_train, y_train)
new_data = pd.DataFrame({
    "Temperature": [64.7],
    "Rainfall": [0.38]
})
prediction = model.predict(new_data)
print(prediction)

score = model.score(X_test, y_test)
print(score)

y_pred = model.predict(X_test)
r2 = r2_score(y_pred, y_test)
print(r2)

