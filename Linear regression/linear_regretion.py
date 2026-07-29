from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd

data ={
    "Hours" : [1,2,3,4,5],
    "Marks": [50,60,67,83,84]
}

df = pd.DataFrame(data)
print(df)

X = df[["Hours"]]
y = df["Marks"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size = 0.2,
    random_state = 42
)

print()
print("X training")
print(X_train)
print()
print("Y testing")
print(X_test)

print()
print("Y training")
print(y_train)
print()
print("Y testing")
print(y_test)

model = LinearRegression()

model.fit(X_train, y_train)

new_data = pd.DataFrame({
    "Hours":[6]
})
prediction = model.predict(new_data)
print()
print("Prediction")
print(prediction)