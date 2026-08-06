from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import numpy as np

df = pd.read_csv("dataset/petrol_consumption.csv")

X = df.drop(columns=["Petrol_Consumption"])
y = df["Petrol_Consumption"]

X_train, X_test, y_train, y_test = train_test_split(
    X,y,
    test_size = 0.2,
    random_state = 42
)

model = RandomForestRegressor(n_estimators=500, random_state=42)

model.fit(X_train, y_train)

new_data = pd.DataFrame({
    'Petrol_tax': [8.50, 7.00],
    'Average_income': [4500, 3800],
    'Paved_Highways': [1500, 2500],
    'Population_Driver_licence(%)': [0.540, 0.580]
})

print("Prediction: ", model.predict(new_data))


y_pred = model.predict(X_test)

print("MAE:",  mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R² Score:", r2_score(y_test, y_pred))