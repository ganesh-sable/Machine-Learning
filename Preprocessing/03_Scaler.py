from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd

data = {
    "ID":[1,2,3,4,5,6,7],
    "Hour":[2,3,4,5,6,7,8],
    "Marks":[30,40,50,64,67,70,80]
}

df = pd.DataFrame(data)

# Standard Scaling
std_scaler = StandardScaler()
std_df = pd.DataFrame(
    std_scaler.fit_transform(df[['Hour','Marks']]),
    columns=['Hour','Marks']
)

print("StandardScaler")
print(std_df)

# MinMax Scaling
minmax_scaler = MinMaxScaler()
minmax_df = pd.DataFrame(
    minmax_scaler.fit_transform(df[['Hour','Marks']]),
    columns=['Hour','Marks']
)

print("\nMinMaxScaler")
print(minmax_df)
