import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

df = pd.read_csv('dataset/ford.csv')
print(df.head())
print()
print(df.isnull().sum())
print()
print(df.columns)
print(df.info())

num_col = ['year', 'price', 'mileage', 'tax', 'mpg', 'engineSize']
corr_matrix = df[num_col].corr()

sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.show()

sns.histplot(data=df, x="price", kde=True, bins=30)
plt.show()

sns.scatterplot(data=df, x="mileage", y="price")
plt.show()

# One-hot encode categoricals (drop_first=True to avoid dummy trap)
df_encoded = pd.get_dummies(df, columns=['model', 'transmission', 'fuelType'], drop_first=True)

# NOTE: removed df_encoded.astype(int) -> it was truncating price/mpg/tax decimals

X = df_encoded.drop(columns='price', axis=1)
y = df_encoded['price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# ---- StandardScaler ----
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit only on train
X_test_scaled = scaler.transform(X_test)         # transform test using same scaler

model = LinearRegression()
model.fit(X_train_scaled, y_train)

# ---- Prepare new data for prediction ----
data = pd.DataFrame({
    'model': ["Fiesta"],
    'year': [2026],
    'transmission': ["Automatic"],
    'mileage': [6894],
    'fuelType': ["Diesel"],
    'tax': [145],
    'mpg': [62.8],
    'engineSize': [2]
})

# Must match drop_first=True used on training data
data_encoded = pd.get_dummies(data, columns=['model', 'transmission', 'fuelType'], drop_first=True)
data_encoded = data_encoded.reindex(columns=X.columns, fill_value=0)

# Scale using the SAME fitted scaler (don't fit again!)
data_encoded_scaled = scaler.transform(data_encoded)

print("Prediction: ", model.predict(data_encoded_scaled))

# ---- Evaluate ----
y_pred = model.predict(X_test_scaled)
r2 = r2_score(y_test, y_pred)
print("R2 score:", r2)