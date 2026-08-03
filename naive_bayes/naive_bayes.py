import pandas as pd
import numpy as np

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv('dataset/hotel_bookings.csv')
print(df.head())
print()
print(df.isnull().sum())
print()
print(df.info())
print()
print(df.columns)
print()
df = df.drop(["company", "reservation_status","reservation_status_date"], axis=1)

print(df.head())

cat_col = ["hotel", "arrival_date_month", "meal", "country", "market_segment", "distribution_channel", 
           "reserved_room_type", "assigned_room_type", "deposit_type", "customer_type"]

df["children"] = df["children"].fillna(df["children"].median())
df['country'] = df['country'].fillna('Unknown')

# Frequency encode agent instead of one-hot (too many unique values)
df['agent'] = df['agent'].fillna(0)
agent_freq = df['agent'].value_counts(normalize=True)
df['agent'] = df['agent'].map(agent_freq) 

df_encoded = pd.get_dummies(df, columns=cat_col, drop_first=True)

X = df_encoded.drop("is_canceled", axis=1)
y = df_encoded["is_canceled"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()
X_train_scaler = scaler.fit_transform(X_train)
X_test_scaler = scaler.transform(X_test)

model = GaussianNB()

model.fit(X_train_scaler, y_train)

y_pred = model.predict(X_test_scaler)

print("\nAccuracy score:")
print(accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification report:")
print(classification_report(y_test, y_pred))


