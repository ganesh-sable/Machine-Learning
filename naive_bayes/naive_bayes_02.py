from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import pandas as pd

df = pd.read_csv("dataset/User_Data.csv")

df["Gender"] = df["Gender"].map({'Male': 1, 'Female': 0})

X = df[["Gender", "Age", "EstimatedSalary"]]
y = df["Purchased"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size = 0.2,
    random_state = 42
)

scaler = StandardScaler()

X_train_scaler = scaler.fit_transform(X_train)
X_test_scaler = scaler.transform(X_test)

model = GaussianNB()

model.fit(X_train_scaler, y_train)

new_data = pd.DataFrame({
    "Gender": [1, 0, 1],               # 1 = Male, 0 = Female
    "Age": [35, 42, 23],               
    "EstimatedSalary": [20000, 85000, 50000]
})

new_data_scaler = scaler.transform(new_data)

print("Prediction: ", model.predict(new_data_scaler))

print("Probability: ")
probability = True
print(model.predict_proba(new_data_scaler))

y_pred = model.predict(X_test_scaler)

print("Accuracy: ", accuracy_score(y_test, y_pred))

print("Confusion matix: ")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("Classification report: ")
print(classification_report(y_test, y_pred))