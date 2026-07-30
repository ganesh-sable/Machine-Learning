from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import pandas as pd


df = pd.read_csv("dataset/diabetes.csv")
X = df[["Pregnancies","PlasmaGlucose","DiastolicBloodPressure","TricepsThickness","SerumInsulin","BMI","DiabetesPedigree"]]
y = df["Diabetic"]

X_train, X_test, y_train, y_test = train_test_split(
    X,y,
    test_size =0.3,
    random_state = 42
)

scaler = StandardScaler()
X_train_scaler = scaler.fit_transform(X_train)
X_test_scaler = scaler.transform(X_test)


model = LogisticRegression(max_iter = 1000)

model.fit(X_train_scaler, y_train)

new_data = pd.DataFrame({
    "Pregnancies":[3],
    "PlasmaGlucose":[114],
    "DiastolicBloodPressure":[53],
    "TricepsThickness":[7],
    "SerumInsulin":[227],
    "BMI":[20.97988119],
    "DiabetesPedigree":[0.190279084]
})

new_data_scaler = scaler.transform(new_data)

prediction = model.predict(new_data_scaler)

print(prediction)

print(model.predict_proba(new_data_scaler))

y_pred = model.predict(X_test_scaler)

accuracy = accuracy_score(y_test, y_pred)
print(accuracy)


cm = confusion_matrix(y_test, y_pred)
print(cm)

print("Precision: ",precision_score(y_test, y_pred))
print("Recall: ", recall_score(y_test, y_pred))
print("F1 Score: ", f1_score(y_test, y_pred))
print("Classification Report: ")
report = classification_report(y_test, y_pred)
print(report)