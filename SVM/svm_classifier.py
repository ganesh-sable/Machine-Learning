from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import pandas as pd

df = pd.read_csv("dataset/iris.csv")

df['species'] = df['species'].map({'setosa': 0, 'versicolor': 1, 'virginica': 2})

X = df.drop(columns=["species"])
y = df["species"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size = 0.2,
    random_state = 42
)

scaler = StandardScaler()

X_train_scaler = scaler.fit_transform(X_train)
X_test_scaler = scaler.transform(X_test)

model = SVC(probability=True)

model.fit(X_train_scaler, y_train)

new_data = pd.DataFrame({
    'sepal_length': [5.1, 5.9, 6.7],
    'sepal_width':  [3.5, 3.0, 3.1],
    'petal_length': [1.4, 4.2, 5.6],
    'petal_width':  [0.2, 1.5, 2.4]
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