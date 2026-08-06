from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, recall_score
import pandas as pd

df = pd.read_csv("dataset/Churn_Modelling.csv")

df = df.drop(columns=['RowNumber', 'CustomerId', 'Surname'])


cat_cols = df.select_dtypes(include= ["object"]).columns

df = pd.get_dummies(df, columns= cat_cols, drop_first= False)

X = df.drop('Exited', axis=1)
y = df['Exited']

X_train, X_test, y_train, y_test = train_test_split(
    X,y,
    test_size = 0.2,
    random_state = 42,
    stratify= y
)

model = RandomForestClassifier(n_estimators=200, random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy: ",accuracy_score(y_test, y_pred))

print("Confusion matrix: ")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("Classification report: ")
print(classification_report(y_test, y_pred))
