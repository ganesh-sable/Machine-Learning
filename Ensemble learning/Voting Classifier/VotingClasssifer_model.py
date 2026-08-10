import pandas as pd

df = pd.read_csv("dataset/data.csv")

print(df.head())

missing_per = df.isnull().mean() * 100
drop_cols = missing_per[missing_per > 80].index

print(drop_cols)

df = df.drop(columns=drop_cols)

df = df.drop("id", axis=1)

print(df.duplicated().sum())

print(df.isnull().sum())
print(df.info())
num_cols = df.select_dtypes(include= "number").columns

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df["diagnosis"] = le.fit_transform(df["diagnosis"])

X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size= 0.2,
    random_state= 42,
    stratify= y
)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

xgb = XGBClassifier(random_state = 42, eval_metric="mlogloss")

lgbm = LGBMClassifier(random_state= 42)

cat = CatBoostClassifier(random_state= 42, verbose=0)

model = {
    "XGBOOST" : xgb,
    "LIGHTGBM" : lgbm,
    "CATBOOT" : cat
}

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

for name, model in model.items():
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"\n{model}")
    print("\nAccuracy: ", accuracy_score(y_test, y_pred))

    print("\nConfusion Matrix")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report")
    print(classification_report(y_test, y_pred))

from sklearn.ensemble import VotingClassifier

hardVoting = VotingClassifier(
    estimators=[
        ("xgb", xgb),
        ("lgbm", lgbm),
        ("cat", cat)
    ],
    voting = "hard"
)

hardVoting.fit(X_train, y_train)

y_pred = hardVoting.predict(X_test)
print("\nHard Voting Classifier Accuracy: ", accuracy_score(y_test, y_pred))


SoftVoting = VotingClassifier(
    estimators=[
        ("xgb", xgb),
        ("lgbm", lgbm),
        ("cat", cat)
    ],
    voting = "hard"
)

SoftVoting.fit(X_train, y_train)

y_pred = SoftVoting.predict(X_test)
print("\nSoft Voting Classifier Accuracy: ", accuracy_score(y_test, y_pred))