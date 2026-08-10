import pandas as pd

train = pd.read_csv("dataset/train.csv")
test = pd.read_csv("dataset/test.csv")

print("\n",train.head())
print("\n",train.info())
print("\n",train.columns)

test_id = test["PassengerId"]
train = train.drop(["PassengerId",'Name'], axis =1)
test = test.drop(["PassengerId",'Name'], axis = 1)

cat_cols = train.select_dtypes(include = ["object"]).columns

num_cols = train.select_dtypes(include = ["int64", "float64"]).columns

train["Cabin"] = train["Cabin"].str[0]
test["Cabin"] = test["Cabin"].str[0]

train[num_cols] = train[num_cols].fillna(train[num_cols].median())
test[num_cols] = test[num_cols].fillna(train[num_cols].median())


for col in cat_cols:
    train[col] = train[col].fillna(train[col].mode()[0])
    test[col] = test[col].fillna(train[col].mode()[0])

train_encoded = pd.get_dummies(train, columns=cat_cols, drop_first=True)
test_encoded = pd.get_dummies(test, columns=cat_cols, drop_first=True)

train, test = train.align(test, join="left", axis=1, fill_value=0)
print(train.head())


X = train_encoded.drop("Transported", axis =1)
y= train_encoded["Transported"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42, stratify = y)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

xgb = XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42
)

lgbm = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=-1,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

cat = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    loss_function="Logloss",
    random_state=42,
    verbose=0
)

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

test_pred = SoftVoting.predict(test_encoded)
submission = pd.DataFrame({
    "PassengerId": test_id,
    "Transported": test_pred
})

submission.to_csv("Submission.csv", index = False)