import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score

train = pd.read_csv("dataset/spaceship_train.csv")
test = pd.read_csv("dataset/spaceship_test.csv")

print("\n",train.head())
print("\n",train.info())
print("\n",train.columns)

test_id = test["PassengerId"]
train = train.drop(["PassengerId", 'Cabin', 'Name'], axis =1)
test = test.drop(["PassengerId", 'Cabin', 'Name'], axis = 1)


cat_cols = train.select_dtypes(include = ["object"]).columns
print("\n",cat_cols)

num_cols = train.select_dtypes(include = ["int64", "float64"]).columns
print("\n",num_cols)



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

X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42, stratify = y)

model = LGBMClassifier(random_state=42)

param = {
    "n_estimators": [100, 200, 300, 500, 700],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "max_depth": [-1, 5, 8, 10, 15],
    "num_leaves": [15, 31, 50, 70, 100],
    "min_child_samples": [10, 20, 30, 50],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "reg_alpha": [0, 0.01, 0.1, 1],
    "reg_lambda": [0, 0.1, 1, 10]
}
Random_search = RandomizedSearchCV(
    estimator= model,
    param_distributions= param,
    n_iter = 10,
    cv = 5,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1
)

Random_search.fit(X_train, y_train)

best_model = Random_search.best_estimator_

y_pred = best_model.predict(X_test)
score = accuracy_score(y_test, y_pred)
print(score)

best_model.fit(X, y)

test_encoded = test_encoded.drop(columns=["Transported"], errors="ignore")
test_result = best_model.predict(test_encoded)
submission = pd.DataFrame({
    "PassengerId" : test_id,
    "Transported": test_result
})


submission["Transported"] = submission["Transported"].astype(bool)
submission.to_csv("spaceship_submission.csv", index=False)