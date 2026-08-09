import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score

train = pd.read_csv("dataset/spaceship_train.csv")
test = pd.read_csv("dataset/spaceship_test.csv")

print(train.head())

print("\n", train.isnull().sum())

test_ids = test["PassengerId"]
train = train.drop(["PassengerId", "Name"], axis=1)
test = test.drop(["PassengerId","Name"], axis=1)

num_col = train.select_dtypes(include = ["int64", "float64"]).columns
cat_col = train.select_dtypes(include = ["object"]).columns

train[num_col] = train[num_col].fillna(train[num_col].median())
test[num_col] = test[num_col].fillna(train[num_col].median())

for col in cat_col:
    train[col] = train[col].fillna(train[col].mode()[0])
    test[col] = test[col].fillna(train[col].mode()[0])

X = train.drop("Transported", axis=1)
y = train["Transported"]



X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state = 42,
    stratify=y
)


cat_features = X_train.select_dtypes(
    include=["object", "bool"]
).columns.tolist()

model = CatBoostClassifier()

param_dist = {
    "iterations": [200, 500, 800, 1000],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "depth": [4, 6, 8, 10],
    "l2_leaf_reg": [1, 3, 5, 7, 9],
    "border_count": [32, 64, 128, 254],
    "bagging_temperature": [0, 1, 3, 5],
    "random_strength": [1, 2, 5, 10]
}

random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=20,
    cv=5,
    scoring="accuracy",
    random_state=42,
    n_jobs=-1,
    verbose=2
)

random_search.fit(
    X_train,
    y_train,
    cat_features=cat_features
)

best_model = random_search.best_estimator_

prediction = best_model.predict(X_test)

accuracy = accuracy_score(y_test, prediction)

print("Accuracy :", accuracy)

best_model.fit(X,y, cat_features=cat_features)
test_pre = best_model.predict(test)
submission = pd.DataFrame({
    "PassengerId": test_ids,
    "Transported": test_pre
})
submission["Transported"] = submission["Transported"].astype(bool)
submission.to_csv("submission.csv", index=False)