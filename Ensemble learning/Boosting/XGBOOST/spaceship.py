import pandas as pd
train = pd.read_csv("dataset/spaceship_train.csv")
test = pd.read_csv("dataset/spaceship_test.csv")

test_ids = test["PassengerId"]

train = train.drop(["PassengerId", "Name", "Cabin"], axis=1)
test = test.drop(["PassengerId","Name", "Cabin"], axis=1)

missing_per = train.isnull().mean() * 100
drop_cols = missing_per[missing_per > 80].index


miss_per = test.isnull().mean() * 100
drop_col = miss_per[miss_per > 80].index


num_cols = train.select_dtypes(include=["int64", "float64"]).columns
cat_cols = train.select_dtypes(include=["object"]).columns

# Numeric
train[num_cols] = train[num_cols].fillna(train[num_cols].median())
test[num_cols] = test[num_cols].fillna(train[num_cols].median())


for col in cat_cols:
    train[col] = train[col].fillna(train[col].mode()[0])
    test[col] = test[col].fillna(train[col].mode()[0])


train_encoded = pd.get_dummies(train, columns = cat_cols, drop_first =True)
test_encoded = pd.get_dummies(test, columns = cat_cols, drop_first =True)
train_encoded, test_encoded = train_encoded.align(
    test_encoded,
    join="left",
    axis=1,
    fill_value=0
)


X = train_encoded.drop("Transported", axis =1)
y = train_encoded["Transported"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X,y,
    test_size = 0.2,
    random_state = 42,
    stratify=y
)

from xgboost import XGBClassifier

model = XGBClassifier(
    random_state=42,
    eval_metric="logloss"
)

from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score

param = {
    "n_estimators":[100,200,300,500],
    "learning_rate":[0.01,0.03,0.05,0.1,0.2],
    "max_depth":[3,4,5,6,7,8],
    "subsample":[0.7,0.8,0.9,1.0],
    "colsample_bytree":[0.7,0.8,0.9,1.0],
    "min_child_weight":[1,3,5],
    "gamma":[0,0.1,0.3]
}

random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param,
    n_iter=5,
    cv=3,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1
)

random_search.fit(X_train, y_train)

best_model = random_search.best_estimator_
print(best_model)



y_pred = best_model.predict(X_test)
score = accuracy_score(y_test, y_pred)
print(score)
'''
from sklearn.model_selection import cross_val_score

cv_score = cross_val_score(
    best_model,
    X,
    y,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

print(f"Cross Validation Accuracy: {cv_score.mean():.4f}")
'''
best_model.fit(X, y)

test_encoded = test_encoded.drop(columns=["Transported"], errors="ignore")
test_pred = best_model.predict(test_encoded)

submission = pd.DataFrame({
    "PassengerId": test_ids,
    "Transported": test_pred
})
submission["Transported"] = submission["Transported"].astype(bool)
submission.to_csv("submission.csv", index=False)

