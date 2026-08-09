import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    balanced_accuracy_score
)

from catboost import CatBoostClassifier

train = pd.read_csv("dataset/Smartphone_Addiction_train.csv")
test = pd.read_csv("dataset/Smartphone_Addiction_test.csv")

print("Train Shape:", train.shape)
print("Test Shape:", test.shape)

print(train.head())

print(train.isnull().sum())

print(train.dtypes)

print(train["addicted_label"].value_counts())


test_ids = test["id"]

train = train.drop("id", axis=1)
test = test.drop("id", axis=1)


missing_per = train.isnull().mean() * 100

drop_cols = missing_per[missing_per > 80].index.tolist()

print("\nColumns having more than 80% missing values:")
print(drop_cols)


# Drop those columns from both train and test
train = train.drop(columns=drop_cols)
test = test.drop(columns=drop_cols)


X = train.drop("addicted_label", axis=1)
y = train["addicted_label"]


num_cols = X.select_dtypes(include=["number"]).columns.tolist()

cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

print("\nNumerical Columns:")
print(num_cols)

print("\nCategorical Columns:")
print(cat_cols)

for col in num_cols:

    median_value = X[col].median()

    X[col] = X[col].fillna(median_value)

for col in cat_cols:

    X[col] = X[col].fillna("Unknown")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

model = CatBoostClassifier(iterations=500,
    learning_rate=0.05,
    depth=7,
    l2_leaf_reg=5,
    loss_function="Logloss",
    eval_metric="Accuracy",
    random_seed=42,
    verbose=100,
    early_stopping_rounds=50
)

model.fit( X_train, y_train,
    cat_features=cat_cols,
    eval_set=(X_test, y_test),
    use_best_model=True
)


y_pred = model.predict(X_test).ravel()


accuracy = accuracy_score(y_test, y_pred)

balanced_acc = balanced_accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

print("Balanced Accuracy:", balanced_acc)


print("\nClassification Report:")
print(classification_report(y_test, y_pred))


print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

feature_importance = pd.DataFrame({

    "Feature": X.columns,

    "Importance": model.feature_importances_

})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)


print(feature_importance)


X_test_final = test.copy()

for col in num_cols:

    median_value = X[col].median()

    X_test_final[col] = X_test_final[col].fillna(median_value)


for col in cat_cols:

    X_test_final[col] = X_test_final[col].fillna("Unknown")

test_pred = model.predict(X_test_final).ravel()

submission = pd.DataFrame({
    "id": test_ids,
    "addicted_label": test_pred.astype(int)

})

submission.to_csv(
    "Smartphone_Addiction_submission.csv", index=False )
