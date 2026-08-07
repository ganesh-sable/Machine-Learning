import pandas as pd
from xgboost import XGBRegressor
from xgboost import plot_importance
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import r2_score

df = pd.read_csv("dataset/House_price.csv")

test = pd.read_csv("dataset/House_price_test.csv")

test_ids = test["Id"]

df = df.drop("Id", axis=1)
test = test.drop("Id", axis=1)

print(df.head())
print()
print(df.describe())
print()
print(df.info())
print()
print(df.isnull().sum())

print()
missing_percent = df.isnull().mean() * 100
test_miss = test.isnull().mean() * 100
drop_cols = missing_percent[missing_percent > 80].index
drop_test = test_miss[missing_percent > 80].index

print(drop_cols)
print(drop_test)

df = df.drop(drop_cols, axis = 1)
test = test.drop(drop_test, axis = 1)


num_cols = df.select_dtypes(include=["int64", "float64"]).columns
test_num_cols = test.select_dtypes(include=["int64", "float64"]).columns
cat_cols = df.select_dtypes(include=["object"]).columns


print(cat_cols)

df[num_cols] = df[num_cols].fillna(df[num_cols].median())
test[test_num_cols] = test[test_num_cols].fillna(test[test_num_cols].median())

for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])
    test[col] = test[col].fillna(test[col].mode()[0])


df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)
test_encoded = pd.get_dummies(test, columns=cat_cols, drop_first=True) 

X = df_encoded.drop("SalePrice", axis=1)
y = df_encoded["SalePrice"]

test_encoded = test_encoded.reindex(columns=X.columns, fill_value=0)
                                    
X_train, X_test, y_train, y_test = train_test_split(
    X,y,
    test_size=0.2,
    random_state=42
)

model = XGBRegressor(random_state =42)

param = {
    "n_estimators": [100, 200, 300],
    "learning_rate": [0.01, 0.1, 0.2],
    "max_depth":[1,3,5],
    "subsample": [0.8, 1.0]
}


random_search = RandomizedSearchCV(estimator=model,
    param_distributions=param,
    n_iter=10,
    cv=5,
    scoring="r2",
    return_train_score=False
)

random_search.fit(X_train, y_train)

best_model = random_search.best_estimator_

y_pred = best_model.predict(X_test)
print("\nR2 score:", r2_score(y_test, y_pred))

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": best_model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print(importance_df)
plot_importance(best_model)

test_pred = best_model.predict(test_encoded)

submission = pd.DataFrame({
    "Id": test_ids,
    "SalePrice": test_pred
})

submission.to_csv("submission.csv", index=False)

