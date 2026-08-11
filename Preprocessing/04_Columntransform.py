import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv("dataset/bank-full.csv", sep=";")
print(df.head())

print("\n", df.isnull().sum())

print("\n", df.duplicated().sum())


num_cols = df.select_dtypes(include= "number").columns
cat_cols = df.select_dtypes(include= ["object"]).columns


df[cat_cols] = df[cat_cols].replace("unknown", pd.NA)

print("\n", df.isnull().sum())

missing_per = df.isnull().mean() * 100
drop_cols = missing_per[missing_per > 80].index
print("\n", drop_cols)

df = df.drop('poutcome', axis = 1)

#num_cols = df.select_dtypes(include= "number").columns
cat_cols = df.select_dtypes(include= ["object"]).columns

for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

print("\n", df.isnull().sum())


X = df.drop('y', axis= 1)
y = df["y"]

num_cols = X.select_dtypes(include=["int64","float64"]).columns
cat_cols = X.select_dtypes(include=["object"]).columns

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size= 0.2,
    random_state= 42,
    stratify= y
)

preprocessor = ColumnTransformer(
    transformers = [
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(), cat_cols)
    ]
)

X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

model = KNeighborsClassifier(n_neighbors=17, weights= "uniform", leaf_size= 2)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("\nAccuracy score:",accuracy_score(y_test, y_pred))