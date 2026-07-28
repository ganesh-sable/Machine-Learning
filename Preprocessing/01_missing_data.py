import pandas as pd

df = pd.read_csv("dataset/titanic.csv")

titanic = df.copy()
print()
print(titanic.info())

print(titanic.isnull().sum())

print(titanic.columns)
titanic['Age'] = titanic['Age'].fillna(titanic['Age'].median())


titanic["Cabin_Known"] = titanic["Cabin"].notnull().astype(int)
titanic.drop(columns=["Cabin"], inplace=True)

duplicates_before = titanic.duplicated().sum()
titanic.drop_duplicates(inplace=True)

titanic["Age"] = titanic["Age"].astype(int)
titanic["Survived"] = titanic["Survived"].astype("category")
titanic["Sex"] = titanic["Sex"].astype("category")
titanic["Name"] = titanic["Name"].astype("category")
titanic["Ticket"] = titanic["Ticket"].astype("category")
titanic["Embarked"] = titanic["Embarked"].astype("category")

print("\nMissing Values After Cleaning:")
print(titanic.isnull().sum())

print("\nDataset info after cleaning:")
titanic.info()

print("\nFirst 5 Rows of Cleaned Dataset:")
print(titanic.head())

titanic.to_csv("Clean_titanic.csv", index =False)

