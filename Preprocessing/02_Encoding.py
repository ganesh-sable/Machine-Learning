from sklearn.preprocessing import LabelEncoder
import pandas as pd

df = pd.read_csv("dataset/iris.csv")

df_label = df.copy()

le = LabelEncoder()

df_label["species_encoded"] = le.fit_transform(df_label['species'])
print(df_label.head())

df_encoded = pd.get_dummies(df_label, columns=['species'])
print(df_encoded)
