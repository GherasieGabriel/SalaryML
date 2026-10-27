import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

columns = [
    "age", "workclass", "fnlwgt", "education",
    "education-num", "marital-status", "occupation",
    "relationship", "race", "sex", "capital-gain",
    "capital-loss", "hours-per-week", "native-country",
    "income"
]

train = pd.read_csv(
    "Input/adult.data",
    names=columns,
    skipinitialspace=True,
    na_values="?"
)

test = pd.read_csv(
    "Input/adult.test",
    names=columns,
    skipinitialspace=True,
    na_values="?",
    comment="|"
)

for df in (train, test):
    df["income"] = (
        df["income"]
        .astype("string")
        .str.strip()
        .str.replace(r"\.$", "", regex=True)
    )

df = train.copy()

print("Dimensiune:", df.shape)
print("\nTipuri de date:")
print(df.dtypes)

print("\nStatistici numerice:")
print(df.describe())

print("\nStatistici categorice:")
print(df.describe(include="object"))

print("\nValori lipsă:")
print(df.isna().sum().sort_values(ascending=False))

print("\nDuplicate:")
print(df.duplicated().sum())