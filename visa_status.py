#import packages

import numpy as np
import pandas as pd # used to store and manipulate the data in table form
from datetime import datetime # helps to convert the date strings into the real date objects

data = {
    "application_date": ["2024-01-01", None, "2024-03-10"],
    "decision_date": ["2024-02-01", "2024-03-20", None],
    "country": ["India", None, "UK"],
    "visa_type": ["Student", "Tourist", "Work"],
    "processing_office": ["Delhi", "New York", None]
}

df = pd.DataFrame(data)
print("Original DataFrame with Missing Values:\n", df)

miss= df.isnull().sum()
# isnull-> finds the missing values
# sum() -> counts them in colm-wise
print("\nMissing values count:\n", miss)

# mode()[0]-> returns the most frequ date in the colm
# fillna()-> replaces the missing values inside the df
df["application_date"].fillna(df["application_date"].mode()[0], inplace=True)
df["decision_date"].fillna(df["decision_date"].mode()[0], inplace=True)

df["country"].fillna("Unknown", inplace=True)
df["processing_office"].fillna("Unknown", inplace=True)

df["application_date"] = pd.to_datetime(df["application_date"])
df["decision_date"] = pd.to_datetime(df["decision_date"])

print(df)

df["processing_days"] = (df["decision_date"] - df["application_date"]).dt.days
print("\nAfter calculating processing days:\n", df)

df_encoded = pd.get_dummies(df, columns=["country", "visa_type", "processing_office"])
print("\nEncoded DataFrame:\n", df_encoded)
# converts the text to numeric columns
# one-hot encoding - as the ML model requires the numeric data

