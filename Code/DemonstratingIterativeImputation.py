import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestRegressor
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import pickle
import json
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_excel("../data/E_Commerce_Dataset.xlsx", sheet_name="E Comm")

# Drop unnecessary columns and preprocess
df.drop(columns="CustomerID", inplace=True)
df.columns = [col.lower() for col in df.columns]

print("Before Imputation")
tenure_column = df['tenure']
print(tenure_column)


print("------------------------------")

# Function to fill missing values and preprocess data
def fill_missing_values(df, random_state=None):
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

    numeric_imputer = SimpleImputer(strategy='mean')
    df[numeric_columns] = numeric_imputer.fit_transform(df[numeric_columns])

    for col in categorical_columns:
        if df[col].dtype == 'object':
            encoded_cols = pd.get_dummies(df[col], prefix=col)
            df = pd.concat([df.drop(col, axis=1), encoded_cols], axis=1)

    rf_imputer = IterativeImputer(estimator=RandomForestRegressor(random_state=random_state))
    df = pd.DataFrame(rf_imputer.fit_transform(df), columns=df.columns)
    return df

df = fill_missing_values(df, random_state=42)

print("After Imputation")
tenure_column = df['tenure']
print(tenure_column)





