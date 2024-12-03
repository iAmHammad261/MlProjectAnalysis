import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.model_selection import train_test_split, cross_validate
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
pd.set_option('display.max_columns', None)
df = pd.read_excel("../data/E_Commerce_Dataset.xlsx", sheet_name="E Comm")

# Preprocess the dataset
df.drop(columns="CustomerID", inplace=True)
df.columns = [col.lower() for col in df.columns]

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

# Split dataset into features and target
X = df.drop(columns=["churn"])
y = df["churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Apply SMOTE to balance the dataset
sm = SMOTE(sampling_strategy=1, random_state=1)
X_train_s, y_train_s = sm.fit_resample(X_train, y_train.ravel())

# Define the XGBoost model
xgb_model = XGBClassifier(random_state=42)
print("Evaluating XGBoost Model...")

# Cross-validation for XGBoost
scoring = ['accuracy', 'precision', 'recall', 'f1']
cv_results = cross_validate(
    xgb_model, X, y, cv=5, scoring=scoring, return_train_score=True
)

# Display metrics
metrics_cols = ['test_accuracy', 'test_precision', 'test_recall', 'test_f1']
metrics_data = {
    metric: [round(cv_results[metric].mean(), 3) * 100]
    for metric in metrics_cols
}
model_metrics = pd.DataFrame(metrics_data, index=["XGBoost"])
print("\nModel Performance Metrics:")
print(model_metrics)

# Train final model for feature importance
final_model = XGBClassifier(random_state=42)
final_model.fit(X_train_s, y_train_s)

# Display feature importance (weight)
print("\nFeature Importance (Weight):")
feature_importance_weight = final_model.get_booster().get_score(importance_type="weight")
data_weight = pd.DataFrame(
    list(feature_importance_weight.items()), columns=["Feature", "Weight"]
).sort_values(by="Weight", ascending=False)
print(data_weight)

# Display feature importance (gain)
print("\nFeature Importance (Gain):")
feature_importance_gain = final_model.get_booster().get_score(importance_type="gain")
data_gain = pd.DataFrame(
    list(feature_importance_gain.items()), columns=["Feature", "Gain"]
).sort_values(by="Gain", ascending=False)
print(data_gain)

# Display feature importance (Weight) in a horizontal bar chart for all features
plt.figure(figsize=(15, 10))
sns.barplot(x='Weight', y='Feature', data=data_weight, palette='viridis')
plt.title('Feature Importance by Weight')
plt.xlabel('Feature Weight')
plt.ylabel('Feature')
# plt.subplots_adjust(left=0.5, right=0.10, top=0.9, bottom=0.1)  # Move the plot to the right
plt.show()

# Display feature importance (Gain) in a horizontal bar chart for all features
plt.figure(figsize=(12, 10))
sns.barplot(x='Gain', y='Feature', data=data_gain, palette='viridis')
plt.title('Feature Importance by Gain')
plt.xlabel('Feature Gain')
plt.ylabel('Feature')
# plt.subplots_adjust(left=0.5, right=0.10, top=0.9, bottom=0.1)  # Move the plot to the right
plt.show()
