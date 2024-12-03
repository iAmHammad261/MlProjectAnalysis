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


print(df.head)

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

# Split features and target
X = df.drop(columns=["churn"])
y = df["churn"]

# Drop less relevant columns
cols_to_drop = [
    'preferredlogindevice_Computer', 'preferredlogindevice_Mobile Phone', 'preferredlogindevice_Phone',
    'preferredpaymentmode_CC', 'preferredpaymentmode_COD', 'preferredpaymentmode_Cash on Delivery',
    'preferredpaymentmode_Credit Card', 'preferredpaymentmode_Debit Card', 'preferredpaymentmode_E wallet',
    'preferredpaymentmode_UPI', 'preferedordercat_Fashion', 'preferedordercat_Grocery',
    'preferedordercat_Laptop & Accessory', 'preferedordercat_Mobile', 'preferedordercat_Mobile Phone',
    'preferedordercat_Others'
]
X.drop(cols_to_drop, axis=1, inplace=True)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Handle class imbalance using SMOTE
sm = SMOTE(sampling_strategy=1, random_state=1)
X_train_s, y_train_s = sm.fit_resample(X_train, y_train.ravel())

# Define the XGBClassifier model with best parameters
model = XGBClassifier(subsample=0.5, scale_pos_weight=1, n_estimators=150, min_child_weight=1,
                      max_depth=10, learning_rate=0.3, gamma=0, colsample_bytree=0.9)

# Perform cross-validation
scoring = ['accuracy', 'precision', 'recall', 'f1']
cv_results = cross_validate(model, X_train_s, y_train_s, cv=5, scoring=scoring, return_train_score=True)

# Print cross-validation results
print("Cross-Validation Results:")
print(f"Accuracy: {round(cv_results['test_accuracy'].mean(), 3) * 100}%")
print(f"Precision: {round(cv_results['test_precision'].mean(), 3) * 100}%")
print(f"Recall: {round(cv_results['test_recall'].mean(), 3) * 100}%")
print(f"F1-Score: {round(cv_results['test_f1'].mean(), 3) * 100}%")

# Extract metrics for visualization
metrics_data = {
    "Accuracy": cv_results['test_accuracy'].mean() * 100,
    "Precision": cv_results['test_precision'].mean() * 100,
    "Recall": cv_results['test_recall'].mean() * 100,
    "F1-Score": cv_results['test_f1'].mean() * 100
}

# Prepare a DataFrame for heatmap visualization
cv_metrics_df = pd.DataFrame([metrics_data])

# Visualize the cross-validation metrics as a heatmap
plt.figure(figsize=(8, 4))
sns.heatmap(cv_metrics_df, annot=True, fmt=".2f", cmap="Blues", cbar=True)
plt.title("Cross-Validation Metrics (Mean of 5-Fold CV)", fontsize=16)
plt.xlabel("Metric", fontsize=12)
plt.ylabel("CV Results", fontsize=12)
plt.show()

# Train the model
model.fit(X_train_s, y_train_s)

# Evaluate on test set
y_pred = model.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred) * 100
precision = precision_score(y_test, y_pred, zero_division=1) * 100
recall = recall_score(y_test, y_pred) * 100
f1 = f1_score(y_test, y_pred) * 100
conf_matrix = confusion_matrix(y_test, y_pred)

# Normalize the confusion matrix in percentage form
conf_matrix_percentage = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

# Print evaluation metrics
print("\nModel Evaluation Metrics:")
print(f"Accuracy: {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1-Score: {f1:.3f}")

# Visualize the normalized confusion matrix in percentage form using Seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_percentage, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
plt.title("Normalized Confusion Matrix (Percentage)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()



# Create a dictionary of the evaluation metrics
metrics = {
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1-Score": f1
}

#Create a seaborn barplot
sns.set(style="whitegrid")
plt.figure(figsize=(8, 6))
ax = sns.barplot(x=list(metrics.keys()), y=list(metrics.values()), palette="Blues_d")

# Add titles and labels
ax.set_title("Model Evaluation Metrics", fontsize=16)
ax.set_xlabel("Metric", fontsize=12)
ax.set_ylabel("Score", fontsize=12)

# Display the values inside the bars
for p in ax.patches:
    ax.annotate(f'{p.get_height():.3f}',
                (p.get_x() + p.get_width() / 2., p.get_height() / 2),  # Position text in the middle of the bar
                ha='center', va='center', fontsize=12, color='white', xytext=(0, 0), textcoords='offset points')

plt.show()

# Save the trained model
with open('../SavedModel/xgb_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Save column names
columns = {'data_columns': [col.lower() for col in X.columns]}
with open("../SavedModel/columns.json", "w") as f:
    f.write(json.dumps(columns))

print("\nModel saved in 'SavedModel/xgb_model.pkl'")
print("Json saved in 'SavedModel/columns.json'")
