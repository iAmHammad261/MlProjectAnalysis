import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
import pickle

# Load the dataset
df = pd.read_excel("../data/E_Commerce_Dataset.xlsx", sheet_name="E Comm")

# Drop unnecessary columns
df.drop(columns="CustomerID", inplace=True)

# Standardize column names to lowercase
df.columns = [col.lower() for col in df.columns]

# Function to handle missing values and apply iterative imputation
def fill_missing_values(df, random_state=None):
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

    # Impute numeric columns
    numeric_imputer = SimpleImputer(strategy='mean')
    df[numeric_columns] = numeric_imputer.fit_transform(df[numeric_columns])

    # One-hot encode categorical columns
    for col in categorical_columns:
        encoded_cols = pd.get_dummies(df[col], prefix=col)
        df = pd.concat([df.drop(col, axis=1), encoded_cols], axis=1)

    # Apply iterative imputation
    rf_imputer = IterativeImputer(random_state=random_state)
    df = pd.DataFrame(rf_imputer.fit_transform(df), columns=df.columns)

    return df

# Fill missing values
df = fill_missing_values(df, random_state=42)

# Define features and target variable
X = df.drop(columns=["churn"])
y = df["churn"]

# Drop specific columns
cols_to_drop = [
    'preferredlogindevice_Computer', 'preferredlogindevice_Mobile Phone', 'preferredlogindevice_Phone',
    'preferredpaymentmode_CC', 'preferredpaymentmode_COD', 'preferredpaymentmode_Cash on Delivery', 'preferredpaymentmode_Credit Card',
    'preferredpaymentmode_Debit Card', 'preferredpaymentmode_E wallet', 'preferredpaymentmode_UPI',
    'preferedordercat_Fashion', 'preferedordercat_Grocery', 'preferedordercat_Laptop & Accessory',
    'preferedordercat_Mobile', 'preferedordercat_Mobile Phone', 'preferedordercat_Others'
]
X.drop(cols_to_drop, axis=1, inplace=True)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Handle class imbalance
sm = SMOTE(sampling_strategy=1, random_state=1)
X_train_s, y_train_s = sm.fit_resample(X_train, y_train.ravel())

# Define hyperparameter grid
param_grid = {
    'n_estimators': [50, 100, 150, 200],
    'learning_rate': [0.01, 0.1, 0.2, 0.3],
    'max_depth': [3, 5, 7, 10],
    'subsample': [0.5, 0.7, 0.9, 1.0],
    'colsample_bytree': [0.5, 0.7, 0.9, 1.0],
    'gamma': [0, 0.1, 0.3, 0.5],
    'min_child_weight': [1, 3, 5, 7],
    'scale_pos_weight': [1, 2, 5]
}

# Initialize and perform RandomizedSearchCV
xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')


random_search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_grid,
    n_iter=50,
    scoring='accuracy',
    cv=5,
    verbose=2,
    random_state=42,
    n_jobs=-1
)


random_search.fit(X_train_s, y_train_s)

# Get the best parameters and train the model
print("Best Parameters:", random_search.best_params_)
# best_model = random_search.best_estimator_

# # Evaluate on the test data
# accuracy = best_model.score(X_test, y_test)
# print(f"\nFinal Accuracy on Test Data: {accuracy * 100:.2f}%")

# Save the model
# with open('../SavedModel/xgb_best_model.pkl', 'wb') as f:
#     pickle.dump(best_model, f)
#
# print("\nModel with best parameters saved as 'SavedModel/xgb_best_model.pkl'")
