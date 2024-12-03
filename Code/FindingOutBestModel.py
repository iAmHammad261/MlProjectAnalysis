import numpy as np  # Imports the NumPy library for numerical operations and array handling.
import pandas as pd  # Imports the pandas library for data manipulation and analysis.
import matplotlib.pyplot as plt  # Imports Matplotlib's pyplot for creating static visualizations.
# import seaborn as sns  # Imports Seaborn for statistical data visualization, built on top of Matplotlib.
# import plotly.express as px  # Imports Plotly Express for easy-to-use interactive visualizations.
# import missingno as msno  # Imports Missingno for visualizing missing data.
from sklearn.pipeline import Pipeline  # Imports Pipeline for creating machine learning workflows.
from sklearn.linear_model import LogisticRegression  # Imports LogisticRegression for classification tasks.
from sklearn.ensemble import RandomForestClassifier, \
    RandomForestRegressor  # Imports RandomForestClassifier for ensemble-based classification.
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, \
    precision_score, f1_score, recall_score  # Imports metrics for evaluating model performance.
from sklearn.model_selection import train_test_split  # Imports function to split data into training and testing sets.
from sklearn.experimental import enable_iterative_imputer  # Enables the experimental IterativeImputer in scikit-learn.
from sklearn.impute import IterativeImputer, SimpleImputer  # Imports imputers to handle missing data.
from sklearn.preprocessing import StandardScaler, OneHotEncoder  # Imports tools for data preprocessing (scaling and encoding).
from sklearn.compose import ColumnTransformer  # Imports ColumnTransformer for applying different preprocessing steps to different columns.
from sklearn.model_selection import GridSearchCV  # Imports GridSearchCV for hyperparameter tuning.
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn import model_selection
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_validate
from sklearn import model_selection
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV, RidgeClassifierCV
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
import pandas as pd
import seaborn as sns  # To use for plotting the graph





pd.set_option('display.max_columns',None)
df = pd.read_excel("../data/E_Commerce_Dataset.xlsx", sheet_name="E Comm")

df.drop(columns="CustomerID", inplace=True)

df.columns = [col.lower() for col in df.columns]


def fill_missing_values(df, random_state=None):
    # Step 1: Identify numeric and categorical columns
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()  # Include both string and category data

    # Step 2: Impute numeric columns
    numeric_imputer = SimpleImputer(strategy='mean')
    df[numeric_columns] = numeric_imputer.fit_transform(df[numeric_columns])

    # Step 3: Handle categorical columns
    for col in categorical_columns:
        if df[col].dtype == 'object':
            # Convert categorical column to one-hot encoded representation
            encoded_cols = pd.get_dummies(df[col], prefix=col)
            # Concatenate one-hot encoded columns
            df = pd.concat([df.drop(col, axis=1), encoded_cols], axis=1)

    # Step 4: Random Forest Iterative Imputer for the entire DataFrame
    rf_imputer = IterativeImputer(estimator=RandomForestRegressor(random_state=random_state))
    df = pd.DataFrame(rf_imputer.fit_transform(df), columns=df.columns)

    return df

# Call the function to fill missing values
df = fill_missing_values(df, random_state=42)

X = df.drop(columns=["churn"])
y = df["churn"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


sm = SMOTE(sampling_strategy = 1, random_state=1)
X_train_s, y_train_s = sm.fit_resample(X_train, y_train.ravel())


from sklearn.model_selection import GridSearchCV  # For hyperparameter tuning.
from sklearn.preprocessing import StandardScaler  # For feature scaling.
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay  # For evaluation metrics.
from xgboost import XGBClassifier

# Define models
models = [
    # Ensemble Models
    AdaBoostClassifier(),
    BaggingClassifier(),
    GradientBoostingClassifier(),
    RandomForestClassifier(),

    # Linear Models
    LogisticRegressionCV(),
    RidgeClassifierCV(),

    # Nearest Neighbor
    KNeighborsClassifier(),

    # XGBoost
    XGBClassifier()
]

# Metrics columns for evaluation
metrics_cols = ['model_name', 'test_accuracy', 'test_precision', 'test_recall', 'test_f1']

# Initialize lists to store evaluation metrics
model_name = []
test_accuracy = []
test_precision = []
test_recall = []
test_f1 = []

# Model evaluation
scoring = ['accuracy', 'precision', 'recall', 'f1']

# Cross-validation for all models
for model in models:
    print(f"Evaluating {model.__class__.__name__}...")
    cv_results = cross_validate(
        model, X, y, cv=5, scoring=scoring, return_train_score=True
    )
    model_name.append(model.__class__.__name__)
    test_accuracy.append(round(cv_results['test_accuracy'].mean(), 3) * 100)
    test_precision.append(round(cv_results['test_precision'].mean(), 3) * 100)
    test_recall.append(round(cv_results['test_recall'].mean(), 3) * 100)
    test_f1.append(round(cv_results['test_f1'].mean(), 3) * 100)

# Create DataFrame for metrics
metrics_data = [model_name, test_accuracy, test_precision, test_recall, test_f1]
metrics_dict = {n: m for n, m in zip(metrics_cols, metrics_data)}
model_metrics = pd.DataFrame(metrics_dict)

# Create a DataFrame for cross-validation metrics
cv_results_df = pd.DataFrame({
    'Model': model_name,
    'Accuracy (%)': test_accuracy,
    'Precision (%)': test_precision,
    'Recall (%)': test_recall,
    'F1 Score (%)': test_f1
})


# Set the Model column as the index
cv_results_df.set_index('Model', inplace=True)

# Plot the metrics using a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(cv_results_df, annot=True, fmt=".2f", cmap="YlGnBu", cbar=True, linewidths=0.5)

# Customize the heatmap
plt.title("Cross-Validation Metrics for Models", fontsize=16)
plt.xlabel("Metrics", fontsize=12)
plt.ylabel("Models", fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10, rotation=0)

# Display the heatmap
plt.show()


# Sort and display the DataFrame
model_metrics = model_metrics.sort_values('test_accuracy', ascending=False)
print(model_metrics)

# Initialize lists to store accuracy values
accuracy_values = []
precision_values = []
f1_values = []
recall_values = []


# Model evaluation and prediction
for model in models:
    print(f"Evaluating {model.__class__.__name__}...")

    # Fit the model on the training data
    model.fit(X_train_s, y_train_s)

    # Predict on the test data
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred) * 100
    accuracy_values.append(accuracy)

    # Calculate precision
    precision = precision_score(y_test, y_pred) * 100
    precision_values.append(precision)

    f1 = f1_score(y_test, y_pred) * 100
    recall = recall_score(y_test, y_pred) * 100

    f1_values.append(f1)
    recall_values.append(recall)

# Create DataFrame for the accuracy values
accuracy_df = pd.DataFrame({
    'Model': [model.__class__.__name__ for model in models],
    'Accuracy (%)': accuracy_values
})

# Create DataFrame for the precision values
precision_df = pd.DataFrame({
    'Model': [model.__class__.__name__ for model in models],
    'Precision (%)': precision_values
})

# Create DataFrames for F1 score and recall values
f1_df = pd.DataFrame({
    'Model': [model.__class__.__name__ for model in models],
    'F1 Score (%)': f1_values
})

# Create DataFrames for F1 score and recall values
f1_df = pd.DataFrame({
    'Model': [model.__class__.__name__ for model in models],
    'F1 Score (%)': f1_values
})

recall_df = pd.DataFrame({
    'Model': [model.__class__.__name__ for model in models],
    'Recall (%)': recall_values
})

# Plot accuracy using seaborn
plt.figure(figsize=(10, 14))
ax = sns.barplot(x='Accuracy (%)', y='Model', data=accuracy_df, palette='viridis')

# Annotate each bar with the actual accuracy value
for p in ax.patches:
    ax.annotate(f'{p.get_width():.2f}%',
                (p.get_x() + p.get_width() - 5, p.get_y() + p.get_height() / 2),
                ha='center', va='center', color='white', fontsize=12)

plt.title('Model Accuracy Comparison')
plt.show()

# Plot precision using seaborn (new figure)
plt.figure(figsize=(10, 6))  # Create a new figure for precision plot
ax2 = sns.barplot(x='Precision (%)', y='Model', data=precision_df, palette='viridis')

# Annotate each bar with the actual precision value
for p in ax2.patches:
    ax2.annotate(f'{p.get_width():.2f}%',
                 (p.get_x() + p.get_width() - 5, p.get_y() + p.get_height() / 2),
                 ha='center', va='center', color='white', fontsize=12)

plt.title('Model Precision Comparison')
plt.show()

plt.figure(figsize=(10, 6))  # Create a new figure for F1 score plot
ax1 = sns.barplot(x='F1 Score (%)', y='Model', data=f1_df, palette='viridis')

# Annotate each bar with the actual F1 score value
for p in ax1.patches:
    ax1.annotate(f'{p.get_width():.2f}%',
                 (p.get_x() + p.get_width() - 5, p.get_y() + p.get_height() / 2),
                 ha='center', va='center', color='white', fontsize=12)

plt.title('Model F1 Score Comparison')
plt.show()  # Ensure that the F1 score plot is displayed

# Plot Recall using seaborn (new figure)
plt.figure(figsize=(10, 6))  # Create a new figure for Recall plot
ax2 = sns.barplot(x='Recall (%)', y='Model', data=recall_df, palette='viridis')

# Annotate each bar with the actual recall value
for p in ax2.patches:
    ax2.annotate(f'{p.get_width():.2f}%',
                 (p.get_x() + p.get_width() - 5, p.get_y() + p.get_height() / 2),
                 ha='center', va='center', color='white', fontsize=12)

plt.title('Model Recall Comparison')
plt.show()  # Ensure that the Recall plot is displayed