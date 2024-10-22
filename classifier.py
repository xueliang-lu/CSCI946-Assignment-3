import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV

import joblib  # Import joblib for saving the model
import lightgbm as lgb


from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc

import matplotlib.pyplot as plt
import numpy as np

def load_and_drop_columns(file_path, drop_columns=['Unnamed: 0', 'path', 'label']):
    """
    Load a CSV file and drop the specified columns.

    Parameters:
    file_path (str): The path to the CSV file.
    drop_columns (list): A list of columns to drop. Default is ['Unnamed: 0', 'path', 'label'].

    Returns:
    X (pd.DataFrame): The DataFrame with the dropped columns.
    y (pd.Series): The label column that was dropped.
    """
    data = pd.read_csv(file_path)
    X = data.drop(columns=drop_columns)  # Features
    y = data['label']  # Label
    return X, y

train_data_path = "data/features/train_eva02_large_patch14_448.mim_m38m_ft_in22k_in1k.csv"
val_data_path_1 = "data/features/val_eva02_large_patch14_448.mim_m38m_ft_in22k_in1k.csv"
val_data_path_2 = "data/features/v2_eva02_large_patch14_448.mim_m38m_ft_in22k_in1k.csv"
# Load and procondacess the datasets
X_train_full, y_train_full = load_and_drop_columns(train_data_path)
X_val_1, y_val_1 = load_and_drop_columns(val_data_path_1)
X_val_2, y_val_2 = load_and_drop_columns(val_data_path_2)


# Split the training data into training and internal validation sets for hyperparameter tuning
X_train, X_internal_val, y_train, y_internal_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)


# Initialize the LightGBM classifier
lgbm = lgb.LGBMClassifier()
# lgbm.fit(X_train, y_train)
lgbm.fit(
    X_train, y_train, 
    eval_set=[(X_internal_val, y_internal_val)], 
    callbacks=[
        lgb.early_stopping(stopping_rounds=10,verbose=True),
    ],  # Stops training if no improvement after 10 rounds
)

# Save the best model to a file
model_filename = 'alex_lgbm_model.joblib'
joblib.dump(lgbm, model_filename)
print(f"Model saved to {model_filename}")

## Alternatively, you can use the following code to fine tune hyperparameter use GridSearchCV
# # Define a parameter grid for GridSearchCV
# param_grid = {
#     'num_leaves': [15, 31],        # Range of number of leaves
#     'max_depth': [7, 10, 15],      # Different max depth values
#     'learning_rate': [0.01, 0.1],  # Test different learning rates
#     'n_estimators': [100, 200],    # Number of boosting iterations
#     'min_data_in_leaf': [100, 500],  # Minimum data in one leaf
#     'lambda_l1': [0.0, 0.1],       # L1 regularization
#     'lambda_l2': [0.0, 0.1],       # L2 regularization
#     'random_state': [42]           # Fixed random state for reproducibility
# }

# # Initialize the LightGBM classifier
# lgbm = lgb.LGBMClassifier()

# # Set up the GridSearchCV with 5-fold cross-validation using internal validation data
# grid_search = GridSearchCV(
#     estimator=lgbm, 
#     param_grid=param_grid, 
#     cv=5,  # 5-fold cross-validation
#     scoring='accuracy',  # Use accuracy as the evaluation metric
#     verbose=2,  # Show progress during grid search
#     n_jobs=-1   # Use all available cores for computation
# )

# # Fit the model using GridSearchCV with internal validation data
# grid_search.fit(
#     X_train, y_train, 
#     eval_set=[(X_internal_val, y_internal_val)], 
#     eval_metric='multi_logloss',  # Metric for multiclass classification
#     early_stopping_rounds=10,     # Early stopping after 10 rounds of no improvement
#     verbose=True
# )

# # Best parameters from the grid search
# print("Best parameters found by GridSearchCV:")
# print(grid_search.best_params_)

# # Best score from the grid search
# print("Best accuracy score during GridSearchCV:")
# print(grid_search.best_score_)

# # Save the best model
# best_model = grid_search.best_estimator_
# joblib.dump(best_model, 'lightgbm_best_model.joblib')
# print("Model saved as 'lightgbm_best_model.joblib'")





# Evaluate on the internal validation set
y_internal_pred = lgbm.predict(X_internal_val)

# Evaluate on the first external validation set
y_pred_1 = lgbm.predict(X_val_1)

# Evaluate on the second external validation set
y_pred_2 = lgbm.predict(X_val_2)
# Save results to a text file
with open('alex_validation_results.txt', 'w') as file:
    # Internal Validation Results
    file.write("Internal Validation Set Results:\n")
    file.write(classification_report(y_internal_val, y_internal_pred))
    file.write(f"Accuracy: {accuracy_score(y_internal_val, y_internal_pred):.4f}\n\n")

    # External Validation Set 1 Results
    file.write("Validation Set 1 Results:\n")
    file.write(classification_report(y_val_1, y_pred_1))
    file.write(f"Accuracy: {accuracy_score(y_val_1, y_pred_1):.4f}\n\n")

    # External Validation Set 2 Results
    file.write("Validation Set 2 Results:\n")
    file.write(classification_report(y_val_2, y_pred_2))
    file.write(f"Accuracy: {accuracy_score(y_val_2, y_pred_2):.4f}\n")

print("Validation results saved to 'validation_results.txt'")

