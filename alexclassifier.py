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


# param_grid = {
#     'num_leaves': 31,
#     'max_depth': 10,
#     'learning_rate': 0.1,
#     'n_estimators': 200,
#     'force_col_wise': True,
#     'random_state':42,
#     # 'device': ['gpu'],
# }
# param_grid = {
#     'num_leaves': 15,  
#     'max_depth': 7,  
#     'learning_rate': 0.1,
#     'n_estimators': 200,  # Setting to 200 and still using early stopping
#     'min_data_in_leaf': 500,
#     'lambda_l1': 0.1,
#     'lambda_l2': 0.1,
#     'random_state': 42,
# }

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
# # Evaluate on the internal validation set
y_internal_pred = lgbm.predict(X_internal_val)
# y_internal_proba = lgbm.predict_proba(X_internal_val)[:, 1] 
# y_pred_proba_1 = lgbm.predict_proba(X_val_1)[:, 1]
# y_pred_proba_2 = lgbm.predict_proba(X_val_2)[:, 1]
# # Function to plot ROC curve
# def plot_roc_curve(y_true, y_proba, dataset_name):
#     fpr, tpr, _ = roc_curve(y_true, y_proba)
#     roc_auc = auc(fpr, tpr)
    
#     plt.figure()
#     plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
#     plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')  # Diagonal line
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title(f'ROC Curve - {dataset_name}')
#     plt.legend(loc='lower right')
#     plt.grid()
#     plt.savefig(f'roc_curve_{dataset_name}.png')  # Save the figure
#     plt.close()  # Close the figure to avoid display

# # Plot ROC curves
# plot_roc_curve(y_internal_val, y_internal_proba, 'Internal Validation Set')
# plot_roc_curve(y_val_1, y_pred_proba_1, 'Validation Set 1')
# plot_roc_curve(y_val_2, y_pred_proba_2, 'Validation Set 2')
# print("Internal Validation Set Results:")
# print(classification_report(y_internal_val, y_internal_pred))
# print("Accuracy:", accuracy_score(y_internal_val, y_internal_pred))

# # Evaluate on the first external validation set
y_pred_1 = lgbm.predict(X_val_1)
# print("Validation Set 1 Results:")
# print(classification_report(y_val_1, y_pred_1))
# print("Accuracy:", accuracy_score(y_val_1, y_pred_1))

# # Evaluate on the second external validation set
y_pred_2 = lgbm.predict(X_val_2)
# print("Validation Set 2 Results:")
# print(classification_report(y_val_2, y_pred_2))
# print("Accuracy:", accuracy_score(y_val_2, y_pred_2))

# Save the best model to a file
model_filename = 'alex_lgbm_model.joblib'
joblib.dump(lgbm, model_filename)
print(f"Model saved to {model_filename}")

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

