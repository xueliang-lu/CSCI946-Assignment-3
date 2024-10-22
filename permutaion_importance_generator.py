


import pandas as pd
import joblib
from sklearn.inspection import permutation_importance
import logging
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
    logging.info(f"Loading data from {file_path}")
    data = pd.read_csv(file_path)
    X = data.drop(columns=drop_columns)  # Features
    y = data['label']  # Label
    logging.info(f"Data loaded: {data.shape[0]} rows and {data.shape[1]} columns")
    return X, y

# Load the model
logging.info("Loading the LightGBM model")
best_lgbm = joblib.load('alex_lgbm_model.joblib')

# Load the validation datasets
val_data_path_1 = "data/features/val_eva02_large_patch14_448.mim_m38m_ft_in22k_in1k.csv"
val_data_path_2 = "data/features/v2_eva02_large_patch14_448.mim_m38m_ft_in22k_in1k.csv"

# Load and preprocess the validation datasets
logging.info("Loading validation dataset 1")
X_val_1, y_val_1 = load_and_drop_columns(val_data_path_1)

logging.info("Loading validation dataset 2")
X_val_2, y_val_2 = load_and_drop_columns(val_data_path_2)

# Step 1: Compute permutation importance for validation set 1
logging.info("Computing permutation importance for validation set 1")
start_time_val1 = time.time()
perm_importance_val1 = permutation_importance(best_lgbm, X_val_1, y_val_1, n_repeats=10, random_state=42, n_jobs=-1)
end_time_val1 = time.time()
logging.info(f"Permutation importance for validation set 1 computed in {end_time_val1 - start_time_val1:.2f} seconds.")

# Step 2: Compute permutation importance for validation set 2
logging.info("Computing permutation importance for validation set 2")
start_time_val2 = time.time()
perm_importance_val2 = permutation_importance(best_lgbm, X_val_2, y_val_2, n_repeats=10, random_state=42, n_jobs=-1)
end_time_val2 = time.time()
logging.info(f"Permutation importance for validation set 2 computed in {end_time_val2 - start_time_val2:.2f} seconds.")

# Step 3: Create DataFrames to save the results
importance_val1_df = pd.DataFrame({
    'feature': X_val_1.columns,
    'importance_mean': perm_importance_val1.importances_mean,
    'importance_std': perm_importance_val1.importances_std
})

importance_val2_df = pd.DataFrame({
    'feature': X_val_2.columns,
    'importance_mean': perm_importance_val2.importances_mean,
    'importance_std': perm_importance_val2.importances_std
})

# Step 4: Save the results to CSV files
importance_val1_df.to_csv('permutation_importance_val1.csv', index=False)
importance_val2_df.to_csv('permutation_importance_val2.csv', index=False)

logging.info("Permutation importance saved to CSV files.")


