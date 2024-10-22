
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score
from sklearn.inspection import permutation_importance
import logging
import time
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

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

'''
Plot model feature importance and permutation importance for each val sets
'''

# Step 1: Make predictions for validation set 1
logging.info("Making predictions for validation set 1")
y_pred_val1 = best_lgbm.predict(X_val_1)
baseline_acc_1 = accuracy_score(y_val_1, y_pred_val1)
logging.info(f"Baseline accuracy for validation set 1: {baseline_acc_1:.4f}")

# Step 2: Make predictions for validation set 2
logging.info("Making predictions for validation set 2")
y_pred_val2 = best_lgbm.predict(X_val_2)
baseline_acc_2 = accuracy_score(y_val_2, y_pred_val2)
logging.info(f"Baseline accuracy for validation set 2: {baseline_acc_2:.4f}")

# Step 3: Get model feature importance
model_importance = best_lgbm.feature_importances_
model_importance_df = pd.DataFrame({
    'feature': X_val_1.columns,
    'importance': model_importance
})

# Get top 50 model features
top_50_features_model = model_importance_df.nlargest(50, 'importance')


# Calculate relative importance
max_importance = model_importance_df['importance'].max()
top_50_features_model['relative_importance'] = top_50_features_model['importance'] / max_importance

# Step 4: Plot top 50 model feature relative importances
plt.figure(figsize=(12, 6))
plt.barh(top_50_features_model['feature'], top_50_features_model['relative_importance'], color='skyblue')
plt.xlabel('Relative Importance')
plt.title('Top 50 Model Feature Relative Importances')
plt.gca().invert_yaxis()  # Invert y-axis for better readability
plt.show()

# Step 5: Load the CSV files for both validation sets
perm_importance_val1 = pd.read_csv('permutation_importance_val1.csv')
perm_importance_val2 = pd.read_csv('permutation_importance_val2.csv')

# Step 6: Calculate relative importance for both validation sets
perm_importance_val1['relative_importance'] = perm_importance_val1['importance_mean'] / baseline_acc_1
perm_importance_val2['relative_importance'] = perm_importance_val2['importance_mean'] / baseline_acc_2

# Get top 50 model features
top_50_features_v1 = perm_importance_val1.nlargest(50, 'importance_mean')

# Plot top 50 model feature importances
# Plotting feature importances
plt.figure(figsize=(12, 6))
plt.barh(range(len(top_50_features_v1)), top_50_features_v1['relative_importance'], color='skyblue')
plt.yticks(range(len(top_50_features_v1)), top_50_features_v1['feature'])
plt.xlabel('Importance')
plt.title('Top 50 Feature Importances (Validation Set 1)')
plt.gca().invert_yaxis()  # Invert y-axis for better readability
plt.show()


# Get top 50 model features
top_50_features_v2 = perm_importance_val2.nlargest(50, 'importance_mean')

# Plot top 50 model feature importances
# Plotting feature importances
plt.figure(figsize=(12, 6))
plt.barh(range(len(top_50_features_v2)), top_50_features_v2['relative_importance'], color='skyblue')
plt.yticks(range(len(top_50_features_v2)), top_50_features_v2['feature'])
plt.xlabel('Importance')
plt.title('Top 50 Feature Importances (Validation Set 2)')
plt.gca().invert_yaxis()  # Invert y-axis for better readability
plt.show()



'''
Clutering importance
'''

# Step 1: Merge the two DataFrames based on 'feature'
merged_df = pd.merge(perm_importance_val1[['feature', 'relative_importance']], 
                     perm_importance_val2[['feature', 'relative_importance']], 
                     on='feature', 
                     suffixes=('_val1', '_val2'))

# Step 2: Use relative importance for clustering
X = merged_df[['relative_importance_val1', 'relative_importance_val2']].values

# Step 3: Apply K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=0).fit(X)

# Step 4: Add the cluster labels to the merged DataFrame
merged_df['cluster'] = kmeans.labels_

# Step 5: Plot the clusters
plt.figure(figsize=(8, 6))
plt.scatter(merged_df['relative_importance_val1'], merged_df['relative_importance_val2'], 
            c=merged_df['cluster'], cmap='viridis', s=100)
plt.xlabel('Relative Importance in Val1')
plt.ylabel('Relative Importance in Val2')
plt.title('KMeans Clustering of Feature Importance (Val1 vs Val2)')
plt.colorbar(label='Cluster')
plt.show()

'''
Compare distribution between val sets under feature 170, which is the highest importance feature of both sets
'''
# Step 1: Extract Feature 170
feature_index = 170 
feature_name = X_val_1.columns[feature_index]  # Get the name of feature 170

feature_170_val1 = X_val_1.iloc[:, feature_index]  # Feature 170 from validation set 1
feature_170_val2 = X_val_2.iloc[:, feature_index]  # Feature 170 from validation set 2

# Step 2: Plot the distributions
plt.figure(figsize=(12, 6))

# Histogram for validation set 1
plt.subplot(1, 2, 1)
plt.hist(feature_170_val1, bins=30, alpha=0.7, color='blue', edgecolor='black')
plt.title(f'Distribution of {feature_name} (Validation Set 1)')
plt.xlabel('Feature 170 Value')
plt.ylabel('Frequency')

# Histogram for validation set 2
plt.subplot(1, 2, 2)
plt.hist(feature_170_val2, bins=30, alpha=0.7, color='orange', edgecolor='black')
plt.title(f'Distribution of {feature_name} (Validation Set 2)')
plt.xlabel('Feature 170 Value')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

'''
Compare significance between val sets under feature 170, which is the highest importance feature of both sets
'''
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Load the validation datasets
val_data_path_1 = "data/features/val_eva02_large_patch14_448.mim_m38m_ft_in22k_in1k.csv"
val_data_path_2 = "data/features/v2_eva02_large_patch14_448.mim_m38m_ft_in22k_in1k.csv"

# Load and preprocess the validation datasets
X_val_1, y_val_1 = load_and_drop_columns(val_data_path_1)
X_val_2, y_val_2 = load_and_drop_columns(val_data_path_2)

# Extract feature 170
feature_index = 170  
feature_170_val1 = X_val_1.iloc[:, feature_index]
feature_170_val2 = X_val_2.iloc[:, feature_index]

# Step 1: Check normality using the Shapiro-Wilk test
shapiro_val1 = stats.shapiro(feature_170_val1)
shapiro_val2 = stats.shapiro(feature_170_val2)

print(f"Shapiro-Wilk Test for Val1: Statistic={shapiro_val1.statistic:.4f}, p-value={shapiro_val1.pvalue:.4f}")
print(f"Shapiro-Wilk Test for Val2: Statistic={shapiro_val2.statistic:.4f}, p-value={shapiro_val2.pvalue:.4f}")

# Step 2: If both distributions are normal, perform the t-test
if shapiro_val1.pvalue > 0.05 and shapiro_val2.pvalue > 0.05:
    t_statistic, p_value = stats.ttest_ind(feature_170_val1, feature_170_val2)
    print(f"T-test Statistic: {t_statistic:.4f}, p-value: {p_value:.4f}")

    # Step 3: Interpretation
    alpha = 0.05  # significance level
    if p_value < alpha:
        print("Reject the null hypothesis: The two distributions are significantly different.")
    else:
        print("Fail to reject the null hypothesis: The two distributions are not significantly different.")
else:
    print("One or both distributions are not normally distributed. Consider using a non-parametric test.")

# Step 2: Perform Mann-Whitney U test
u_statistic, p_value_mw = stats.mannwhitneyu(feature_170_val1, feature_170_val2, alternative='two-sided')

# Step 3: Interpretation
print(f"Mann-Whitney U Test Statistic: {u_statistic:.4f}, p-value: {p_value_mw:.4f}")

alpha = 0.05  # significance level
if p_value_mw < alpha:
    print("Reject the null hypothesis: The two distributions are significantly different.")
else:
    print("Fail to reject the null hypothesis: The two distributions are not significantly different.")
