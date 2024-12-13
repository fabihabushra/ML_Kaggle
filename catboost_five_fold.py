import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
import os

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# Load data
train_df = pd.read_csv(r'e:\Study\HW\ML\kaggle\ml-2024-f\main_data\train_final.csv')
test_df = pd.read_csv(r'e:\Study\HW\ML\kaggle\ml-2024-f\test_final.csv')

# Ensure 'ID' column is present in test data
if 'ID' not in test_df.columns:
    raise ValueError("Test data must contain an 'ID' column.")

# Save 'ID' column for output
test_ids = test_df['ID'].copy()

# Remove 'ID' column from test data for processing
test_df.drop('ID', axis=1, inplace=True)

# Replace '?' with 'Missing' in both datasets
train_df.replace('?', 'Missing', inplace=True)
test_df.replace('?', 'Missing', inplace=True)

# Strip leading/trailing spaces from column names
train_df.columns = train_df.columns.str.strip()
test_df.columns = test_df.columns.str.strip()

# Map target variable to 0 and 1 in the training data
train_df['income>50K'] = train_df['income>50K'].astype(int)


# Identify numerical and categorical features
numerical_features = ['age', 'fnlwgt', 'education.num', 'capital.gain',
                      'capital.loss', 'hours.per.week']
categorical_features = ['workclass', 'education', 'marital.status', 'occupation',
                        'relationship', 'race', 'sex', 'native.country']

# Ensure that categorical features are of type 'object' and numerical features are numerical
for col in categorical_features:
    train_df[col] = train_df[col].astype('str')
    test_df[col] = test_df[col].astype('str')

for col in numerical_features:
    train_df[col] = pd.to_numeric(train_df[col], errors='coerce')
    test_df[col] = pd.to_numeric(test_df[col], errors='coerce')

# Prepare the data
X = train_df.drop('income>50K', axis=1)
y = train_df['income>50K']


X_test = test_df

# Initialize StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Lists to store results
fold_accuracies = []
fold_aurocs = []

print("\nPerforming Stratified 5-Fold Cross-Validation with CatBoost...\n")

# Import CatBoost
from catboost import CatBoostClassifier, Pool

# Define your best hyperparameters
cat_best_params = {
    'border_count': 52,
    'depth': 5,
    'iterations': 240,
    'l2_leaf_reg': 2.0,
    'learning_rate': 0.12,
    'verbose': 0
}

# Cross-validation
fold = 1
for train_index, val_index in skf.split(X, y):
    print(f"Fold {fold}")
    X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
    y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]
    
    # Create Pool objects for training and validation data
    train_pool = Pool(data=X_train_fold, label=y_train_fold, cat_features=categorical_features)
    val_pool = Pool(data=X_val_fold, label=y_val_fold, cat_features=categorical_features)
    
    # Initialize the CatBoostClassifier with the new hyperparameters
    cat_clf = CatBoostClassifier(**cat_best_params)
    cat_clf.fit(train_pool)
    
    # Predict on validation data
    val_preds = cat_clf.predict(val_pool)
    val_proba = cat_clf.predict_proba(val_pool)[:, 1]
    
    # Compute metrics
    val_acc = accuracy_score(y_val_fold, val_preds)
    val_auroc = roc_auc_score(y_val_fold, val_proba)
    print(f"Validation Accuracy: {val_acc:.4f}, AUROC: {val_auroc:.4f}\n")
    
    fold_accuracies.append(val_acc)
    fold_aurocs.append(val_auroc)
    
    fold += 1

# Print cross-validation results
print("Cross-Validation Results:")
print(f"Mean Accuracy: {np.mean(fold_accuracies):.4f}, Std: {np.std(fold_accuracies):.4f}")
print(f"Mean AUROC: {np.mean(fold_aurocs):.4f}, Std: {np.std(fold_aurocs):.4f}")

# Now, train on the full training data
print("\nTraining on full training data...")

# Create a Pool object for the full training data
full_train_pool = Pool(data=X, label=y, cat_features=categorical_features)

# Initialize the CatBoostClassifier with the new hyperparameters
cat_clf_full = CatBoostClassifier(**cat_best_params)
cat_clf_full.fit(full_train_pool)

# Predict probabilities on test data
# Create a Pool object for the test data
test_pool = Pool(data=X_test, cat_features=categorical_features)

test_proba = cat_clf_full.predict_proba(test_pool)[:, 1]
test_preds = cat_clf_full.predict(test_pool)


# Create the output DataFrame
output_df = pd.DataFrame({
    'ID': test_ids,
    'Prediction': test_proba
})

# Save the predictions to a CSV file
output_path = r'e:\Study\HW\ML\kaggle\code submission\results\catboost\predictions.csv'  
os.makedirs(os.path.dirname(output_path), exist_ok=True)
output_df.to_csv(output_path, index=False)
print(f"\nPredictions saved to {output_path}")
