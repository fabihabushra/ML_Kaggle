import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import os

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# Load data
train_df = pd.read_csv(r'e:\Study\HW\ML\kaggle\ml-2024-f\main_data\train_final.csv')
test_df = pd.read_csv(r'e:\Study\HW\ML\kaggle\ml-2024-f\test_final.csv')

# Ensure 'ID' column is present in test data
if 'ID' in test_df.columns:
    test_ids = test_df['ID'].copy()
    test_df.drop('ID', axis=1, inplace=True)
else:
    test_ids = None

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

# One-Hot Encoding of Categorical Features (without combining datasets)
# Training data
train_df_encoded = pd.get_dummies(train_df, columns=categorical_features)
X = train_df_encoded.drop('income>50K', axis=1)
y = train_df_encoded['income>50K']

# Test data
test_df_encoded = pd.get_dummies(test_df, columns=categorical_features)
X_test = test_df_encoded

# Align the test data to have the same columns as training data
X_test = X_test.reindex(columns=X.columns, fill_value=0)

# Standardize numerical features (optional for XGBoost)
scaler = StandardScaler()
X[numerical_features] = scaler.fit_transform(X[numerical_features])
X_test[numerical_features] = scaler.transform(X_test[numerical_features])

# Initialize StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Lists to store results
fold_accuracies = []
fold_aurocs = []

print("\nPerforming Stratified 5-Fold Cross-Validation with XGBoost...\n")

# Cross-validation
fold = 1
for train_index, val_index in skf.split(X, y):
    print(f"Fold {fold}")
    X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
    y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]

    # Initialize XGBoost classifier
    xgb_clf = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )

    # Train the model
    xgb_clf.fit(X_train_fold, y_train_fold)

    # Predict on validation data
    val_preds = xgb_clf.predict(X_val_fold)
    val_proba = xgb_clf.predict_proba(X_val_fold)[:, 1]

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

# Train the model on full training data
xgb_clf_full = xgb.XGBClassifier(
    n_estimators=285,
    max_depth=8,
    learning_rate=0.07084844859190755,
    subsample=0.7599443886861021,
    colsample_bytree=0.6727299868828402,
    gamma=0.9170225492671691,
    reg_alpha=0.11531212520707879,
    reg_lambda=2.6238733012919457,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)




xgb_clf_full.fit(X, y)

# Predict on test data
test_preds = xgb_clf_full.predict(X_test)
test_proba = xgb_clf_full.predict_proba(X_test)[:, 1]


# Create the output DataFrame
if test_ids is not None:
    output_df = pd.DataFrame({
        'ID': test_ids,
        'Prediction': test_proba
    })
else:
    output_df = pd.DataFrame({
        'Prediction': test_proba
    })

# # Save the output to a CSV file
output_path = r'e:\Study\HW\ML\kaggle\code submission\results\xgboost\predictions.csv'
output_df.to_csv(output_path, index=False)
print(f"\nPredictions saved to {output_path}")
