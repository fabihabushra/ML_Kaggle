import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
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


# Standardize numerical features (essential for SVM)
scaler = StandardScaler()
X[numerical_features] = scaler.fit_transform(X[numerical_features])
X_test[numerical_features] = scaler.transform(X_test[numerical_features])

# Initialize StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Lists to store results
fold_accuracies = []
fold_aurocs = []

print("\nPerforming Stratified 5-Fold Cross-Validation with SVM...\n")

# Cross-validation
fold = 1
for train_index, val_index in skf.split(X, y):
    print(f"Fold {fold}")
    X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
    y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]

    # Standardize numerical features within each fold
    scaler_fold = StandardScaler()
    X_train_fold[numerical_features] = scaler_fold.fit_transform(X_train_fold[numerical_features])
    X_val_fold[numerical_features] = scaler_fold.transform(X_val_fold[numerical_features])

    # Initialize SVM classifier
    svm_clf = SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        probability=True,
        random_state=42
    )

    # Train the model
    svm_clf.fit(X_train_fold, y_train_fold)

    # Predict on validation data
    val_preds = svm_clf.predict(X_val_fold)
    val_proba = svm_clf.predict_proba(X_val_fold)[:, 1]

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

# Standardize numerical features on full training data
scaler_full = StandardScaler()
X[numerical_features] = scaler_full.fit_transform(X[numerical_features])
X_test[numerical_features] = scaler_full.transform(X_test[numerical_features])

# Initialize SVM classifier
svm_clf_full = SVC(
    kernel='rbf',
    C=1.0,
    gamma='scale',
    probability=True,
)

# Train the model on full training data
svm_clf_full.fit(X, y)

# Predict on test data
test_preds = svm_clf_full.predict(X_test)
test_proba = svm_clf_full.predict_proba(X_test)[:, 1]


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

# Save the output to a CSV file
output_path = r'e:\Study\HW\ML\kaggle\code submission\results\svm\predictions.csv'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
output_df.to_csv(output_path, index=False)
print(f"\nPredictions saved to {output_path}")
