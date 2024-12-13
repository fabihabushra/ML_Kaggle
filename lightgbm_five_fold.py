import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier
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

# Prepare the data
X = train_df.drop('income>50K', axis=1)
y = train_df['income>50K']


X_test = test_df

# LightGBM requires categorical features to be of type 'category'
for col in categorical_features:
    X[col] = X[col].astype('category')
    X_test[col] = X_test[col].astype('category')

# Initialize StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Lists to store results
fold_accuracies = []
fold_aurocs = []

print("\nPerforming Stratified 5-Fold Cross-Validation with LightGBM...\n")

# Cross-validation
fold = 1
for train_index, val_index in skf.split(X, y):
    print(f"Fold {fold}")
    X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
    y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]

    # Initialize LGBMClassifier
    lgbm_clf = LGBMClassifier(
        objective='binary',
        boosting_type='gbdt',
        learning_rate=0.1,
        n_estimators=100,
        num_leaves=31,
        max_depth=-1,
        random_state=42,
        n_jobs=-1,
    )

    # Train the model
    lgbm_clf.fit(
        X_train_fold,
        y_train_fold,
        eval_set=[(X_val_fold, y_val_fold)],
        eval_metric='auc',
        categorical_feature=categorical_features,
    )

    # Predict on validation data
    val_proba = lgbm_clf.predict_proba(X_val_fold)[:, 1]
    val_preds = (val_proba >= 0.5).astype(int)

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

# Initialize LGBMClassifier
lgbm_clf_full = LGBMClassifier(
    objective='binary',
    boosting_type='gbdt',
    learning_rate=0.1,
    n_estimators=100,
    num_leaves=31,
    max_depth=-1,
    n_jobs=-1,
)

# Train the model on full training data
lgbm_clf_full.fit(
    X,
    y,
    categorical_feature=categorical_features,
)

# Predict on test data
test_proba = lgbm_clf_full.predict_proba(X_test)[:, 1]
test_preds = (test_proba >= 0.5).astype(int)


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
output_path = r'e:\Study\HW\ML\kaggle\code submission\results\lightgbm\predictions.csv'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
output_df.to_csv(output_path, index=False)
print(f"\nPredictions saved to {output_path}")
