import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import os


import warnings
warnings.filterwarnings('ignore')


train_df = pd.read_csv(r'e:\Study\HW\ML\kaggle\ml-2024-f\main_data\train_final.csv')
test_df = pd.read_csv(r'e:\Study\HW\ML\kaggle\ml-2024-f\train_final.csv')


if 'ID' not in test_df.columns:
    raise ValueError("Test data must contain an 'ID' column.")


test_ids = test_df['ID'].copy()


test_df.drop('ID', axis=1, inplace=True)


train_df.replace('?', 'Missing', inplace=True)
test_df.replace('?', 'Missing', inplace=True)


train_df.columns = train_df.columns.str.strip()
test_df.columns = test_df.columns.str.strip()


train_df['income>50K'] = train_df['income>50K'].astype(int)



numerical_features = ['age', 'fnlwgt', 'education.num', 'capital.gain',
                      'capital.loss', 'hours.per.week']
categorical_features = ['workclass', 'education', 'marital.status', 'occupation',
                        'relationship', 'race', 'sex', 'native.country']


X = train_df.drop('income>50K', axis=1)
y = train_df['income>50K']


X_test = test_df


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


fold_accuracies = []
fold_aurocs = []

print("\nPerforming Stratified 5-Fold Cross-Validation with Random Forest...\n")


fold = 1
for train_index, val_index in skf.split(X, y):
    print(f"Fold {fold}")
    X_train_fold, X_val_fold = X.iloc[train_index].copy(), X.iloc[val_index].copy()
    y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]

    
    label_encoders = {}
    for col in categorical_features:
        le = LabelEncoder()
        X_train_fold[col] = le.fit_transform(X_train_fold[col])
        
        X_val_fold[col] = X_val_fold[col].map(lambda s: '<unknown>' if s not in le.classes_ else s)
        le.classes_ = np.append(le.classes_, '<unknown>')
        X_val_fold[col] = le.transform(X_val_fold[col])
        label_encoders[col] = le

    
    rf_clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        n_jobs=-1
    )

    
    rf_clf.fit(X_train_fold, y_train_fold)

    
    val_preds = rf_clf.predict(X_val_fold)
    val_proba = rf_clf.predict_proba(X_val_fold)[:, 1]

    
    val_acc = accuracy_score(y_val_fold, val_preds)
    val_auroc = roc_auc_score(y_val_fold, val_proba)
    print(f"Validation Accuracy: {val_acc:.4f}, AUROC: {val_auroc:.4f}\n")

    fold_accuracies.append(val_acc)
    fold_aurocs.append(val_auroc)

    fold += 1


print("Cross-Validation Results:")
print(f"Mean Accuracy: {np.mean(fold_accuracies):.4f}, Std: {np.std(fold_accuracies):.4f}")
print(f"Mean AUROC: {np.mean(fold_aurocs):.4f}, Std: {np.std(fold_aurocs):.4f}")


print("\nTraining on full training data...")


label_encoders_full = {}
for col in categorical_features:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    if col in X_test.columns:
        
        X_test[col] = X_test[col].map(lambda s: '<unknown>' if s not in le.classes_ else s)
        le.classes_ = np.append(le.classes_, '<unknown>')
        X_test[col] = le.transform(X_test[col])
    label_encoders_full[col] = le


rf_clf_full = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)


rf_clf_full.fit(X, y)


test_preds = rf_clf_full.predict(X_test)
test_proba = rf_clf_full.predict_proba(X_test)[:, 1]



output_df = pd.DataFrame({
    'ID': test_ids,
    'Prediction': test_proba
})


output_path = r'e:\Study\HW\ML\kaggle\code submission\results\random_forest\predictions.csv'
output_df.to_csv(output_path, index=False)
print(f"\nPredictions saved to {output_path}")
