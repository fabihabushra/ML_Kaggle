import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import os


import warnings
warnings.filterwarnings('ignore')


train_df = pd.read_csv(r'e:\Study\HW\ML\kaggle\ml-2024-f\main_data\train_final.csv')
test_df = pd.read_csv(r'e:\Study\HW\ML\kaggle\ml-2024-f\main_data\test_final.csv')


if 'ID' in test_df.columns:
    test_ids = test_df['ID'].copy()
    test_df.drop('ID', axis=1, inplace=True)
else:
    test_ids = None


train_df.replace('?', 'Missing', inplace=True)
test_df.replace('?', 'Missing', inplace=True)


train_df.columns = train_df.columns.str.strip()
test_df.columns = test_df.columns.str.strip()


train_df['income>50K'] = train_df['income>50K'].astype(int)



numerical_features = ['age', 'fnlwgt', 'education.num', 'capital.gain',
                      'capital.loss', 'hours.per.week']
categorical_features = ['workclass', 'education', 'marital.status', 'occupation',
                        'relationship', 'race', 'sex', 'native.country']



train_df_encoded = pd.get_dummies(train_df, columns=categorical_features)
X = train_df_encoded.drop('income>50K', axis=1)
y = train_df_encoded['income>50K']


test_df_encoded = pd.get_dummies(test_df, columns=categorical_features)
X_test =  test_df_encoded


X_test = X_test.reindex(columns=X.columns, fill_value=0)



skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


fold_accuracies = []
fold_aurocs = []

print("\nPerforming Stratified 5-Fold Cross-Validation with AdaBoost...\n")


fold = 1
for train_index, val_index in skf.split(X, y):
    print(f"Fold {fold}")
    X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
    y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]

    
    ada_clf = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
        n_estimators=100,
        learning_rate=1.0,
        algorithm='SAMME.R',
    )

    
    ada_clf.fit(X_train_fold, y_train_fold)

    
    val_preds = ada_clf.predict(X_val_fold)
    val_proba = ada_clf.predict_proba(X_val_fold)[:, 1]

    
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


ada_clf_full = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
    n_estimators=100,
    learning_rate=1.0,
    algorithm='SAMME.R',
    random_state=42
)
ada_clf_full.fit(X, y)


test_preds = ada_clf_full.predict(X_test)
test_proba = ada_clf_full.predict_proba(X_test)[:, 1]



if test_ids is not None:
    output_df = pd.DataFrame({
        'ID': test_ids,
        'Prediction': test_proba
    })
else:
    output_df = pd.DataFrame({
        'Prediction': test_proba
    })


output_path = r'e:\Study\HW\ML\kaggle\ml-2024-f\main_data\predictions.csv' 
output_df.to_csv(output_path, index=False)
print(f"\nPredictions saved to {output_path}")
