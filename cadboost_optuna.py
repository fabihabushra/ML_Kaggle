import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostClassifier
import optuna
import os
import warnings

warnings.filterwarnings('ignore')


train_path = r'e:\Study\HW\ML\kaggle\ml-2024-f\main_data\cleaned_train_data.csv'


train_df = pd.read_csv(train_path)
train_df.replace('?', 'Missing', inplace=True)
train_df.columns = train_df.columns.str.strip()


if 'income>50K' not in train_df.columns:
    raise ValueError("Train data must contain 'income>50K' as a target column.")
train_df['income>50K'] = train_df['income>50K'].astype(int)



categorical_features = ['workclass', 'education', 'marital.status', 'occupation',
                        'relationship', 'race', 'sex', 'native.country']
numerical_features = ['age', 'fnlwgt', 'education.num', 'capital.gain',
                      'capital.loss', 'hours.per.week']


train_df_cat = train_df.copy()


for col in categorical_features:
    le = LabelEncoder()
    le.fit(train_df_cat[col])

    if '<unknown>' not in le.classes_:
        le.classes_ = np.append(le.classes_, '<unknown>')

    train_df_cat[col] = le.transform(train_df_cat[col])

X_train_cat = train_df_cat.drop('income>50K', axis=1)
y_train_cat = train_df_cat['income>50K']


def objective(trial):
    
    param = {
        'iterations': trial.suggest_int('iterations', 100, 600),
        'depth': trial.suggest_int('depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-2, 10.0, log=True),
        'border_count': trial.suggest_int('border_count', 32, 128),
        'verbose': 0,
        'cat_features': categorical_features
    }

    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = []

    for train_index, val_index in skf.split(X_train_cat, y_train_cat):
        X_tr, X_val = X_train_cat.iloc[train_index], X_train_cat.iloc[val_index]
        y_tr, y_val = y_train_cat.iloc[train_index], y_train_cat.iloc[val_index]

        model = CatBoostClassifier(**param)
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=50, use_best_model=True)

        val_proba = model.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, val_proba)
        auc_scores.append(val_auc)

    
    return np.mean(auc_scores)


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=25, show_progress_bar=True)

print("\nBest Trial:")
print(study.best_trial)
print("\nBest Hyperparameters:")
print(study.best_trial.params)

best_params = study.best_trial.params
best_params.update({
    'random_seed': 42,
    'verbose': 0,
    'cat_features': categorical_features
})


final_model = CatBoostClassifier(**best_params)
final_model.fit(X_train_cat, y_train_cat)


train_proba = final_model.predict_proba(X_train_cat)[:, 1]
train_preds = (train_proba >= 0.5).astype(int)
train_acc = accuracy_score(y_train_cat, train_preds)
train_auroc = roc_auc_score(y_train_cat, train_proba)
print(f"\nTraining Accuracy: {train_acc:.4f}, AUROC: {train_auroc:.4f}")
