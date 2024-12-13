import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier
import optuna
import os
import warnings
from lightgbm import early_stopping, log_evaluation

warnings.filterwarnings('ignore')


train_path = r'e:\Study\HW\ML\kaggle\ml-2024-f\main_data\train_final.csv'
test_path = r'e:\Study\HW\ML\kaggle\ml-2024-f\main_data\test_final.csv'
output_path = r'e:\Study\HW\ML\kaggle\ml-2024-f\main_data\lgbm_predictions_optuna.csv'


train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)


if 'ID' not in test_df.columns:
    raise ValueError("Test data must contain an 'ID' column.")

test_ids = test_df['ID'].copy()
test_df.drop('ID', axis=1, inplace=True)


train_df.replace('?', 'Missing', inplace=True)
test_df.replace('?', 'Missing', inplace=True)


train_df.columns = train_df.columns.str.strip()
test_df.columns = test_df.columns.str.strip()


if 'income>50K' not in train_df.columns:
    raise ValueError("Train data must contain 'income>50K' as a target column.")

train_df['income>50K'] = train_df['income>50K'].astype(int)




categorical_features = ['workclass', 'education', 'marital.status', 'occupation',
                        'relationship', 'race', 'sex', 'native.country']
numerical_features = ['age', 'fnlwgt', 'education.num', 'capital.gain',
                      'capital.loss', 'hours.per.week']


train_df_cat = train_df.copy()
test_df_cat = test_df.copy()


for col in categorical_features:
    le = LabelEncoder()
    le.fit(train_df_cat[col])

    
    test_df_cat[col] = test_df_cat[col].apply(lambda x: x if x in le.classes_ else '<unknown>')
    if '<unknown>' not in le.classes_:
        le.classes_ = np.append(le.classes_, '<unknown>')

    train_df_cat[col] = le.transform(train_df_cat[col])
    test_df_cat[col] = le.transform(test_df_cat[col])

X_train_cat = train_df_cat.drop('income>50K', axis=1)
y_train_cat = train_df_cat['income>50K']


X_test_cat = test_df_cat


def objective(trial):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 600),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 15, 100),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-2, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-2, 10.0, log=True),
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = []

    for train_idx, val_idx in skf.split(X_train_cat, y_train_cat):
        X_tr, X_val = X_train_cat.iloc[train_idx], X_train_cat.iloc[val_idx]
        y_tr, y_val = y_train_cat.iloc[train_idx], y_train_cat.iloc[val_idx]

        model = LGBMClassifier(**param)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric="auc",
            callbacks=[early_stopping(stopping_rounds=50), log_evaluation(0)]
        )

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
best_params.update({'random_state': 42})


final_model = LGBMClassifier(**best_params)
final_model.fit(X_train_cat, y_train_cat)


train_proba = final_model.predict_proba(X_train_cat)[:, 1]
train_preds = (train_proba >= 0.5).astype(int)
train_acc = accuracy_score(y_train_cat, train_preds)
train_auroc = roc_auc_score(y_train_cat, train_proba)
print(f"\nTraining Accuracy: {train_acc:.4f}, AUROC: {train_auroc:.4f}")


test_proba = final_model.predict_proba(X_test_cat)[:, 1]
print("\nFinal test data does not contain labels. Predictions generated without evaluation.")


output_df = pd.DataFrame({
    'ID': test_ids,
    'Prediction': test_proba
})

os.makedirs(os.path.dirname(output_path), exist_ok=True)
output_df.to_csv(output_path, index=False)
print(f"\nPredictions saved to {output_path}")
