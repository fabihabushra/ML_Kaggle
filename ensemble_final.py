import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder
import os
import warnings

warnings.filterwarnings('ignore')


train_path = r'e:\Study\HW\ML\kaggle\ml-2024-f\main_data\train_final.csv'
test_path = r'e:\Study\HW\ML\kaggle\ml-2024-f\main_data\test_final.csv'
output_path = r'e:\Study\HW\ML\kaggle\ml-2024-f\main_data\ensemble_predictions_optuna.csv'


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

X_train = train_df_cat.drop('income>50K', axis=1)
y_train = train_df_cat['income>50K']


X_test = test_df_cat


cat_params = {
    'iterations': 351,
    'depth': 4,
    'learning_rate': 0.15896864973143476,
    'l2_leaf_reg': 0.17662809103861252,
    'border_count': 110,
    'random_seed': 42,
    'verbose': 0,
    'cat_features': categorical_features
}

lgbm_params = {
    'n_estimators': 216,
    'max_depth': 7,
    'learning_rate': 0.05225513931333601,
    'num_leaves': 63,
    'min_child_samples': 19,
    'subsample': 0.8289919289739973,
    'colsample_bytree': 0.5835573172773423,
    'reg_alpha': 0.07842097032962848,
    'reg_lambda': 0.8055232718714938,
    'random_state': 42
}


cat_model = CatBoostClassifier(**cat_params)
cat_model.fit(X_train, y_train)


lgbm_model = LGBMClassifier(**lgbm_params)
lgbm_model.fit(X_train, y_train)



cat_test_proba = cat_model.predict_proba(X_test)[:, 1]
lgbm_test_proba = lgbm_model.predict_proba(X_test)[:, 1]

ensemble_test_proba = 0.5 * cat_test_proba + 0.5 * lgbm_test_proba


output_df = pd.DataFrame({
    'ID': test_ids,
    'Prediction': ensemble_test_proba
})

os.makedirs(os.path.dirname(output_path), exist_ok=True)
output_df.to_csv(output_path, index=False)
print(f"\nEnsemble predictions saved to {output_path}")
