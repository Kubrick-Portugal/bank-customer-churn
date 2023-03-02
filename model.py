
# %% --------------------------------------------------------------------------
# imports
# -----------------------------------------------------------------------------

import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import resample
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# %% --------------------------------------------------------------------------
# Set Random State
# -----------------------------------------------------------------------------
rng = np.random.RandomState(123)

# %% --------------------------------------------------------------------------
# load data
# -----------------------------------------------------------------------------
df = pd.read_csv(r'data\cleaned_data.csv')


# %% --------------------------------------------------------------------------
# Add new column
# -----------------------------------------------------------------------------
new_df = df
new_df["duration_open_months"] = new_df["duration_open"] / 30
new_df['net_monthly_transactions_duration'] = new_df['total_amount'] / new_df['duration_open_months']
new_df['net_monthly_transactions_duration'].replace([np.inf, -np.inf, np.nan], 0, inplace=True)
new_df['net_monthly_transactions_duration'] = new_df['net_monthly_transactions_duration'].apply(lambda x: 0.01 if x==0 else x)



# %% --------------------------------------------------------------------------
# 
# -----------------------------------------------------------------------------
new

# %% --------------------------------------------------------------------------
# 
# -----------------------------------------------------------------------------
new_df.hist(figsize=(20,20))
plt.show()


# %% --------------------------------------------------------------------------
# train test split
# -----------------------------------------------------------------------------
columns_to_keep = [
    'start_balance',
    'total_deposit',
    'num_deposits', 
    'total_withdrawal', 
    'num_withdrawals',
    'interest_rate',
    'inflation_expectation',
    'unemployment_rate',
    'consumer_sent', 
    'duration_open',
    'current_balance', 
    'balance_ratio',
    'normalized_significant_withdrawals',
    'balance_variance', 
    'consecutive_deficits'
]

print('I')

X = df[columns_to_keep]
print('II')
y = df['churned']


# %% --------------------------------------------------------------------------
# feature engineering
# -----------------------------------------------------------------------------
def age_group(age):
    if age >= 18 and age <= 25:
        return '18-25'
    elif age > 25 and age <= 40:
        return '25-40'
    elif age > 40 and age <= 55:
        return '40-55' 
    else:
        return '55+'


# %% --------------------------------------------------------------------------
# ohe
# -----------------------------------------------------------------------------
# from sklearn.preprocessing import OneHotEncoder
# # one hot encoding
# enc = OneHotEncoder(sparse=False)
# color_onehot = enc.fit_transform(X_train[['color']])
# #to print the encoded features for train data
# pd.DataFrame(color_onehot, columns=list(enc.categories_[0]))
# # tranform encoding for test data
# test_onehot = enc.transform(X_test[['color']])
# #to print the encoded features for train data
# pd.DataFrame(test_onehot, columns=list(enc.categories_[0]))


# %% --------------------------------------------------------------------------
# 
# -----------------------------------------------------------------------------
df_train, df_test = train_test_split(X, y, random_state=123, test_size=0.2, stratify=y)

# %% --------------------------------------------------------------------------
# 
# -----------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=123, test_size=0.2, stratify=y)

# %% --------------------------------------------------------------------------
# XGboost 1 plain
# -----------------------------------------------------------------------------
model_1 = xgb.XGBClassifier()
model_1.fit(X_train,y_train)
y_pred = model_1.predict(X_test)

# %% --------------------------------------------------------------------------
# explore XGboost 1
# -----------------------------------------------------------------------------

features = model_1.get_booster().feature_names
gain_importance = model_1.get_score(importance_type='gain')
weight_importance = model_1.get_score(importance_type='weight')
cover_importance = model_1.get_score(importance_type='cover')
total_gain_importance = model_1.get_score(importance_type='total_gain')
total_cover_importance = model_1.get_score(importance_type='total_cover')
print(pd.DataFrame(zip(features, gain_importance, weight_importance, cover_importance, total_gain_importance, total_cover_importance), columns=['feature', 'gain', 'weight', 'cover', 'total_gain', 'total_cover']).set_index('feature'))

print(classification_report(y_test, y_pred))

# %% --------------------------------------------------------------------------
# XGboost 2 - accounting for imbalance
# -----------------------------------------------------------------------------
scale = df['churned'].value_counts().tolist()
scale_neg = scale[0]
scale_pos = scale[1]
scale_pos_weight = (scale_neg / scale_pos)
model_2 = xgb.XGBClassifier(scale_pos_weight=scale_pos_weight)
model_2.fit(X_train,y_train)
y_pred = model_2.predict(X_test)


# %% --------------------------------------------------------------------------
# explore XGboost 2
# -----------------------------------------------------------------------------
features = model_2.get_booster().feature_names
importances = model_2.feature_importances_
print(pd.DataFrame(zip(features, importances), columns=['feature', 'importance']).set_index('feature').sort_values(['importance']))

print(classification_report(y_test, y_pred))

# %% --------------------------------------------------------------------------
# XGboost 3 -resampling
# -----------------------------------------------------------------------------
# Class count
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler()
X_resampled, y_resampled = ros.fit_resample(X_train, y_train)

model_3 = xgb.XGBClassifier()
model_3.fit(X_resampled, y_resampled)
y_pred = model_3.predict(X_test)


# %% --------------------------------------------------------------------------
# explore XGboost 3
# -----------------------------------------------------------------------------
features = model_3.get_booster().feature_names
importances = model_3.feature_importances_
print(pd.DataFrame(zip(features, importances), columns=['feature', 'importance']).set_index('feature').sort_values(['importance']))

print(classification_report(y_test, y_pred))


# %% --------------------------------------------------------------------------
# XGboost 4 -resampling
# -----------------------------------------------------------------------------
# Class count
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler()
X_resampled, y_resampled = rus.fit_resample(X_train, y_train)

model_4 = xgb.XGBClassifier()
model_4.fit(X_resampled, y_resampled)
y_pred = model_4.predict(X_test)


# %% --------------------------------------------------------------------------
# explore XGboost 4
# -----------------------------------------------------------------------------
features = model_4.get_booster().feature_names
importances = model_4.feature_importances_

print(pd.DataFrame(zip(features, importances), columns=['feature', 'importance']).set_index('feature').sort_values(['importance']))

print(classification_report(y_test, y_pred))

# %% --------------------------------------------------------------------------
# model 2
# -----------------------------------------------------------------------------

from sklearn.ensemble import RandomForestClassifier
model2 = RandomForestClassifier()
# model_1 = xgb.XGBClassifier()
model2.fit(X_train, y_train)
y_pred = model2.predict(X_test)
print(classification_report(y_test, y_pred))

# %% --------------------------------------------------------------------------
# heatmap
# -----------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(X.corr(), annot=True)
# num_deposit & num_withdrawals have high correlation

# %% --------------------------------------------------------------------------
# 
# -----------------------------------------------------------------------------

