
# %% --------------------------------------------------------------------------
# imports
# -----------------------------------------------------------------------------

import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import matplotlib.pyplot as plt

# %% --------------------------------------------------------------------------
# load data
# -----------------------------------------------------------------------------
df = pd.read_csv(r'Banking Customer Churn\bank-customer-churn\data\cleaned_data.csv')

# %% --------------------------------------------------------------------------
# train test split
# -----------------------------------------------------------------------------

excluded_variables = ['state','customer_id','dob','creation_date','date','start_balance','total_amount','num_transactions','interest_rate','inflation_expectation','unemployment_rate']
df = df.drop(columns=excluded_variables)

# ohe
df = pd.get_dummies(df, drop_first=True)
X = df.drop(columns=['churned','num_withdrawals','num_deposits','gdp'])
y = df['churned']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=123, test_size=0.2, stratify=y)

# %% --------------------------------------------------------------------------
# model 1
# -----------------------------------------------------------------------------

model_1 = xgb.XGBClassifier()
model_1.fit(X_train,y_train)
y_pred = model_1.predict(X_test)
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

