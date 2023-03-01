# imports

# %% --------------------------------------------------------------------------
# Imports 
# -----------------------------------------------------------------------------

import pandas as pd
import datetime
from datetime import date
import numpy as np
import matplotlib.pyplot as plt


# %% --------------------------------------------------------------------------
# Load Data 
# -----------------------------------------------------------------------------

# load data
customers = pd.read_csv(r'data/customers_tm1_e.csv')
transactions = pd.read_csv(r'data/transactions_tm1_e.csv')

transactions.head()

# create new dataset containing relevant columns
df = customers[['customer_id','dob','state','start_balance','creation_date']]

# %% --------------------------------------------------------------------------
# Clean the customers dataset
# -----------------------------------------------------------------------------
df.isna().sum()
df.dropna(inplace=True)

# fixing states
df['state'].replace(to_replace='TX', value='Texas', inplace=True)
df['state'].replace(to_replace='CALIFORNIA', value='California', inplace=True)
df['state'].replace(to_replace='MASS', value='Massachusetts', inplace=True)
df['state'].replace(to_replace='NY', value='New York', inplace=True)
# drop unidentifiable states
df = df[(df.state != '-999') & (df.state != 'UNK') & (df.state != 'Australia')]

# %% --------------------------------------------------------------------------
# remove outliers for the customers dataset
# -----------------------------------------------------------------------------

# Calculate absolute z-score of start balance
z_scores = np.abs(df['start_balance'] - np.mean(df['start_balance'])) / np.std(df['start_balance'])

# Filter out rows with z-scores > 3
df = df[z_scores < 3]

# %% --------------------------------------------------------------------------
# Adding features 
# -----------------------------------------------------------------------------

# final transaction date
transactions['transaction_date'] = pd.to_datetime(transactions['transaction_date']) # convert datetime
last_transaction = transactions.groupby('customer_id', as_index=False)['transaction_date'].max() # create table of last transaction and customer id
last_transaction.rename(columns={'transaction_date':'final_transaction_date'},inplace=True) # rename
df = df.merge(last_transaction, how='left', on='customer_id') # merge to new df

# final deposit date
transactions['deposit_date'] = pd.to_datetime(transactions['transaction_date']) # convert datetime
last_transaction = transactions[transactions["deposit"] > 0].groupby('customer_id', as_index=False)['deposit_date'].max() # create table of last transaction and customer id
last_transaction.rename(columns={'deposit_date':'final_deposit_date'},inplace=True) # rename
df = df.merge(last_transaction, how='left', on='customer_id') # merge to new df

# first transaction date
first_transaction = transactions.groupby('customer_id', as_index=False)['transaction_date'].min()
first_transaction.rename(columns={'transaction_date':'first_transaction_date'},inplace=True)
df = df.merge(first_transaction, how='left', on='customer_id')

# total deposits
tot_deposits = transactions.groupby('customer_id', as_index=False)['deposit'].sum()
tot_deposits.rename(columns={'deposit':'total_deposits'}, inplace=True)
df = df.merge(tot_deposits, how='left', on='customer_id')

# total withdrawals
tot_withdraws = transactions.groupby('customer_id', as_index=False)['withdrawal'].sum()
tot_withdraws.rename(columns={'withdrawal':'total_withdrawals'}, inplace=True)
df = df.merge(tot_withdraws, how='left', on='customer_id')
df['total_withdrawals'] = df['total_withdrawals']

# final balance
df['final_balance'] = df['start_balance'] + df['total_deposits'] + df['total_withdrawals']

# duration open
df['creation_date'] = pd.to_datetime(df['creation_date'])
df['duration_open'] = (df['final_transaction_date'] - df['first_transaction_date'])
df['duration_open'] = df['duration_open'].dt.days

# age on final transaction date
df['dob'] = pd.to_datetime(df['dob'])
df['age'] = (((df['final_transaction_date'] - df['dob']).dt.days)/365).apply(np.floor)

# avg deposits
avg_deposits = transactions.groupby('customer_id', as_index=False)['deposit'].mean()
avg_deposits.rename(columns={'deposit':'avg_deposit_val'}, inplace=True)
df = df.merge(avg_deposits, how='left', on='customer_id')

# avg withdrawals
avg_withdrawals = transactions.groupby('customer_id', as_index=False)['withdrawal'].mean()
avg_withdrawals.rename(columns={'withdrawal':'avg_withdrawal_val'}, inplace=True)
df = df.merge(avg_withdrawals, how='left', on='customer_id')

# number of deposits and withdrawals
transactions['deposit_with_nas'] = transactions['deposit'].replace({0:np.nan})
transactions['withdrawal_with_nas'] = transactions['withdrawal'].replace({0:np.nan})
new_df = transactions[['customer_id','deposit_with_nas','withdrawal_with_nas']]
df = df.merge(new_df.groupby('customer_id')['deposit_with_nas'].agg('count'), how='left', on='customer_id')
df = df.merge(new_df.groupby('customer_id')['withdrawal_with_nas'].agg('count'), how='left', on='customer_id')
df.rename(columns={'deposit_with_nas':'num_deposits','withdrawal_with_nas':'num_withdrawals'}, inplace=True)

# regions
state_groups = {'Northeast': ['New York', 'Pennsylvania', 'New Jersey', 'Connecticut', 'Massachusetts', 'Rhode Island', 'Maine', 'Vermont', 'New Hampshire', 'Maryland'],
                'Midwest': ['Illinois', 'Ohio', 'Michigan', 'Indiana', 'Wisconsin', 'Minnesota', 'Iowa', 'Missouri', 'North Dakota', 'South Dakota', 'Nebraska', 'Kansas'],
                'South': ['Texas', 'Florida', 'North Carolina', 'Georgia', 'Virginia', 'Tennessee', 'South Carolina', 'Alabama', 'Louisiana', 'Kentucky', 'Oklahoma', 'Arkansas', 'West Virginia', 'Mississippi'],
                'West': ['California', 'Washington', 'Arizona', 'Colorado', 'Oregon', 'Utah', 'Nevada', 'New Mexico', 'Idaho', 'Montana', 'Wyoming', 'Alaska', 'Hawaii', 'District of Columbia', 'Delaware']}
state_to_region = {}
for region, states in state_groups.items():
    for state in states:
        state_to_region[state] = region

# Apply the mapping to the 'state' column to create a new 'Region' column
df["region"] = df['state'].apply(lambda x: state_to_region[x] if x in state_to_region else 'Other')

# %% --------------------------------------------------------------------------
# Addition of Macroeconomic Features 
# -----------------------------------------------------------------------------

def load_data(file_path, column_names):
    df = pd.read_csv(file_path)
    df = df.rename(columns=column_names)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    return df

# Load the Interest Rates
interest_rates = load_data("data/INTDSRUSM193N.csv", {"DATE": "date", "INTDSRUSM193N": "interest_rate"})
# Load the GDP Rates
gdp = load_data("data/GDPC1.csv", {"DATE": "date", "GDPC1": "gdp"})
# Load the Inflation data
inflation = load_data("data/MICH.csv", {"DATE": "date", "MICH": "inflation_expectation"})
# Load the Unemployment data
unemployment = load_data("data/UNRATE.csv", {"DATE": "date", "UNRATE": "unemployment_rate"})
# Load Customer Sentiment Data
consumer_sent = load_data("data/UMCSENT.csv", {"DATE": "date", "UMCSENT": "consumer_sent"})


# Create monthly column for final transaction dates 
df['month_year'] = df['final_transaction_date'].dt.strftime('%Y-%m')
df['month_year'] = pd.to_datetime(df['month_year'])

# Map new dfs to df based on the month_year column
df['interest_rate'] = df['month_year'].map(interest_rates['interest_rate'])
df['gdp'] = df['month_year'].map(gdp['gdp'].resample('M').ffill())
df['inflation_expectation'] = df['month_year'].map(inflation['inflation_expectation'])
df['unemployment_rate'] = df['month_year'].map(unemployment['unemployment_rate'])
df['consumer_sent'] = df['month_year'].map(consumer_sent['consumer_sent'])
# Drop the month_year column as it is no longer needed
df.drop('month_year', axis=1, inplace=True)


# %% --------------------------------------------------------------------------
# Adding monthly customer data
# -----------------------------------------------------------------------------

# convert date columns to datetime objects
customers["creation_date"] = pd.to_datetime(customers["creation_date"])
transactions["date"] = pd.to_datetime(transactions["date"])
transactions["transaction_date"] = pd.to_datetime(transactions["transaction_date"])

# convert date columns to datetime objects
customers["creation_date"] = pd.to_datetime(customers["creation_date"])
transactions["date"] = pd.to_datetime(transactions["date"])
transactions["transaction_date"] = pd.to_datetime(transactions["transaction_date"])

# group transactions by customer and month
monthly_transactions = transactions.groupby([transactions["customer_id"], pd.Grouper(key="date", freq="M")]).agg(
    # total_amount=('amount', 'sum'),
    num_transactions=('amount', 'count'),
    total_deposit=('deposit', 'sum'),
    num_deposits=('deposit', 'count'),
    total_withdrawal=('withdrawal', 'sum'),
    num_withdrawals=('withdrawal', 'count')
).reset_index()
monthly_transactions["total_amount"] = monthly_transactions["total_deposit"] + monthly_transactions["total_withdrawal"]

# %% --------------------------------------------------------------------------
# Add the macro economical data
# -----------------------------------------------------------------------------
# Create monthly column for final transaction dates 
monthly_transactions['month_year'] = monthly_transactions['date'].dt.strftime('%Y-%m')
monthly_transactions['month_year'] = pd.to_datetime(monthly_transactions['month_year'])

# Map new dfs to df based on the month_year column
monthly_transactions['interest_rate'] = monthly_transactions['month_year'].map(interest_rates['interest_rate'])
# df['gdp'] = df['month_year'].map(gdp['gdp'])
monthly_transactions['inflation_expectation'] = monthly_transactions['month_year'].map(inflation['inflation_expectation'])
monthly_transactions['unemployment_rate'] = monthly_transactions['month_year'].map(unemployment['unemployment_rate'])
monthly_transactions['consumer_sent'] = monthly_transactions['month_year'].map(consumer_sent['consumer_sent'])
# Drop the month_year column as it is no longer needed
monthly_transactions.drop('month_year', axis=1, inplace=True)

# %% --------------------------------------------------------------------------
# Add extra features to customer-month data
# -----------------------------------------------------------------------------

new_df = df[["customer_id", "dob", "state", "region", "start_balance", "creation_date"]]
# Merge with customer data
customer_month_data = pd.merge(new_df, monthly_transactions, on='customer_id')

# Calculate age of account
customer_month_data['duration_open'] = (pd.to_datetime(customer_month_data["date"]) - pd.to_datetime(customer_month_data['creation_date'])).dt.days
customer_month_data['age'] = (pd.to_datetime(customer_month_data["date"]) - pd.to_datetime(customer_month_data['dob'])).dt.days // 365

# Get the current balance for that month 
customer_month_data['starting_balance'] = customer_month_data.groupby('customer_id')['start_balance'].transform('first')
customer_month_data['current_balance'] = customer_month_data.groupby('customer_id')['total_amount'].cumsum() + customer_month_data['starting_balance']
customer_month_data['current_balance'] = customer_month_data['current_balance'].fillna(0)

# Calculate the max balance per customer
max_balance = customer_month_data.groupby('customer_id')['current_balance'].cummax()
# Calculate the balance ratio for each month
customer_month_data['balance_ratio'] = customer_month_data['current_balance'] / max_balance

# Add z-score of the withdrawals 
customer_means = customer_month_data.groupby('customer_id')['total_withdrawal'].mean()
customer_stds = customer_month_data.groupby('customer_id')['total_withdrawal'].std()
customer_month_data['normalized_significant_withdrawals'] = customer_month_data.apply(lambda x: (x['total_withdrawal'] - customer_means[x['customer_id']]) / customer_stds[x['customer_id']], axis=1)


# %% --------------------------------------------------------------------------
# 
# -----------------------------------------------------------------------------
# create a new column with NaN values to store the number of days since last deposit
customer_month_data['days_since_last_deposit'] = np.nan

# group the data by customer and sort by date
grouped = customer_month_data.groupby('customer_id').apply(lambda x: x.sort_values('date'))

for cust, data in grouped.groupby('customer_id'):
    # find the index of the last row where total_deposit > 0
    last_deposit_idx = data[data['total_deposit'] > 0].index[-1]
    
    # iterate through each row for that customer, starting from the first row,
    # and calculate the number of days since the last deposit
    days_since_last_deposit = 0
    for idx, row in data.iterrows():
        if idx > last_deposit_idx:
            break
        if row['total_deposit'] > 0:
            last_deposit_idx = idx
            days_since_last_deposit = 0
        else:
            days_since_last_deposit += (row['date'] - data.loc[last_deposit_idx, 'date']).days
        
        # store the calculated value in the 'days_since_last_deposit' column
        customer_month_data.at[idx, 'days_since_last_deposit'] = days_since_last_deposit

# %% --------------------------------------------------------------------------
# Add features that include prior month data
# -----------------------------------------------------------------------------
# add
window_size = 3

# calculate the 3 month rolling total amount
customer_month_data['rolling_total_amount']=(customer_month_data.groupby('customer_id')['total_amount'].rolling(window_size, min_periods=1).sum()).reset_index()['total_amount'].fillna(0)

# calculate the balance variance
customer_month_data['balance_variance'] = customer_month_data.groupby('customer_id')['current_balance'].rolling(window_size).var()

# customer 
customer_month_data['consecutive_deficits'] = customer_month_data.groupby('customer_id')['total_amount'].apply(lambda x: x.rolling(min_periods=1, window=len(x)).apply(lambda y: sum(y < 0) if sum(y < 0) == len(y) else 0, raw=True))

# %% --------------------------------------------------------------------------
# Add the churn values
# -----------------------------------------------------------------------------

# get the churned column
customer_month_data['churned'] = customer_month_data.groupby('customer_id')['date'].transform('last').eq(customer_month_data['date']).astype(int)

# remove all the rows with the dates less than march 2020 (since we use three months) 
# Keep rows with transaction_date before March 2020
customer_month_data = customer_month_data[customer_month_data['date'] < '2020-03-01']

# %% --------------------------------------------------------------------------
# Clean the final dataset 
# -----------------------------------------------------------------------------

# remove the columns we do not need for the ML model
# we remove here so we don't mess with the previous dataset incase we want to make changes
final_customer_month_data = customer_month_data.dropna()
final_customer_month_data = customer_month_data.drop(columns = ['dob','state','creation_date', 'date', 'customer_id'])

# %% --------------------------------------------------------------------------
# Split the dataset
# -----------------------------------------------------------------------------

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

X = final_customer_month_data.drop(columns=["churned"])
y = final_customer_month_data['churned']

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, random_state=42)

# %% --------------------------------------------------------------------------
# train the model
# -----------------------------------------------------------------------------

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X_train, y_train)

# %% --------------------------------------------------------------------------
# showcase the data
# -----------------------------------------------------------------------------
y_pred = model.predict(X_test)
classification_report(y_test, y_pred)
