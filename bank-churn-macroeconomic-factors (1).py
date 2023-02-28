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
customers = pd.read_csv(r'customers_tm1_e.csv')
transactions = pd.read_csv(r'transactions_tm1_e.csv')

transactions.head()

# create new dataset containing relevant columns
df = customers[['customer_id','dob','state','start_balance','creation_date']]


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
state_groups = {'Northeast': ['New York', 'Pennsylvania', 'New Jersey', 'Connecticut', 'Massachusetts', 'Rhode Island', 'Maine', 'Vermont', 'New Hampshire'],
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
interest_rates = load_data("INTDSRUSM193N.csv", {"DATE": "date", "INTDSRUSM193N": "interest_rate"})
# Load the GDP Rates
# gdp = load_data("data/GDPC1.csv", {"DATE": "date", "GDPC1": "gdp"})
# Load the Inflation data
inflation = load_data("MICH.csv", {"DATE": "date", "MICH": "inflation_expectation"})
# Load the Unemployment data
unemployment = load_data("UNRATE.csv", {"DATE": "date", "UNRATE": "unemployment_rate"})
# Load Customer Sentiment Data
consumer_sent = load_data("UMCSENT.csv", {"DATE": "date", "UMCSENT": "consumer_sent"})

# Set index of new dfs to date column
#interest_rates = interest_rates.set_index('date')
#gdp = gdp.set_index('date')
#inflation = inflation.set_index('date')
#unemployment = unemployment.set_index('date')
#consumer_sent = consumer_sent.set_index('date')

# Create monthly column for final transaction dates 
df['month_year'] = df['final_transaction_date'].dt.strftime('%Y-%m')
df['month_year'] = pd.to_datetime(df['month_year'])

# Map new dfs to df based on the month_year column
df['interest_rate'] = df['month_year'].map(interest_rates['interest_rate'])
# df['gdp'] = df['month_year'].map(gdp['gdp'])
df['inflation_expectation'] = df['month_year'].map(inflation['inflation_expectation'])
df['unemployment_rate'] = df['month_year'].map(unemployment['unemployment_rate'])
df['consumer_sent'] = df['month_year'].map(consumer_sent['consumer_sent'])
# Drop the month_year column as it is no longer needed
df.drop('month_year', axis=1, inplace=True)


# %% --------------------------------------------------------------------------
# Adding montly customer data
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
    total_amount=('amount', 'sum'),
    num_transactions=('amount', 'count'),
    total_deposit=('deposit', 'sum'),
    num_deposits=('deposit', 'count'),
    total_withdrawal=('withdrawal', 'sum'),
    num_withdrawals=('withdrawal', 'count')
).reset_index()


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

# Merge with customer data
customer_month_data = pd.merge(customers, monthly_transactions, on='customer_id')

# Calculate age of account
customer_month_data['duration_open'] = (pd.to_datetime(customer_month_data["date"]) - pd.to_datetime(customer_month_data['creation_date'])).dt.days
customer_month_data['age'] = (pd.to_datetime(customer_month_data["date"]) - pd.to_datetime(customer_month_data['dob'])).dt.days // 365

# Get the current balance for that month 
customer_month_data['current_balance'] = customer_month_data['start_balance'] + customer_month_data['total_amount'].cumsum()

# Calculate the max balance per customer
max_balance = customer_month_data.groupby('customer_id')['current_balance'].cummax()
# Calculate the balance ratio for each month
customer_month_data['balance_ratio'] = customer_month_data['current_balance'] / max_balance

# # Calculate the net consective net works
# customer_month_data['consecutive_net_contributions'] = (customer_month_data.groupby('customer_id')['total_amount']
#                                                         .apply(lambda x: (x >= 0).cumprod().diff().fillna(0).eq(-1).cumsum()))

# # Fill NaN values with 0
# customer_month_data['consecutive_net_contributions'] = customer_month_data.groupby('customer_id')['consecutive_net_contributions'].apply(lambda x: x.fillna(method='ffill').fillna(0))

# customer_month_data

# Drop unnecessary columns
# customer_month_data.drop(['transaction_date'], axis=1, inplace=True)
# customer_month_data

# %%
import matplotlib.pyplot as plt
import seaborn as sns
# sns.pairplot(customer_month_data[0:10000])

fig, ax = plt.subplots(figsize=(20, 20))
sns.heatmap(customer_month_data.corr(), annot=True)


# %%

# %% --------------------------------------------------------------------------
# Drop insignificant variables
# -----------------------------------------------------------------------------
# df.drop(columns=['creation_date', 'dob', 'customer_id'], inplace=True)
# df


df['churn']

# %%
