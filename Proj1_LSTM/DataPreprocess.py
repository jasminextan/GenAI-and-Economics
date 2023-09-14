import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense

minwage_df = pd.read_csv('Proj1_LSTM/MinimumWage.csv')
partcon_df = pd.read_csv('Proj1_LSTM/MinWage_PartyControl.csv')

print(minwage_df.head())
print(partcon_df.head())

#Combine the two datasets with the key 'Year'
combined_df = pd.merge(minwage_df, partcon_df, on='Year')
print(combined_df)
combined_df.to_csv('Proj1_LSTM/combined_dataset.csv', index=False)

# Load the dataset
df = pd.read_csv('Proj1_LSTM/combined_dataset.csv')

# Drop duplicate features
df = df.drop(['FederalMinimumWage'], axis=1)

# Handle missing values
df = df.dropna()

print(df.head())

# Convert the money columns to float
df['FedMinWage'] = df['FedMinWage'].str.replace('$', '').astype(float)
df['MinWageIndexedLastRaiseYear'] = df['MinWageIndexedLastRaiseYear'].str.replace('$', '').astype(float)
df['RateChange'] = df['RateChange'].str.replace('$', '').astype(float)
     
# Verify the data types
print(df.dtypes)

"""
# Perform feature scaling
scaler = MinMaxScaler()
numeric_features = ['FedMinWage', 'RateChange', 'PercentChange', 'YearsSinceLastChange', 'FederalMinimumWage', 'MeanAnnualInflation', 'MinWageIndexedLastRaiseYear', 'UnemploymentRateDecember', 'GDP_AnnualGrowth']
df[numeric_features] = scaler.fit_transform(df[numeric_features])

# Encode categorical variables
categorical_features = ['PresParty', 'SenParty', 'HouseParty', 'TrifectaFlag', 'IncreaseFlag']
encoder = OneHotEncoder(sparse=False)
encoded_features = pd.DataFrame(encoder.fit_transform(df[categorical_features]))
df = pd.concat([df, encoded_features], axis=1).drop(categorical_features, axis=1)

# Create input sequences and target variables
n_steps = 5  # Number of time steps in each input sequence
X = []
y = []
for i in range(n_steps, len(df)):
    X.append(df.iloc[i-n_steps:i, :].values)
    y.append(df.iloc[i, -1])
X = np.array(X)
y = np.array(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
"""