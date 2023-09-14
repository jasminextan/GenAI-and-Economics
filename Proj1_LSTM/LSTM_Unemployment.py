import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score



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

# Verify the data types
print(df.dtypes)

# Convert the money columns to float
df['FedMinWage'] = df['FedMinWage'].str.replace('$', '').astype(float)
df['MinWageIndexedLastRaiseYear'] = df['MinWageIndexedLastRaiseYear'].str.replace('$', '').astype(float)
df['RateChange'] = df['RateChange'].str.replace('$', '').astype(float)

# Convert the percentage columns to float
df['MeanAnnualInflation'] = df['MeanAnnualInflation'].str.replace('%', '')
df['MeanAnnualInflation'] = df['MeanAnnualInflation'].astype(float)
df['UnemploymentRateDecember'] = df['UnemploymentRateDecember'].str.replace('%', '')
df['UnemploymentRateDecember'] = df['UnemploymentRateDecember'].astype(float)
df['GDP_AnnualGrowth'] = df['GDP_AnnualGrowth'].str.replace('%', '')
df['GDP_AnnualGrowth'] = df['GDP_AnnualGrowth'].astype(float)

# Verify the data types
print(df.dtypes)

# Perform feature scaling
scaler = MinMaxScaler()
numeric_features = ['FedMinWage', 'RateChange', 'PercentChange', 'YearsSinceLastChange', 'MeanAnnualInflation', 'MinWageIndexedLastRaiseYear', 'UnemploymentRateDecember', 'GDP_AnnualGrowth']
df[numeric_features] = scaler.fit_transform(df[numeric_features])

print(df.head())

# Encode categorical variables
categorical_features = ['PresParty', 'SenParty', 'HouseParty', 'TrifectaFlag', 'IncreaseFlag']
df_encoded = pd.get_dummies(df, columns=categorical_features)
print(df_encoded)
print(df_encoded.dtypes)

# Drop perfect complements
df_encoded = df_encoded.drop(['PresParty_Republican'], axis=1)
df_encoded = df_encoded.drop(['SenParty_Republican'], axis=1)
df_encoded = df_encoded.drop(['HouseParty_Republican'], axis=1)
df_encoded = df_encoded.drop(['IncreaseFlag_False'], axis=1)

print(df_encoded.dtypes)
df_encoded.to_csv('Proj1_LSTM/encoded_dataset.csv', index=False)

# Move target var to end of dataset, change datatype
unemployment_rate = df_encoded.pop('UnemploymentRateDecember')
df_encoded.insert(len(df_encoded.columns), 'UnemploymentRateDecember', unemployment_rate)

# Create input sequences and target variables
n_steps = 5  # Number of time steps in each input sequence
X = []
y = []
for i in range(n_steps, len(df)):
    X.append(df_encoded.iloc[i-n_steps:i, :-1].values)  # Exclude the last column (target variable)
    y.append(df_encoded.iloc[i, -1])  # Use the last column (UnemploymentRateDecember) as the target variable
X = np.array(X)
y = np.array(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Convert the NumPy arrays to TensorFlow tensors
X_train = X_train.astype(float)
X_test = X_test.astype(float)
y_train = y_train.astype(float)
y_test = y_test.astype(float)
X_train = tf.convert_to_tensor(X_train)
X_test = tf.convert_to_tensor(X_test)
y_train = tf.convert_to_tensor(y_train)
y_test = tf.convert_to_tensor(y_test)

# Define the model
n_features = 14
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32)

# Evaluate the model
loss = model.evaluate(X_test, y_test)
print(f'Loss: {loss}')

"""
mae, precision, recall = model.evaluate(X_test, y_test)
f1_score = 2 * (precision * recall) / (precision + recall)
print(f'MAE: {mae}, Precision: {precision}, Recall: {recall}, F1-score: {f1_score}')
"""
# Loss: The value of the loss function, which is a measure of the model's error. In this case, we use binary cross-entropy as it's a binary classification problem.
# Accuracy: The proportion of correct predictions out of the total number of instances.
# MAE: The average of the absolute differences between the predicted and actual values.
# Precision: The proportion of true positive predictions out of the total predicted positives.
# Recall: The proportion of true positive predictions out of the total actual positives.
# F1-score: The harmonic mean of precision and recall, providing a balance between the two metrics.