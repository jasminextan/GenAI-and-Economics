import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error


df = pd.read_csv('Proj1_LSTM/encoded_dataset.csv')
X = df.iloc[:, :-1]  # input features (all columns except the last one)
y = df.iloc[:, -1]   # target variable (last column)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)

print('Mean Squared Error:', mse)
print('Mean Absolute Error:', mae)

# Mean Squared Error: 0.05844202039506963
# Mean Absolute Error: 0.20288927439422033