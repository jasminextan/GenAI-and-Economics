import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error

df = pd.read_csv('Proj1_LSTM/encoded_dataset.csv')

# Parameter Selection: Use auto_arima() from the pmdarima package to automatically select the best p, d, q parameters for the ARIMA model.
model = auto_arima(df['UnemploymentRateDecember'], trace=True, error_action='ignore', suppress_warnings=True)
model.fit(df['UnemploymentRateDecember'])

# Display the selected ARIMA order
print(model.order)

# Model fitting
p, d, q = model.order
model = ARIMA(df['UnemploymentRateDecember'], order=(p,d,q))
model_fit = model.fit()

# Forecasting
forecast = model_fit.forecast(steps=12) # forecast next 12 points
print("forecast: ", forecast)

# Model Evaluation
test = df['UnemploymentRateDecember'][-12:]
predictions = forecast
mse = mean_squared_error(test, predictions)
rmse = mse ** 0.5
print(f'MSE: {mse}')
print(f'RMSE: {rmse}')

# MSE: 0.011880394889816057
# RMSE: 0.1089972242298677
