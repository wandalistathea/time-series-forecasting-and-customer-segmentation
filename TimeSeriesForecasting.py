# Import Packages -----
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
# !pip install statsmodels
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
from itertools import product
from statsmodels.tools.eval_measures import rmse
from pandas.tseries.offsets import DateOffset

# Data Preparation -----
transaction = pd.read_csv("Data Project Kalbe/Case Study - Transaction.csv", sep = ";")
# Change date type
transaction["Date"] = pd.to_datetime(transaction["Date"], format = "%d/%m/%Y")

customer = pd.read_csv("Data Project Kalbe/Case Study - Customer.csv", sep = ";")
product = pd.read_csv("Data Project Kalbe/Case Study - Product.csv", sep = ";")
store = pd.read_csv("Data Project Kalbe/Case Study - Store.csv", sep = ";")
# Drop some columns from "store" dataframe
store = store.drop(columns = ["Latitude", "Longitude"])

# Merge "transaction" dataframe with other tables
df = transaction.merge(customer, how = "left", on = "CustomerID").merge(store, how = "left", on = "StoreID").merge(product, how = "left", on = "ProductID")
df = df.drop(columns = ["Price_y"])
df.rename(columns = {"Price_x": "Price"}, inplace = True)

# Time Series Forecasting -----
# Create a new dataframe for regression
# Group data by the "Date" column, calculate the sum of "Qty" for each date
ts_df = df[["Date", "Qty"]].groupby("Date").sum()
ts_df["Qty"] = ts_df["Qty"].astype("float64")

# Using ARIMA for making a prediction
ts_df.plot(figsize = (15, 5), color = "darkorange")

# Testing for stationarity
test_result = adfuller(ts_df["Qty"])

# H0: It is non-stationary
# H1: It is stationary
def adfuller_test(data):
    result = adfuller(data)
    labels = ["ADF Test Statistic", "p-value", "Number of Lags Used", "Number of Observations Used"]
    
    for label, value in zip(labels, result):
        print(label + ": " + str(value))
        
    if result[1] <= 0.05:
        print("Strong evidence against the null hypothesis (H0), reject the null hypothesis")
        print("Data has no unit root and it is stationary")
    else:
        print("Weak evidence against the null hypothesis (H0), accept the null hypothesis")
        print("Time series has a unit root, indicating it is non-stationary")

result_adfuller = adfuller_test(ts_df["Qty"])

# 80/100 * len(ts_df) # data for training

# Splitting data into training and testing data
ts_train = ts_df[:292]
ts_test = ts_df[292:]

# Autocorrelation (ACF) and Partial Autocorrelation (PACF)
# p, d, q
# p -> AR model lags (AR model is often best done with the PACF)
# d-> differencing
# q -> MA lags (MA model is often best done with the ACF rather than the PACF)

# Determine p, d, and q using grid search
# p_val = range(0, 3)
# d_val = [0] # because the data is stationary
# q_val = range(0, 3)

# pdq_val = list(product(p_val, d_val, q_val))

# aic_scores = []
# for param in pdq_val:
#     try:
#         model = ARIMA(ts_train, order = param)
#         model_fit = model.fit(method = "css")
#         aic_scores.append({"par": param, "aic": model_fit.aic})
    
#     except:
#         # Skip the parameter combination if it leads to an error
#         continue

# # Choose the lowest AIC score
# best_aic = min(aic_scores, key = lambda x: x["aic"])
# print("Best ARIMA Order (p, d, q): ", best_aic["par"])
# print("AIC Score: ", best_aic["aic"])

# Retraining data using the best order
# This model will be used to get the evaluation score
# model_train = ARIMA(ts_train, order = best_aic["par"])
model_train = ARIMA(ts_train, order = (2, 0, 2))
model_train_fit = model_train.fit()

# Forecasting to compare with testing data
preds = model_train_fit.forecast(len(ts_test))[0]

# Evaluating using RMSE
rmse_score = rmse(ts_test["Qty"], preds)

preds_df = pd.DataFrame(preds, index = ts_test.index, columns = ts_test.columns)

plt.figure(figsize = (15, 5))
plt.plot(ts_df, label = "Actual", color = "darkorange")
plt.plot(preds_df, label = "Forecast", color = "blue")
plt.legend()

# Retraining all the actual data (not just training data) using previous model ARIMA(2, 0, 2)
# This model will be used to predict the future

# model_all = ARIMA(ts_df, order = best_aic["par"])
model_all = ARIMA(ts_df, order = (2, 0, 2))
model_all_fit = model_all.fit()

# Graphic for prediction the future (for the next 30 days)
fig, ax = plt.subplots(figsize = (15, 5))
model_all_fit.plot_predict(start = 0, end = 395, ax = ax) # 365 + 30
plt.show()

future_preds = model_all_fit.forecast(30)[0]

# Making the future dates
future_dates = [ts_df.index[-1] + DateOffset(days = x) for x in range(0, 31)]

future_preds_df = pd.DataFrame(future_preds, index = future_dates[1:], columns = ts_df.columns)

# Making the prediction value as a whole number because the data is about quantity
print(round(future_preds_df))
















