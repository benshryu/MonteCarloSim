#https://medium.com/analytics-vidhya/monte-carlo-simulations-for-predicting-stock-prices-python-a64f53585662
import numpy as np
import pandas as pd
from numpy import ndarray
from pandas_datareader import data as wb
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.linear_model import LinearRegression

ticker = 'AAPL'
data = pd.DataFrame()
data[ticker] = wb.DataReader(ticker, data_source = 'yahoo', start = '2010-1-1')['Adj Close']
#Plot
data.plot(figsize=(15, 6))

log_return = np.log(1 + data.pct_change())
#Plot
#sns.displot(log_return.iloc[1:])
#plt.xlabel("Daily Return")
#plt.ylabel("Frequency")

#work on tunning the variables
u = log_return.mean()
var = log_return.var()
drift = u - (0.5*var)

stdev = log_return.std()
days = 50
trials = 10000
Z = norm.ppf(np.random.rand(days, trials)) #days, trials
daily_returns = np.exp(drift.values + stdev.values * Z)

price_paths: ndarray = np.zeros_like(daily_returns)
price_paths[0] = data.iloc[-1]
for t in range(1, days):
    price_paths[t] = price_paths[t-1] * daily_returns[t]

sns.displot()
plt.plot(price_paths)
plt.show()