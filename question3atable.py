import numpy as np
import csv
from question2 import *
from tabulate import tabulate

# import data
with open('MATH60046CW/158.csv') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    data = np.array(list(reader)[0], dtype='float64')

observed = data[:118]
best_p = 9 # for AR(p) according to lowest AIC
fitted_phis, fitted_sigma2 = approxMLE(observed, best_p)
max_l = len(data) - len(observed)


def forecast_AR(observed, phis, l):
    """"
    Calculate the next l forecasts using the model AR(len(phis)).

    :param observed: n-dim array, observed time series,
                     len(observed) should be more than len(phis)
    :param phis: p-dim numpy array, phis for the AR(p) model.
    :param l: int, how many future time points to be forecasted

    :return forecast: l-dim array, forecasted time series
    """

    p = len(phis)
    forecast = np.zeros(l)
    last_p = observed[-p:]
    for i in range(l):
        next_forecast = np.sum(last_p * np.array(phis)[::-1])
        last_p[:-1] = last_p[1:]
        last_p[-1] = next_forecast
        forecast[i] = next_forecast

    return np.array(forecast)


forecasted = forecast_AR(observed, fitted_phis, max_l)
actual_vals = data[118:]
indices = np.arange(119, 129)

table_data = np.concatenate(([indices], [actual_vals], 
                             [forecasted])).T
print(tabulate(table_data, 
               headers=['t', 'actual value', 'forecasted value']))
