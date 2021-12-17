import numpy as np
import csv
from question2 import *
from tabulate import tabulate
import matplotlib.pyplot as plt
from numpy import random

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
    Calculate the next l forecasts using the model AR(p).

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



def next_point(past_points, phis, sigma2):
    """
    Calculate the next point in the model trajectory.
    X_t = phi_1 X_t-1 + ... + phi_p X_t-p + e_t
    e_t white noise with var sigma2

    :param past_points: n x t dim numpy array,
                        past simulated or observed points
    :param phis: p dim array, phis for AR(p) model
    :param sigma2: float, variance of white noise/innovations

    :return next: n dim array, next points of simulated paths
    """

    innovations = random.normal(0, np.sqrt(sigma2),
                                len(past_points))
    next = np.sum(past_points[:,-len(phis):] * phis[::-1],
                  axis=1) + innovations
    return next


def sim_trajectory(n, observed, phis, sigma2):
    """
    Simulate 999 trajectories of time series observed up to 
    n time points in the future using Monte Carlo simulation.

    :param n: int, no. of time points to be simulated
    :param observed: t dim array, observed time series
    :param phis: p dim array, phis for AR(p) model
    :param sigma2: float, variance of white noise/innovations

    :return current: 999 x n dim array, 999 simulated paths
    """
    current = np.array([observed[-len(phis):]
                        for i in range(999)])
    for i in range(n):
        next = next_point(current, phis, sigma2)
        current = np.concatenate((current,
                                  np.atleast_2d(next).T),
                                 axis=1)

    return current[:,-n:]


n = 10
trajectory = sim_trajectory(n, observed, fitted_phis, fitted_sigma2)
transpose_trajectory = trajectory.T
transpose_trajectory.sort()
higher_lim= transpose_trajectory[:,-50]
lower_lim = transpose_trajectory[:,-950]


plt.figure(0)
# plot the true trajectory
true_path = data[99:]
timepoints = np.arange(100, 129)
plt.plot(timepoints, true_path, 'b', label="true path")

# plot predicted path using forecasting
timepoints2 = np.arange(119, 129)
predicted_path = forecasted
plt.plot(timepoints2, predicted_path, 'g', label="predicted path")

# plot the prediction interval
plt.plot(timepoints2, higher_lim, '--r', label="interval")
plt.plot(timepoints2, lower_lim, '--r')

plt.legend()
plt.xlabel('time')
plt.title('True path vs predicted')
plt.show()