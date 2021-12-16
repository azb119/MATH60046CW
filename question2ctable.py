import numpy as np
from question2 import *
import csv


with open('MATH60046CW/158.csv') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    data = np.array(list(reader)[0], dtype='float64')


def AIC(N, p, sigma2):
    """
    Calculate the AIC for fitting N time series points to a
    Gaussian AR(p) model with white noise variance sigma2.

    :param p: int, parameter for AR(p)
    :param N: int, length of time series, len(X)
    :param sigma2: float, variance for white noise

    :return aic: float, AIC for those parameters
    """
    aic = 2*p + N * np.log(sigma2)
    return aic


YW_aic = np.zeros(20)
YW_tapered_aic = np.zeros(20)
MLE_aic = np.zeros(20)
pvals = np.arange(1,21)
N = len(data)

for i in range(len(pvals)):
    p = pvals[i]
    _, sigma2 = YW(data, p)
    YW_aic[i] = AIC(N, p, sigma2)

    _, sigma2 = YW_50taper(data, p)
    YW_tapered_aic[i] = AIC(N, p, sigma2)

    _, sigma2 = approxMLE(data, p)
    MLE_aic[i] = AIC(N, p, sigma2)

print(np.round(np.concatenate(([YW_aic],
      [YW_tapered_aic], [MLE_aic])).T, decimals=3))
