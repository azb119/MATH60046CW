import numpy as np
from scipy.linalg import solve_toeplitz


# Yule-Walker untapered
def acvs_estimator(x):
    """
    Calculate the estimator for the acvs sequence for
    the time series x with 0 mean.

    The estimator used is the biased estimator;
    s_tau = 1/N sum_{t=1}^{N-tau} X_t * X_t+tau

    :param x: n-dim array, time series
    :return acvs: n-dim array, acvs estimate from s_0 to s_{N-1}
    """

    N = len(x)
    acvs = []
    for i in range(N):
        lagged = np.zeros(N)
        lagged[:N-i] = x[i:]
        s_i = np.sum(lagged * x)/N
        acvs.append(s_i)
    
    return acvs


def YW(x, p):
    """
    Fit time series data x to an AR(p) model using untapered 
    Yule-Walker.

    :param x: n-dim array, time series
    :param p: integer, parameter for AR(p)

    :return phis: p-dim array, fitted phi values for AR(p) model
    :return sigma2: float, fitted variance for AR(p) model
    """

    # calc acvs estimator
    acvs = acvs_estimator(x)
    # construct matrix and vectors
    # if p > N, s_tau = 0 for tau > N-1
    gamma = np.zeros(p)
    gamma[:len(x)] = acvs[1:p+1]
    toeplitz_matrix = np.zeros(p)
    toeplitz_matrix[:len(x)] = acvs[:p]

    # use the levinson recursion to solve the system of eqns
    phis = solve_toeplitz(toeplitz_matrix, gamma)
    sigma2 = acvs[0] - sum(phis * acvs[1:len(phis)+1])
    return phis, sigma2


# 50% cosine tapered Yule-Walker
def cos_taper(N, p):
    """
    Calculate ht for the px100% cosine taper for N data points.

    :param N: integer, no. of data points, len(X)
    :param p: float, 0 < p < 1 how much is tapered

    :return ht: N-dim array, ht values
    """
    a = int(np.floor(p*N/2))
    b = np.floor(p*N)
    ht = np.zeros(N)
    for i in range(a):
        ht[i] = 0.5 * (1 - np.cos(2 * np.pi * i / (b+1)))
    for i in range(a, N-a):
        ht[i] = 1
    for i in range(N-a,N):
        ht[i] = 0.5 * (1 - np.cos(2 * np.pi * (N+1-i) / (b+1)))
    C = np.sqrt(np.sum(ht**2))
    ht = ht / C

    return ht


def YW_50taper(x, p):
    """
    Fit time series x to AR(p) model using the tapered 
    Yule-Walker method, with a 50% cosine taper.

    :param x: n-dim array, time series
    :param p: integer, parameter for AR(p)

    :return phis: p-dim array, fitted phi values for AR(p) model
    :return sigma2: float, fitted variance for AR(p) model
    """

    n = len(x)
    ht = cos_taper(n, 0.5)
    htx = ht * x # tapered time series
    htx = htx * np.sqrt(n) # to offset the diff with normal acvs
    # reuse the function from above
    phis, sigma2= YW(htx, p)
    return phis, sigma2


# approx maximum likelihood, we assume p < N
def approxMLE(x, p):
    """
    Fit time series x to AR(p) model using the maximum 
    likelihood method, and assuming the first p data points 
    are deterministic.

    :param x: n-dim array, time series
    :param p: integer, parameter for AR(p)

    :return phis: p-dim array, fitted phi values for AR(p) model
    :return sigma2: float, fitted variance for AR(p) model
    """
    F = np.zeros((len(x)-p,p))
    for i in range(len(x)-p):
        F[i] = x[i:p+i][::-1]
    X = x[p:]
    b = F.T @ X
    A = F.T @ F
    phis = np.linalg.solve(A, b)
    
    tocalcsigma = X - F @ phis
    sigma2 = np.inner(tocalcsigma, tocalcsigma)/ (len(x) -p)

    return phis, sigma2
