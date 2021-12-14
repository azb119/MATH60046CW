import numpy as np
import matplotlib.pyplot as plt

def S_ARMA(f, phis, thetas, sigma2):
    """
    Compute the spectral density for ARMA(len(phis),len(thetas)) process.

    If p = 0, q > 0, i.e. you pass in an empty array for phis, 
    then it will compute the spectrum of an MA(q) process. 
    If p > 0, q = 0, i.e. you pass in an empty array for thetas, 
    it will compute the spectrum of an AR(p) process.

    :param f: n-dim numpy array, freqs at which it should be evaluated.
    :param phis: p-dim numpy array, the vector [φ1,p, ..., φp,p].
    :param thetas: q-dim numpy array, the vector [θ1,q, ..., θq,q]
    :param sigma2: float, a scalar for the variance of the white noise.

    :return sf: n-dim numpy array, vector of sdf evaluated at each f
    """

    def Gsq(param):
        total = 1
        for i in range(len(param)):
            total -= param[i] * np.exp(-2j * np.pi * f * (i+1))
        return total.conjugate() * total
    
    return sigma2 * Gsq(thetas) / Gsq(phis)


def ARMA22_sim(phis, thetas, sigma2, N):
    """
    Simulate N values of a Gaussian ARMA(2,2) process.
    
    :param phis: 2-dim array, phi values for AR part.
    :param thetas: 2-dim array, theta values for MA part.
    :param sigma2: float, variance of the white noise.
    :param N: integer, number of values to be simulated

    :return x: N-dim array, simulated values
    """
    sd = np.sqrt(sigma2) # standard deviation
    x = np.array([0, 0]) # initialise burn in method
    err = np.random.normal(scale=sd, size=2) # first two error

    for i in range(98+N):
        nexterr = np.random.normal(scale=sd)
        nextx = phis[0]*x[-1] + phis[1]*x[-2] + nexterr \
               - thetas[0]*err[-1] - thetas[1]*err[-2]
        x = np.append(x, nextx)
        err = err[1], nexterr
    
    return x[-N:]


def periodogram(X):
    """
    Compute the periodogram for the time series X.

    :param X: n-dim array, time series
    :return spf:  n-dim array, periodogram at fourier frequencies
    """

    Ak = np.fft.fft(X)
    spf = (Ak.conjugate() * Ak) / len(X)

    return spf


def direct(X,p):
    """
    Compute the direct spectral estimate for time series X,
    using the p x 100% cosine taper.

    :param X: n-dim array, time series
    :param p: float, 0 <= p <= 1 for tapering
    :return sdf: n-dim array, direct spectral est at fourier freqs
    """
    N = X.shape[-1]
    a = int(np.floor(p*N/2))
    b= np.floor(p*N)
    ht = np.zeros(N)
    for i in range(a):
        ht[i] = 0.5 * (1 - np.cos(2 * np.pi * i / (b+1)))
    for i in range(a, N-a):
        ht[i] = 1
    for i in range(N-a,N):
        ht[i] = 0.5 * (1 - np.cos(2 * np.pi * (N+1-i) / (b+1)))
    C = np.sqrt(np.sum(ht**2))
    ht = ht / C

    taperX = ht*X
    Ak = np.fft.fft(taperX)
    sdf = Ak.conjugate() * Ak

    return sdf


def calc_phis(r, f=12/128):
    """
    Calculate phis for AR(2) process that exhibits pseudo cylical behaviour
    and have roots at z1 = 1/r*e^(i2f pi) and z2 = 1/r*e^(-i2f pi)

    :param r: float
    :param f: float, frequency to be calculated at
    :return phis: 2-dim array, phis defining the AR(2) process 
    """

    phis = np.array([2*r*np.cos(2*np.pi*f), -r**2])
    return phis


# these wont be changing so we define them globally
thetas = np.array([-0.5, -0.2])
N = 128
sigma2 = 1
freqs = np.array([12, 32, 60]) # index of frequencies that we want

def dpartA():
    r = 0.8
    phis = calc_phis(r)
    realizations = 100
    x = np.zeros((realizations, N))
    for i in range(realizations):
        x[i] = ARMA22_sim(phis, thetas, sigma2, N)

    spf = periodogram(x)[:,freqs]

    pvals = [0.05, 0.1, 0.25, 0.5]
    sdf0 = direct(x, pvals[0])[:,freqs]
    sdf1 = direct(x, pvals[1])[:,freqs]
    sdf2 = direct(x, pvals[2])[:,freqs]
    sdf3 = direct(x, pvals[3])[:,freqs]

    data = [spf, sdf0, sdf1, sdf2, sdf3]

    return data

def dpartB():
    data = dpartA()
    f = freqs/128
    phis = calc_phis(0.8)
    theoretical = S_ARMA(f, phis, thetas, sigma2)
    allsamplebias = np.zeros((5, len(freqs)))
    i = 0
    for sf in data:
        samplemean = np.mean(sf, axis=0)
        samplebias = samplemean - theoretical
        allsamplebias[i] = samplebias
        i += 1

    return allsamplebias

def dpartC(r):
    """
    Repeating steps A and B for different values of r
    """
    # part A
    phis = calc_phis(r)
    realizations = 100
    x = np.zeros((realizations, N))
    for i in range(realizations):
        x[i] = ARMA22_sim(phis, thetas, sigma2, N)

    spf = periodogram(x)[:,freqs]

    pvals = [0.05, 0.1, 0.25, 0.5]
    sdf0 = direct(x, pvals[0])[:,freqs]
    sdf1 = direct(x, pvals[1])[:,freqs]
    sdf2 = direct(x, pvals[2])[:,freqs]
    sdf3 = direct(x, pvals[3])[:,freqs]

    data = [spf, sdf0, sdf1, sdf2, sdf3]

    # part B
    f = freqs/128
    theoretical = S_ARMA(f, phis, thetas, sigma2)
    allsamplebias = np.zeros((5, len(freqs)))
    i = 0
    for sf in data:
        samplemean = np.mean(sf, axis=0)
        samplebias = samplemean - theoretical
        allsamplebias[i] = samplebias
        i += 1

    return allsamplebias, theoretical

rvals = np.arange(0.8, 1, 0.01)
bias_each_r = []
theo_each_r = []
for r in rvals:
    bias, theo = dpartC(r)
    bias_each_r.append(bias)
    theo_each_r.append(theo)
bias_each_r = np.array(bias_each_r)
theo_each_r = np.array(theo_each_r)

for i in range(3):
    plt.figure(i)
    for j in range(5):
        plt.plot(rvals, bias_each_r[:,j, i])
    plt.legend(['Periodogram', 'p=0.05', 'p=0.1', 'p=0.25', 'p=0.5'])
    plt.xlabel('r')
    plt.ylabel('bias')
    plt.title(f'f={freqs[i]}/128')
plt.show()
