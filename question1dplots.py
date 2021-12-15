from question1 import *
import numpy as np
import matplotlib.pyplot as plt

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
    realizations = 10000
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
