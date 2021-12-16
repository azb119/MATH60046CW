import numpy as np
import csv
import matplotlib.pyplot as plt
from question1 import S_ARMA
from question2 import *


with open('MATH60046CW/158.csv') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    data = np.array(list(reader)[0], dtype='float64')

YW_phis, YW_s2 = YW(data, 5)
YWt_phis, YWt_s2 = YW_50taper(data, 8)
MLE_phis, MLE_s2 = approxMLE(data, 9)

name_list = ["YW", "YW tapered", "MLE"]
phis_list = [YW_phis, YWt_phis, MLE_phis]
sigma2_list = [YW_s2, YWt_s2, MLE_s2]
n = 1000
f = np.linspace(-0.5, 0.5, n)

sf_list = []
for i in range(3):
    sf = S_ARMA(f, phis_list[i], [], sigma2_list[i])
    sf_list.append(sf)

    plt.figure(0)
    plt.plot(f, sf, label=name_list[i])
plt.xlabel("frequency")
plt.ylabel("S(f)")
plt.title("Spectral Density Functions of different models")
plt.legend()
plt.show()
