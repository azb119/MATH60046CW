import numpy as np
import matplotlib.pyplot as plt
import csv
from question1 import *

# import the data
with open('MATH60046CW/158.csv') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    data = np.array(list(reader)[0], dtype='float64')

spf = np.fft.fftshift(periodogram(data))
sdf = np.fft.fftshift(direct(data, 0.5))

f = np.linspace(-0.5, 0.5, len(spf)+1)[:-1] # to not incl 0.5

plt.figure('periodogram')
plt.plot(f, spf)

plt.figure('direct spectral estimate')
plt.plot(f,sdf)
plt.title('Direct Spectral Estimate')
plt.xlabel('frequency')
plt.ylabel()

plt.show()