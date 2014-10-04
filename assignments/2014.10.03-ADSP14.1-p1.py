# -*- coding: utf-8 -*-

import matplotlib
matplotlib.use('Qt4Agg', warn=False)

import numpy as np
import scipy.signal as sig
import scipy as sp
import pylab


pylab.close('all')


#%%

#####
# Part a
#####
NBINS = 50
RANGE = 15.0
NVALS = 1000
SIGMA = 2.0
MU = 0

print("**** Part a ****")

pylab.figure()
values = np.random.normal(MU, SIGMA, NVALS)
xvals = np.linspace(-RANGE/2, RANGE/2, NBINS)
bins = np.zeros_like(xvals)
for v in values:
    dst = np.floor((v + RANGE/2) * NBINS / RANGE)
    try:
        bins[dst] += (NBINS/RANGE)/NVALS
    except:
        print("Sample out of range")

def normpdf(x, mu, sigma):
    return ((1.0/(sigma * np.sqrt(2*np.pi))) *
           np.e**(-(x - mu)**2/(2*sigma**2)))

pylab.plot(xvals, bins)
pylab.plot(xvals, normpdf(xvals, MU, SIGMA))
pylab.title("Part A: Estimated PDF vs Gaussian")
pylab.show()

#%%
#####
# Part B
#####

print("**** Part b ****")

KVALS = [40, 1000]
SIGMA = 2.0
MU = 0
NVALS = 1000
XVALS = []


for K in KVALS:
    X = np.random.normal(MU, SIGMA, (K, NVALS))
    XVALS.append(X)

#%%
#####
# Part C
#####

print("**** Part c ****")

pylab.figure()
ax = None
for X, K, i in zip(XVALS, KVALS, range(len(XVALS))):    
    ax = pylab.subplot2grid((len(XVALS), 1), (i, 0),
                            sharey=ax)
    
    # A line of the actual mean values
    line = np.zeros(K)
    # The K mean values
    means = np.mean(X, 1)
    pylab.plot(range(K), line)
    pylab.scatter(range(K), means)

pylab.suptitle("Part C: Sample Means vs Expected")
pylab.show()

#%%
#####
# Part D
#####

print("**** Part d ****")

print("Each estimator (dataset) should have variance sigma**2 / N")
SIGMA2_est = SIGMA**2/NVALS

for X, K, i in zip(XVALS, KVALS, range(len(XVALS))):
    print("K = {}:".format(K))
    print("\tSample mean: {}".format(np.mean(X)))
    var = np.var(np.mean(X, 1))
    print("\tSample mean variance: {}".format(var))
    print("\tExpected estimator variance: {}".format(SIGMA2_est))

#%%
#####
# Part E
#####

print("**** Party ****")

# Screw the other one
m = np.mean(XVALS[1], 1)
K = KVALS[1]
NBINS = 200
RANGE = 3.0
xvals = np.linspace(-RANGE/2, RANGE/2, NBINS, dtype=np.float128)
bins = np.zeros_like(xvals)

pylab.figure()
pdf = normpdf(xvals, MU, np.sqrt(SIGMA2_est))
for mu in m:
    dst = np.floor((mu + RANGE/2) * NBINS / RANGE)
    try:
        bins[dst] += (NBINS/RANGE)/len(m)
    except:
        print("Sample out of range")

pylab.plot(xvals, bins)
pylab.plot(xvals, pdf)
pylab.title("Part E: Estimated PDF vs Expected")
pylab.show()
