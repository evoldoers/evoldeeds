import math
import numpy as np
import logging

import h20

smallest_float32 = np.finfo('float32').smallest_normal
def logTransitionMatrixPolynomialApproximation (indelParams, alphabetSize, degree = 8, tmin = 1e-6, tmax = 2, steps = 100, epsilon = 1e-9):
    lam, mu, x, y = indelParams
    if tmax is None:
        tmax = tbound (lam, mu, x, y, alphabetSize)
    ts = np.geomspace (tmin, tmax, steps)
    ms = np.stack ([h20.transitionMatrix(t,indelParams,alphabetSize) for t in ts], axis=0)  # (steps,3,3)
    coeffs = np.array ([[np.polynomial.polynomial.Polynomial.fit(ts, ms[:,i,j], degree, window=(0,tmax)).coef for j in range(3)] for i in range(3)])  # (3,3,degree+1)
    tpow = ts[None,:] ** np.arange(degree+1)[:,None]  # (degree+1,steps)
    ms_predicted = coeffs @ tpow  # (3,3,steps)
    ms_predicted = np.moveaxis (ms_predicted, 2, 0)  # (steps,3,3)
    ms_rmserr = np.sqrt (np.mean ((ms_predicted - ms) ** 2, axis=0))  # (3,3)
    return coeffs, tmin, tmax, ms_rmserr

def substitutionMatrixDiagonalForm (subRate):
    evals, evecs_r = np.linalg.eig (subRate)
    evecs_l = np.linalg.inv (evecs_r)
    subRate_fit = evecs_r @ np.diag(evals) @ evecs_l
    subRate_rmsErr = np.sqrt (np.mean ((subRate_fit - subRate) ** 2))
    logging.warning (f"Substitution matrix RMS error: {subRate_rmsErr}")
    return evecs_l, evals, evecs_r

# calculate the upper bound time beyond which an alignment is unrecognizable
def tbound(lam,mu,x,y,alphabetSize):
    k = 2  # precautionary factor representing how unrecognizable we want the alignment to be
    t = max((1-x)/lam,(1-y)/mu) * math.log (1+alphabetSize)
    for i in range(100):
        t = tnext(t,lam,x,mu,y,alphabetSize,k)
    return t

def tnext(t,l,x,m,y,a,k):
    return math.log (1 + k * math.pow (a, 1 / (1 - math.exp(-m*t)))) / (l/(1-x) + m/(1-y))

