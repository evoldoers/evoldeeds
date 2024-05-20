import sys
import json
import logging

import jax.numpy as jnp

import likelihood
import approx

if len(sys.argv) != 3:
    print("Usage: {} model.json method >model_prep.json\n\nmethod can be 'poly' or 'piecewise'".format(sys.argv[0]))
    sys.exit(1)

modelFilename, method = sys.argv[1:]
with open(modelFilename, 'r') as f:
    modelJson = json.load (f)

alphabet, mixture, indelParams = likelihood.parseHistorianParams (modelJson)
alphabetSize = len(alphabet)

if method == 'poly':
    logging.warning("Computing transition matrix polynomial approximation")
    coeffs, tmin, tmax, rmserr = approx.logTransitionMatrixPolynomialApproximation (indelParams, alphabetSize)
    logging.warning(f"RMS error: {rmserr}")
    hmm = { 'tmin': tmin,
            'tmax': tmax,
            'poly': coeffs.tolist() }
elif method == 'piecewise':
    logging.warning("Computing piecewise linear counts approximation")
    lam,mu,x,y = indelParams
    counts, t1, tmax, steps = approx.piecewiseLinearCountsApproximation (indelParams, alphabetSize)
    hmm = { 't1': float(t1),
            'tmax': float(tmax),
            'steps': steps,
            'lmxy': [lam, mu, x, y],
            'abuq': [c.tolist() for c in counts]
            }

mixture_diag = [(rootProb, approx.substitutionMatrixDiagonalForm(subRate)) for subRate, rootProb in mixture]

print (json.dumps({ 'alphabet': alphabet,
                    'hmm': hmm,
                    'mixture': [{'root': rootProb.tolist(),
                                 'evecs_l': evecs_l.tolist(),
                                 'evals': evals.tolist(),
                                 'evecs_r': evecs_r.tolist()} for rootProb, (evecs_l, evals, evecs_r) in mixture_diag]}))
