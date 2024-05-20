import sys
import json
import logging

import jax.numpy as jnp

import likelihood
import approx

if len(sys.argv) != 2:
    print('Usage: {} model.json >model_prep.json'.format(sys.argv[0]))
    sys.exit(1)

modelFilename = sys.argv[1]
with open(modelFilename, 'r') as f:
    modelJson = json.load (f)

alphabet, mixture, indelParams = likelihood.parseHistorianParams (modelJson)
alphabetSize = len(alphabet)

logging.warning("Computing transition matrix polynomial approximation")
coeffs, tmin, tmax, rmserr = approx.logTransitionMatrixPolynomialApproximation (indelParams, alphabetSize)
logging.warning(f"RMS error: {rmserr}")

mixture_diag = [(rootProb, approx.substitutionMatrixDiagonalForm(subRate)) for subRate, rootProb in mixture]

print (json.dumps({ 'alphabet': alphabet,
                    'hmm': { 'tmin': tmin,
                             'tmax': tmax,
                             'poly': coeffs.tolist() },
                    'mixture': [{'root': rootProb.tolist(),
                                 'evecs_l': evecs_l.tolist(),
                                 'evals': evals.tolist(),
                                 'evecs_r': evecs_r.tolist()} for rootProb, (evecs_l, evals, evecs_r) in mixture_diag]}))
