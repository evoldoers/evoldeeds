import sys
import json

import jax.numpy as jnp

import cigartree
import likelihood
import h20

if len(sys.argv) != 4:
    print('Usage: {} model.json time steps'.format(sys.argv[0]))
    sys.exit(1)

modelFilename, t, steps = sys.argv[1:]
with open(modelFilename, 'r') as f:
    modelJson = json.load (f)

t = float(t)
steps = int(steps)

alphabet, mixture, indelParams = likelihood.parseHistorianParams (modelJson)


transMat = h20.transitionMatrix(t,indelParams,alphabetSize=len(alphabet),steps=steps,norm=False,debug=2)

print (transMat)
