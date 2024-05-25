import sys
import json

import jax
import jax.numpy as jnp

import cigartree
import likelihood
import h20

if len(sys.argv) != 4:
    print('Usage: {} tree.nh align.fa model.json'.format(sys.argv[0]))
    sys.exit(1)

treeFilename, alignFilename, modelFilename = sys.argv[1:]
with open(treeFilename, 'r') as f:
    treeStr = f.read()
with open(alignFilename, 'r') as f:
    alignStr = f.read()
with open(modelFilename, 'r') as f:
    modelJson = json.load (f)

ct = cigartree.makeCigarTree (treeStr, alignStr)

alphabet, mixture, indelParams = likelihood.parseHistorianParams (modelJson)
seqs, nodeName, distanceToParent, parentIndex, transCounts = cigartree.getHMMSummaries (treeStr, alignStr, alphabet)

def loss (params):
    subRate = params['subrate']
    rootProb = params['rootprob']
    return -jnp.sum (likelihood.subLogLike (seqs, distanceToParent, parentIndex, subRate, rootProb))

loss_value_and_grad = jax.value_and_grad (loss)
loss_value_and_grad = jax.jit (loss_value_and_grad)

params = {'subrate': mixture[0][0], 'rootprob': mixture[0][1]}

print (loss_value_and_grad (params))
