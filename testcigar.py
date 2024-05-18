import sys
import json

import jax.numpy as jnp

import cigartree
import likelihood

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
seqs, distanceToParent, parentIndex, transCounts = cigartree.getHMMSummaries (treeStr, alignStr, alphabet)

#print(transCounts)

subll = likelihood.subLogLike (seqs, distanceToParent, parentIndex, *mixture[0])
subll_total = float (jnp.sum (subll))

print (json.dumps({'loglike':{'subs':subll_total}, 'cigartree': ct}))
