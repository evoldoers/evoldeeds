import sys
import json

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
seqs, distanceToParent, parentIndex, transCounts = cigartree.getHMMSummaries (treeStr, alignStr, alphabet)

#print(transCounts)

subll = likelihood.subLogLike (seqs, distanceToParent, parentIndex, *mixture[0])
subll_total = float (jnp.sum (subll))

transMat = jnp.stack ([h20.dummyRootTransitionMatrix()] + [h20.transitionMatrix(t,indelParams,alphabetSize=len(alphabet)) for t in distanceToParent[1:]], axis=0)
transMat = jnp.log (jnp.maximum (transMat, h20.smallest_float32))
transll = transCounts * transMat
#print(transMat)
#print(transCounts)
#print(transll)
transll_total = float (jnp.sum (transll))

print (json.dumps({'loglike':{'subs':subll_total,'indels':transll_total}, 'cigartree': ct}))
