import json
from jsonargparse import CLI

import jax.numpy as jnp

import cigartree
import likelihood
import h20

def main (treeFilename: str,
          alignFilename: str,
          modelFilename: str,
          discretize: bool = False,
          ):
    """
    Compute the log likelihood of a tree given an alignment and a model.
    
    Args:
        treeFilename: Newick tree file
        alignFilename: FASTA alignment file
        modelFilename: Historian-format JSON file with model parameters
        discretize: Discretize branch lengths
    """
    with open(treeFilename, 'r') as f:
        treeStr = f.read()
    with open(alignFilename, 'r') as f:
        alignStr = f.read()
    with open(modelFilename, 'r') as f:
        modelJson = json.load (f)

    ct = cigartree.makeCigarTree (treeStr, alignStr)

    alphabet, mixture, indelParams = likelihood.parseHistorianParams (modelJson)
    seqs, nodeName, distanceToParent, parentIndex, transCounts = cigartree.getHMMSummaries (treeStr, alignStr, alphabet)

    subRate, rootProb = mixture[0]
    if discretize:
        discSubMatrix = likelihood.computeSubMatrixForDiscretizedTimes (subRate)
        subMatrix = likelihood.computeSubMatrixForDiscretizedBranchLengths (distanceToParent, discSubMatrix)
        subll = likelihood.subLogLikeForMatrices (seqs, parentIndex, subMatrix, rootProb)
    else:
        subll = likelihood.subLogLike (seqs, distanceToParent, parentIndex, subRate, rootProb)
    subll_total = float (jnp.sum (subll))

    transMat = jnp.stack ([h20.dummyRootTransitionMatrix()] + [h20.transitionMatrix(t,indelParams,alphabetSize=len(alphabet)) for t in distanceToParent[1:]], axis=0)
    transMat = jnp.log (jnp.maximum (transMat, h20.smallest_float32))
    transll = transCounts * transMat
    transll_total = float (jnp.sum (transll))

    print (json.dumps({'loglike':{'subs':subll_total,'indels':transll_total}, 'cigartree': ct}))

if __name__ == '__main__':
    CLI(main, parser_mode='jsonnet')