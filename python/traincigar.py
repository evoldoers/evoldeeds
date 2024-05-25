import json
from jsonargparse import CLI

import jax
import jax.numpy as jnp

import cigartree
import likelihood

def main (treeFilename: str,
          alignFilename: str,
          modelFilename: str,
          discretize: bool = False,
          ):
    """
    Compute derivatives of log-likelihood for tree, alignment, and model.
    
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

    alphabet, mixture, _indelParams = likelihood.parseHistorianParams (modelJson)
    seqs, _nodeName, distanceToParent, parentIndex, _transCounts = cigartree.getHMMSummaries (treeStr, alignStr, alphabet)

    def loss (params):
        subRate = params['subrate']
        rootProb = params['rootprob']
        subRate, rootProb = likelihood.normalizeSubModel (subRate, rootProb)
        if discretize:
            discSubMatrix = likelihood.computeSubMatrixForDiscretizedTimes (subRate)
            subMatrix = likelihood.computeSubMatrixForDiscretizedBranchLengths (distanceToParent, discSubMatrix)
            subll = likelihood.subLogLikeForMatrices (seqs, parentIndex, subMatrix, rootProb)
        else:
            subll = likelihood.subLogLike (seqs, distanceToParent, parentIndex, subRate, rootProb)
        return -jnp.sum (subll)

    loss_value_and_grad = jax.value_and_grad (loss)
    loss_value_and_grad = jax.jit (loss_value_and_grad)

    params = {'subrate': mixture[0][0], 'rootprob': mixture[0][1]}

    print (loss_value_and_grad (params))

if __name__ == '__main__':
    CLI(main, parser_mode='jsonnet')