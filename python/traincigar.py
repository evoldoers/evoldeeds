import json
from jsonargparse import CLI

import jax
import jax.numpy as jnp

import cigartree
import likelihood
import dataset

def main (modelFile: str,
          treeFile: str = None,
          alignFile: str = None,
          dataDir: str = None,
          families: str = None,
          familiesFile: str = None,
          ):
    """
    Compute derivatives of log-likelihood for tree, alignment, and model.
    
    Args:
        modelFile: Historian-format JSON file with model parameters
        treeFile: Newick tree file
        alignFile: FASTA alignment file
        dataDir: Directory with tree and alignment files (suffices .nh and .aa.fasta)
        families: Comma-separated list of families
        familiesFile: File with list of families, one per line
        discretize: Discretize branch lengths
    """

    # Read model and alphabet
    with open(modelFile, 'r') as f:
        modelJson = json.load (f)
    alphabet, mixture, _indelParams = likelihood.parseHistorianParams (modelJson)

    # Create dataset
    if dataDir is not None:
        if families is not None:
            families = families.split(',')
        elif familiesFile is not None:
            with open(familiesFile, 'r') as f:
                families = [line.strip() for line in f]
        else:
            raise ValueError ('Either families or familiesFile must be specified')
        alphabet = likelihood.getAlphabetFromModel (modelFile)
        dataset = list (dataset.loadMultipleTreesAndAlignments (dataDir, dataDir, alphabet, families))
    else:
        seqs, parentIndex, distanceToParent = dataset.loadTreeAndAlignment (treeFile, alignFile, alphabet)
        dataset = [(seqs, parentIndex, distanceToParent)]

    # Create loss function
    loss = dataset.createLossFunction (dataset)
    loss_value_and_grad = jax.value_and_grad (loss)
    loss_value_and_grad = jax.jit (loss_value_and_grad)

    # TODO:
    # Use a parameterization that guarantees rates and probabilities in correct domain
    # Add an optimizer
    # Print out final parameterization

    params = {'subrate': mixture[0][0], 'rootprob': mixture[0][1]}

    print (loss_value_and_grad (params))

if __name__ == '__main__':
    CLI(main, parser_mode='jsonnet')