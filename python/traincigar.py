import json
from jsonargparse import CLI

import jax
import jax.numpy as jnp

import optax

import logging

import likelihood
import dataset
from optimize import optimize
from cherry import optimizeByCompositeEM

#jax.config.update("jax_debug_nans", True)

def main (modelFile: str,
          treeFile: str = None,
          alignFile: str = None,
          dataDir: str = None,
          families: str = None,
          familiesFile: str = None,
          limitFamilies: int = None,
          reversible: bool = False,
          km03: bool = False,
          omitIndels: bool = False,
          omitSubs: bool = False,
          train: bool = True,
          init_lr: float = 1e-3,
          max_iter: int = 1000,
          min_inc: float = 1e-6,
          patience: int = 10,
          show_grads: bool = False,
          use_jit: bool = True,
          cherry: bool = False,
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
        limitFamilies: Limit number of families
        reversible: Use reversible model
        km03: Use Knudsen-Miyamoto (2003) approximation to GGI model, rather than Holmes (2020)
        train: Train model
        init_lr: Initial learning rate
        max_iter: Maximum number of iterations
        min_inc: Minimum fractional increase in log-likelihood
        show_grads: Show gradients
        use_jit: Use JIT compilation
        cherry: use CherryML-style composite likelihood for training
    """

    # Read model and alphabet
    with open(modelFile, 'r') as f:
        modelJson = json.load (f)
    alphabet, mixture, indelParams, alnTypeLogits, colTypeLogits, colShape, nQuantiles = likelihood.parseHistorianParams (modelJson)

    # Convert rates and probabilities to exchangeabilities and logits
    indelParams = [likelihood.indelModelToLogits(*p) for p in indelParams]
    if reversible:
        mixture = [(likelihood.zeroDiagonal(likelihood.subMatrixToExchangeabilityMatrix(subRate,rootProb)),
                    likelihood.probsToLogits(rootProb)) for subRate, rootProb in mixture]
    else:
        mixture = [(likelihood.zeroDiagonal(subRate),
                    likelihood.probsToLogits(rootProb)) for subRate, rootProb in mixture]

    # Create dataset
    if dataDir is not None:
        if families is not None:
            families = families.split(',')
        elif familiesFile is not None:
            with open(familiesFile, 'r') as f:
                families = [line.strip() for line in f]
        else:
            logging.warning("Warning: no family list specified; using all families in %s" % dataDir)
        data = list (dataset.loadTreeFamData (dataDir, alphabet, families=families, limit=limitFamilies))
    elif treeFile is not None and alignFile is not None:
        data = [dataset.loadTreeAndAlignment (treeFile, alignFile, alphabet)]
    else:
        raise ValueError ('Either dataDir, or both treeFile and alignFile, must be specified')

    # Create loss function
    sub_model_factory = likelihood.parametricReversibleSubModel if reversible else likelihood.parametricSubModel
    ggi_model_factory = likelihood.createGGIModelFactory (sub_model_factory, nQuantiles)
    loss = dataset.createLossFunction (data, ggi_model_factory, includeSubs=not omitSubs, includeIndels=not omitIndels, useKM03=km03)

    jit = jax.jit if use_jit else lambda f: f

    # Initialize parameters of the model + optimizer.
    params = { 'indels': indelParams,
               'subs': [{'subrate':s,'rootlogits':r} for (s,r) in mixture],
               'alntypelogits': alnTypeLogits,
               'coltypelogits': colTypeLogits,
               'colshape': colShape }

    # Training
    if train:
        opt_args = {'init_lr':init_lr, 'max_iter':max_iter, 'min_inc':min_inc, 'patience':patience, 'show_grads':show_grads}
        if cherry:
            best_params, best_ll = optimizeByCompositeEM (data, ggi_model_factory, params, use_jit=use_jit, useKM03=km03, **opt_args)
        else:
            loss_value_and_grad = jit (jax.value_and_grad (loss))
            best_params, best_ll = optimize (loss_value_and_grad, params, **opt_args)

        # Convert back to historian format, and output
        subRate, rootProb, indelParams = ggi_model_factory (best_params)
        print (json.dumps (likelihood.toHistorianParams (alphabet, [(subRate, rootProb)], [indelParams], alnTypeLogits, colTypeLogits, colShape, nQuantiles)))
    else:
        loss = jit(loss)
        ll = loss(params)
        print("Loss (negative log-likelihood): %f" % ll)


if __name__ == '__main__':
    CLI(main, parser_mode='jsonnet')