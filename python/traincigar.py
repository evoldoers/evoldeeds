import json
import jax.test_util
from jsonargparse import CLI

import jax
import jax.numpy as jnp

import optax

import logging

import likelihood
import dataset

def main (modelFile: str,
          treeFile: str = None,
          alignFile: str = None,
          dataDir: str = None,
          families: str = None,
          familiesFile: str = None,
          limitFamilies: int = None,
          reversible: bool = False,
          train: bool = True,
          init_lr: float = 1e-3,
          max_iter: int = 1000,
          min_inc: float = 1e-6,
          patience: int = 10,
          show_grads: bool = False,
          check_grads: bool = False,
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
        train: Train model
        init_lr: Initial learning rate
        max_iter: Maximum number of iterations
        min_inc: Minimum fractional increase in log-likelihood
        show_grads: Show gradients
        check_grads: Check gradients
    """

    # Read model and alphabet
    with open(modelFile, 'r') as f:
        modelJson = json.load (f)
    alphabet, mixture, indelParams = likelihood.parseHistorianParams (modelJson)
    mixture = [(subRate, jnp.log(rootProb)) for subRate, rootProb in mixture]

    assert len(mixture) == 1, "Only one mixture component is supported for substitution model"

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
        seqs, parentIndex, distanceToParent = dataset.loadTreeAndAlignment (treeFile, alignFile, alphabet)
        data = [(seqs, parentIndex, distanceToParent)]
    else:
        raise ValueError ('Either dataDir, or both treeFile and alignFile, must be specified')

    # Create loss function
    model_factory = likelihood.parametricReversibleSubModel if reversible else likelihood.parametricSubModel
    loss = dataset.createLossFunction (data, model_factory)

    if check_grads:
        def loss_wrapper (s, r):
            return loss ({ 'subrate': s, 'root': r })
        logging.warning("Checking gradients...")
        jax.test_util.check_grads(loss_wrapper, mixture[0], order=1)
        logging.warning("Gradients OK")

    # Initialize parameters of the model + optimizer.
    params = { 'subrate': mixture[0][0],
               'root': mixture[0][1] }

    # Training loop
    if train:
        loss_value_and_grad = jax.value_and_grad (loss)
        loss_value_and_grad = jax.jit (loss_value_and_grad)

        optimizer = optax.adam(init_lr)
        opt_state = optimizer.init(params)

        best_ll = None
        best_params = params
        patience_counter = 0
        for iter in range(max_iter):
            ll, grads = loss_value_and_grad (params)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            print ("Iteration %d: loss %f" % (iter+1,ll))
            if show_grads:
                print (alphabet, grads)
            inc = (best_ll - ll) / abs(best_ll) if best_ll is not None else 1
            if best_ll is None or ll > best_ll:
                best_params = params
                best_ll = ll
            if inc >= min_inc:
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        # Convert back to historian format, and output
        subRate, rootProb = model_factory (best_params['subrate'], best_params['root'])
        print (json.dumps (likelihood.toHistorianParams (alphabet, [(subRate, rootProb)], indelParams)))
    else:
        loss_jit = jax.jit(loss)
        ll = loss_jit(params)
        print("Loss (negative log-likelihood): %f" % ll)


if __name__ == '__main__':
    CLI(main, parser_mode='jsonnet')