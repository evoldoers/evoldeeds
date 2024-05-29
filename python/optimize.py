from copy import deepcopy
import logging

import jax
import jax.numpy as jnp
import optax

def optimize (loss_value_and_grad, params, init_lr=1e-3, show_grads=False, **kwargs):
    optimizer = optax.adam(init_lr)
    opt_state = optimizer.init(params)
    def take_step (params, _nStep):
        nonlocal opt_state
        loss, grads = loss_value_and_grad (params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        if show_grads:
            logging.warning (grads)
        return params, loss
    return optimize_generic (take_step, params, **kwargs)

def optimize_generic (take_step, params, prefix="Iteration ", max_iter=1000, min_inc=1e-6, patience=10, verbose=True):
    best_loss = None
    best_params = deepcopy(params)
    patience_counter = 0
    last_loss = None
    for iter in range(max_iter):
        next_params, loss = take_step (params, iter)
        inc = (best_loss - loss) / abs(best_loss) if best_loss is not None else 2*min_inc
        if inc >= min_inc:
            patience_counter = 0
        else:
            patience_counter += 1
        change_desc = "first" if best_loss is None else "better" if inc >= min_inc else "underwhelming" if loss < best_loss else "same" if loss == best_loss else "stalled" if loss == last_loss else "rallying" if loss < last_loss else "worse"
        if patience_counter >= patience/2:
            change_desc += "; stopping" + ((" in %d" % (patience - patience_counter)) if patience_counter < patience else "")
        if verbose:
            logging.warning ("%s%d: loss %f (%s)" % (prefix,iter+1,loss,change_desc))
        if best_loss is None or loss < best_loss:
            best_params = deepcopy(params)
            best_loss = loss
        params = next_params
        last_loss = loss
        if patience_counter >= patience:
            break
    if verbose:
        logging.warning ("%s%d: best loss %f" % (prefix,iter+1,best_loss))
    return best_params, best_loss
