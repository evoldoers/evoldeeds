import jax
import jax.numpy as jnp
import optax

def optimize (loss_value_and_grad, params, init_lr=1e-3, show_grads=False, **kwargs):
    optimizer = optax.adam(init_lr)
    opt_state = optimizer.init(params)
    def take_step (params, _nStep):
        nonlocal opt_state
        ll, grads = loss_value_and_grad (params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        if show_grads:
            print (grads)
        return params, ll
    return optimize_generic (take_step, params, **kwargs)

def optimize_generic (take_step, params, prefix="Iteration ", max_iter=1000, min_inc=1e-6, patience=10):
    best_ll = None
    best_params = params
    patience_counter = 0
    for iter in range(max_iter):
        next_params, ll = take_step (params, iter)
        print ("%s%d: loss %f" % (prefix,iter+1,ll))
        inc = (best_ll - ll) / abs(best_ll) if best_ll is not None else 1
        if best_ll is None or ll > best_ll:
            best_params = params
            best_ll = ll
        params = next_params
        if inc >= min_inc:
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    return best_params, best_ll
