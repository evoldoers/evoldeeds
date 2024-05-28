import jax
import jax.numpy as jnp
import optax
    
def optimize (loss_value_and_grad, params, prefix="", init_lr=1e-3, max_iter=1000, min_inc=1e-6, patience=10, use_jit=True, show_grads=False):
    optimizer = optax.adam(init_lr)
    opt_state = optimizer.init(params)

    best_ll = None
    best_params = params
    patience_counter = 0
    for iter in range(max_iter):
        ll, grads = loss_value_and_grad (params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        print ("%sIteration %d: loss %f" % (prefix,iter+1,ll))
        if show_grads:
            print (grads)
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
    
    return best_params, best_ll
