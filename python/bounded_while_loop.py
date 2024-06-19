# Patrick Kidger's bounded_while_loop
# https://github.com/google/jax/issues/8239#issue-1027850539

import jax
import jax.lax as lax

def bounded_while_loop(cond_fun, body_fun, init_val, max_steps):
    """API as `lax.while_loop`, except that it takes an integer `max_steps` argument."""
    if not isinstance(max_steps, int) or max_steps < 0:
        raise ValueError("max_steps must be a non-negative integer")
    if max_steps == 0:
        return init_val
    if max_steps & (max_steps - 1) != 0:
        raise ValueError("max_steps must be a power of two")

    init_data = (cond_fun(init_val), init_val)
    _, val = _while_loop(cond_fun, body_fun, init_data, max_steps)
    return val

def _while_loop(cond_fun, body_fun, data, max_steps):
    if max_steps == 1:
        pred, val = data
        new_val = body_fun(val)
        keep = lambda a, b: lax.select(pred, a, b)
        new_val = jax.tree_map(keep, new_val, val)
        return cond_fun(new_val), new_val
    else:

        def _call(_data):
            return _while_loop(cond_fun, body_fun, _data, max_steps // 2)

        def _scan_fn(_data, _):
            _pred, _ = _data
            return lax.cond(_pred, _call, lambda x: x, _data), None

        return lax.scan(_scan_fn, data, xs=None, length=2)[0]