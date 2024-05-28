import jax.numpy as jnp

t_min = 1e-9
def transitionMatrix (t, indelParams, **kwargs):
    t_safe = jnp.maximum (t, t_min)
    return jnp.where (t >= t_min,
                      transitionMatrix_unsafe (t_safe, indelParams),
                      jnp.eye(3))

def transitionMatrix_unsafe (t, indelParams):
    lam, mu, x, y = indelParams
    r = (lam + mu) / 2
    a = (x + y) / 2
    Pid = 1 - jnp.exp(-2*r*t)
    Pid_prime = 1 - (1 - jnp.exp(-2*r*t)) / (2*r*t)
    T00 = 1 - Pid*(1-Pid_prime*(1-a)/(4+4*a))
    T01 = (1-T00)/2
    T02 = T01
    E10 = 1-a + Pid_prime*a*(1-a)/(2+2*a) - Pid*(7-7*a)/8
    E11 = a + Pid_prime*a*a/(1-a*a) + Pid*(1-a)/2
    E12 = Pid_prime*a*a/(2+2*a) + Pid*(3-3*a)/8
    E1 = 1 + Pid_prime*a/(2-2*a)
    T10 = E10/E1
    T11 = E11/E1
    T12 = E12/E1
    T20 = T10
    T22 = T11
    T21 = T12
    return jnp.array ([[T00, T01, T02],
                       [T10, T11, T12],
                       [T20, T21, T22]])
