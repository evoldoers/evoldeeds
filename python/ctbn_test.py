import jax.numpy as jnp
import ctbn

# Tests
# For the two-component chain, use Example 1 from Cohn et al (2010)
def ising2 (q_same = 10, q_diff = 1):
    J_same = -jnp.log(q_same) / 2
    J_diff = -jnp.log(q_diff) / 2
    C = jnp.eye(2)
    S = jnp.array ([[0, 1], [1, 0]])
    J = jnp.array ([[J_same, J_diff], [J_diff, J_same]])
    h = jnp.zeros(2)
    params = { 'S': S, 'J': J, 'h': h }
    return C, params

# For single-component and independent K-component examples, use the telegraph process
def telegraph (K = 1, lambda1 = 1, lambda2 = 2):
    C = jnp.zeros((K, K))
    S = jnp.eye(2)
    J = jnp.zeros((2,2))
    h = -jnp.log(jnp.array([lambda2, lambda1]))  # lambda2 is the rate from state #2 -> #1, h[0] is the bias of state #1
    params = { 'S': S, 'J': J, 'h': h }
    return C, params

# - For a single component, rho and mu should be equal to the single-component posterior
# - For a single component, the F term should be equal to the log-likelihood
# - For a single component, the log-pseudolikelihood should be equal to the log-likelihood
# - For two components that are not in contact, F (and hence the log-pseudolikelihood) should be equal to the log-likelihood
# - For two components that are in contact, F should be a reasonably close lower bound for the log-likelihood
# - For two components that are in contact, reproduce Figure 3(b)-(d) from Cohn et al (2010)
