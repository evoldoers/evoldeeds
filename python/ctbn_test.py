import jax.numpy as jnp
import ctbn

import unittest

# Tests for continuous-time Bayes network inference algorithms
# For the two-component chain, use Example 1 from Cohn et al (2010)
def ising2 (q_same = 10, q_diff = 1):
    J_same = jnp.log(q_same) / 2
    J_diff = jnp.log(q_diff) / 2
    C = jnp.ones((2,2)) - jnp.eye(2)
    S = jnp.array ([[0, 1], [1, 0]])
    J = jnp.array ([[J_same, J_diff], [J_diff, J_same]])
    h = jnp.zeros(2)
    params = { 'S': S, 'J': J, 'h': h }
    return C, params

# For single-component and independent K-component examples, use the telegraph process
def telegraph (K = 1, lambda1 = 1, lambda2 = 2):
    C = jnp.zeros((K, K))
    S = jnp.array ([[0, 1], [1, 0]])
    J = jnp.zeros((2,2))
    h = jnp.log(jnp.array([lambda2, lambda1]))  # lambda2 is the rate from state #2 -> #1, h[0] is the bias of state #1
    params = { 'S': S, 'J': J, 'h': h }
    return C, params

class TestCTBN (unittest.TestCase):
    # For a single component, the partition function should be sum(exp(h))
    def test_telegraph1_partition (self):
        self.do_test_telegraph_partition(1)

    def do_test_telegraph_partition (self, K):
        C, params = telegraph(K=K)
        seq_mask, nbr_idx, nbr_mask, *_rest = ctbn.get_Markov_blankets(C)
        logZ_expected = K * ctbn.logsumexp(params['h'])
        logZ_exact = ctbn.ctbn_exact_log_Z(seq_mask, nbr_idx, nbr_mask, params)
        self.assertTrue (jnp.allclose(logZ_exact, logZ_expected))

    # For two components that are not in contact, the partition function should be sum(exp(-h))^2
    def test_telegraph2_partition (self):
        self.do_test_telegraph_partition(2)

    # For two components that are in contact, the exact partition function should be greater than its variational lower bound
    def test_ising2_partition (self):
        C, params = ising2()
        seq_mask, nbr_idx, nbr_mask, *_rest = ctbn.get_Markov_blankets(C)
        logZ_exact = ctbn.ctbn_exact_log_Z(seq_mask, nbr_idx, nbr_mask, params)
        logZ_variational, theta = ctbn.ctbn_variational_log_Z(seq_mask, nbr_idx, nbr_mask, params)
        self.assertTrue (jnp.all(logZ_exact > logZ_variational))

    # For a single component, the partition function should be equal to its variational lower bound AND its pseudolikelihood
    # For a single component, rho and mu should be equal to the exact posterior
    # For a single component, the F term should be equal to the log-likelihood
    # For a single component, the log-pseudolikelihood should be equal to the log-likelihood

    # For two components that are not in contact, F (and hence the log-pseudolikelihood) should be equal to the log-likelihood
    # For two components that are not in contact, the partition function should be equal to its variational lower bound AND its pseudolikelihood

    # For two components that are in contact, F should be a reasonably close lower bound for the log-likelihood
    # For two components that are in contact, reproduce Figure 3(b)-(d) from Cohn et al (2010)

    # For a single component, we should be able to recover h by maximizing the log-marginal of a simulated dataset
    # For two components that are not in contact, we should be able to recover h by maximizing the log-marginal of a simulated dataset
    # For two components that are in contact, we should be able to recover h and J by maximizing the log-marginal of a simulated dataset

    # For a single component, we should be able to recover h and S by maximizing the log-joint of a simulated dataset
    # For two components that are not in contact, we should be able to recover h and S by maximizing the log-joint of a simulated dataset
    # For two components that are in contact, we should be able to recover h, J, and S by maximizing the log-joint of a simulated dataset

if __name__ == '__main__':
    unittest.main()