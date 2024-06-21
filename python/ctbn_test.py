import unittest

import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm

import ctbn

jax.config.update('jax_platform_name', 'cpu')

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

    # EQUILIBRIUM STATISTICS (single sequence)

    # For a single component, the partition function and its variational bound should be sum(exp(h))
    def test_telegraph1_partition (self):
        self.do_test_telegraph_partition(1)

    def do_test_telegraph_partition (self, K):
        C, params = telegraph(K=K)
        seq_mask, nbr_idx, nbr_mask, *_rest = ctbn.get_Markov_blankets(C)
        logZ_expected = K * ctbn.logsumexp(params['h'])
        logZ_exact = ctbn.ctbn_exact_log_Z(seq_mask, nbr_idx, nbr_mask, params)
        logZ_variational, theta = ctbn.ctbn_variational_log_Z(seq_mask, nbr_idx, nbr_mask, params)
        self.assertTrue (jnp.allclose(logZ_exact, logZ_expected))
        self.assertTrue (jnp.allclose(logZ_variational, logZ_expected))

    # For two components that are not in contact, the partition function and its variational bound should be sum(exp(-h))^2
    def test_telegraph2_partition (self):
        self.do_test_telegraph_partition(2)

    # For two components that are in contact, the exact partition function should be greater than its variational lower bound
    def test_ising2_partition (self):
        C, params = ising2()
        seq_mask, nbr_idx, nbr_mask, *_rest = ctbn.get_Markov_blankets(C)
        logZ_exact = ctbn.ctbn_exact_log_Z(seq_mask, nbr_idx, nbr_mask, params)
        logZ_variational, theta = ctbn.ctbn_variational_log_Z(seq_mask, nbr_idx, nbr_mask, params)
        self.assertTrue (jnp.all(logZ_exact > logZ_variational))

    # For a single component, the log-pseudolikelihood should be equal to the log-likelihood
    def test_telegraph1_pseudo (self):
        self.do_test_telegraph_pseudo ([0])
        self.do_test_telegraph_pseudo ([1])

    def do_test_telegraph_pseudo (self, xs):
        N = 2
        K = len(xs)
        C, params = telegraph(K=K)
        seq_mask, nbr_idx, nbr_mask, *_rest = ctbn.get_Markov_blankets(C)
        xs = jnp.array(xs)
        xidx = ctbn.seq_to_idx(xs,N)
        q_joint = ctbn.q_joint(nbr_idx, nbr_mask, params)
        q_eqm = ctbn.exact_eqm (q_joint)
        ll_exact_from_eqm = jnp.log(q_eqm[xidx])
        ll_exact = ctbn.ctbn_exact_log_marg (xs, seq_mask, nbr_idx, nbr_mask, params)
        ll_pseudo = ctbn.ctbn_pseudo_log_marg (xs, seq_mask, nbr_idx, nbr_mask, params)
        self.assertTrue (jnp.isclose(ll_exact, ll_exact_from_eqm))
        self.assertTrue (jnp.isclose(ll_exact, ll_pseudo))

    # For two components that are not in contact, the log-pseudolikelihood should be equal to the log-likelihood
    def test_telegraph2_pseudo (self):
        self.do_test_telegraph_pseudo ([0,1])
        self.do_test_telegraph_pseudo ([1,1])

    # TIME-DEPENDENT STATISTICS (two sequences)

    # For a single component, the ODEs for rho and mu should equal the exact results, and F should equal the log-likelihood
    def test_telegraph1_rho_mu (self):
#        self.do_test_telegraph1_rho_mu (0, 0)
        self.do_test_telegraph1_rho_mu (0, 1)
#        self.do_test_telegraph1_rho_mu (1, 0)
#        self.do_test_telegraph1_rho_mu (1, 1)

    def do_test_telegraph1_rho_mu (self, x, y):
        C, params = telegraph(K=1)
        seq_mask, nbr_idx, nbr_mask, *_rest = ctbn.get_Markov_blankets(C)
        T = 1.
        q = ctbn.q_joint(nbr_idx, nbr_mask, params)
        zero = [ctbn.ZeroSolution(2)]
        rho_ode = ctbn.solve_rho (0, jax.nn.one_hot(y,2), seq_mask, nbr_idx, nbr_mask, params, zero, zero, T)
        rho_exact = ctbn.ExactRho (q, T, x, y)
        self.assertTrue (self.close_over_domain (rho_exact, rho_ode, T))
        mu_ode = ctbn.solve_mu (0, jax.nn.one_hot(x,2), seq_mask, nbr_idx, nbr_mask, params, zero, [rho_ode], T)
        mu_exact = ctbn.ExactMu (q, T, x, y)
        self.assertTrue (self.close_over_domain (mu_exact, mu_ode, T))
        ll_exact = jnp.log(expm(T * q)[x, y])
        F = ctbn.solve_F (seq_mask, nbr_idx, nbr_mask, params, [mu_ode], [rho_ode], T)
#        print(f"F={F} ll_exact={ll_exact}")
        self.assertTrue (jnp.isclose (ll_exact, F, rtol=1e-2, atol=1e-1))

    def close_over_domain (self, a, b, T, a_label="exact", b_label="bound", rtol=1e-3, atol=1e-3, steps=10):
        ts = jnp.arange(steps+1) * T / steps
        a_t = jnp.array ([a.evaluate(t) for t in ts])
        b_t = jnp.array ([b.evaluate(t) for t in ts])
        pred = jnp.isclose (a_t, b_t, rtol=rtol, atol=atol)
        if not jnp.all(pred):
            for t, at, bt, p in zip(ts, a_t, b_t, pred):
                print(f"t={t}: {a_label}={at} {b_label}={bt} close={p}")
        return jnp.all(pred)

    # For a single component, the variational lower bound should be equal to the log-likelihood
    def test_telegraph1_variational (self):
        self.do_test_telegraph_variational ([0], [1], 1.0)
    
    def do_test_telegraph_variational (self, xs, ys, T):
        N = 2
        K = len(xs)
        C, params = telegraph(K=K)
        seq_mask, nbr_idx, nbr_mask, *_rest = ctbn.get_Markov_blankets(C)
        xs = jnp.array(xs)
        ys = jnp.array(ys)
        xidx = ctbn.seq_to_idx(xs,N)
        yidx = ctbn.seq_to_idx(ys,N)
        q_joint = ctbn.q_joint(nbr_idx, nbr_mask, params)
        ll_exact = jnp.log(expm(T * q_joint)[xidx, yidx])
        prng = jax.random.PRNGKey(42)
        log_elbo, (mu_elbo, rho_elbo) = ctbn.ctbn_variational_log_cond (prng, xs, ys, seq_mask, nbr_idx, nbr_mask, params, T)
        self.assertTrue (jnp.allclose(log_elbo, ll_exact, rtol=1e-2, atol=1e-1))
        q1 = ctbn.q_single (params)
        for k in range(K):
            mu_exact = ctbn.ExactMu(q1, T, xs[k], ys[k])
            rho_exact = ctbn.ExactRho(q1, T, xs[k], ys[k])
            self.assertTrue (self.close_over_domain(mu_exact, mu_elbo[k], T))
            self.assertTrue (self.close_over_domain(rho_exact, rho_elbo[k], T))

    # For two components that are not in contact, the variational lower bound should be equal to the log-likelihood
    def test_telegraph2_variational (self):
        self.do_test_telegraph_variational ([0,0], [1,1], 1.0)

    # For two components that are in contact, F should be a reasonably close lower bound for the log-likelihood
    def test_ising2_variational (self):
        self.do_test_ising2_variational ([0,1], [1,0], 1.0)

    def do_test_ising2_variational (self, xs, ys, T):
        C, params = ising2()
        seq_mask, nbr_idx, nbr_mask, *_rest = ctbn.get_Markov_blankets(C)
        xs = jnp.array(xs)
        ys = jnp.array(ys)
        xidx = ctbn.seq_to_idx(xs,2)
        yidx = ctbn.seq_to_idx(ys,2)
        q_joint = ctbn.q_joint(nbr_idx, nbr_mask, params)
        ll_exact = jnp.log(expm(T * q_joint)[xidx, yidx])
        prng = jax.random.PRNGKey(42)
        log_elbo, (mu_elbo, rho_elbo) = ctbn.ctbn_variational_log_cond (prng, xs, ys, seq_mask, nbr_idx, nbr_mask, params, T, min_inc=1e-6)
        self.assertTrue (ll_exact > log_elbo)
        dt_steps = 100
        ts = jnp.linspace(0,T,dt_steps+1)
        mu_t = jnp.array ([[mu.evaluate(t) for mu in mu_elbo] for t in ts])
        for t, mu in zip(ts,mu_t):
            print(f"t={t}",*list(f" P(x{i})={mu_i[1].item()}" for i,mu_i in enumerate(mu)))

    # For two components that are in contact, reproduce Figure 3(b)-(d) from Cohn et al (2010)

    # PARAMETER FITTING

    # For a single component, we should be able to recover h by maximizing the log-marginal of a simulated dataset
    # For two components that are not in contact, we should be able to recover h by maximizing the log-marginal of a simulated dataset
    # For two components that are in contact, we should be able to recover h and J by maximizing the log-marginal of a simulated dataset

    # For a single component, we should be able to recover h and S by maximizing the log-joint of a simulated dataset
    # For two components that are not in contact, we should be able to recover h and S by maximizing the log-joint of a simulated dataset
    # For two components that are in contact, we should be able to recover h, J, and S by maximizing the log-joint of a simulated dataset

if __name__ == '__main__':
    unittest.main()