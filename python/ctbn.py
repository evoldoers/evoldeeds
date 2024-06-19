import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm
import diffrax

smallest_float32 = jnp.finfo('float32').smallest_normal

def product(x,axis=None,keepdims=False):
    return jnp.exp(jnp.sum(safe_log(x),axis=axis,keepdims=keepdims))

def offdiag_mask (N):
    return jnp.ones((N,N)) - jnp.eye(N)

def safe_log (x):
    return jnp.log (jnp.where (x == 0, smallest_float32, x))

def symmetrise (matrix):
    return (matrix + matrix.swapaxes(-1,-2)) / 2

def row_normalise (matrix):
    return matrix - jnp.diag(jnp.sum(matrix, axis=-1))

def round_up_to_power (x, base=2):
    if base == 2:  # avoid doubling length due to precision errors
        x = 1 << (len-1).bit_length()
    else:
        x = int (jnp.ceil (base ** jnp.ceil (jnp.log(len) / jnp.log(base))))
    return x

def logsumexp (x, axis=None, keepdims=False):
    max_x = jnp.max (x, axis=axis, keepdims=True)
    return jnp.log(jnp.sum(jnp.exp(x - max_x), axis=axis, keepdims=keepdims)) + max_x

# Implements the algorithm from Cohn et al (2010), for protein Potts models parameterized by contact & coupling matrices
# Cohn et al (2010), JMLR 11:93. Mean Field Variational Approximation for Continuous-Time Bayesian Networks

# Model:
# K components, each with N states
# C = symmetric binary contact matrix (K*K). For i!=j, C_ij=C_ji=1 if i and j are in contact, 0 otherwise. C_ii=0
# S = symmetric exchangeability matrix (N*N). For i!=j, S_ij=S_ji=rate of substitution from i<->j if i and j are equiprobable. S_ii = -sum_j S_ij
# J = symmetric coupling matrix (N*N). For i!=j, J_ij=J_ji=interaction strength between components i and j. J_ii = 0
# h = bias vector (N). h_i=bias of state i

# Since C is a sparse matrix, with each component having at most M<<K neighbors, we will represent it compactly as follows
# nbr_idx = sparse neighbor matrix (M*L). nbr_idx[i,n] is the index of the n-th neighbor of component i
# nbr_mask = sparse neighbor flag matrix (M*L). nbr_mask[i,n] is 1 if nbr_idx[i,n] is a real neighbor, 0 otherwise

def normalise_ctbn_params (params):
    return { 'S' : symmetrise(row_normalise(jnp.abs(params['S']))),
             'J' : symmetrise(params['J']),
             'h' : params['h'] }

# Endpoint-conditioned variational approximation:
# mu = (K,N) matrix of mean-field probabilities
# rho = (K,N) matrix where entry (i,x_i) is the probability of reaching the final state given that component #i is in state x_i

# Rate for substitution x_i->y_i
# i = 1..K
# x = (K,) vector of integers from 0..N-1
# y_i = integer from 0..N-1
def q_k (i, x, y_i, nbr_idx, nbr_mask, params):
    S = params['S']
    J = params['J']
    h = params['h']
    return S[x[i],y_i] * jnp.exp (-h[y_i] - 2*jnp.dot (nbr_mask[i], J[y_i,x[nbr_idx[i]]]))

# Mean-field averaged rates for a continuous-time Bayesian network
# Returns (A,N,N) matrix where entry (a,x_{idx[a]},y_{idx[a]}) is mean-field averaged rate matrix for component idx[a]
def q_bar (idx, nbr_idx, nbr_mask, params, mu):
    N = mu.shape[-1]
    S = params['S']
    J = params['J']
    h = params['h']
    exp_2J = jnp.exp (2 * J)  # (y_i,x_k)
    exp_2JC = exp_2J[None,None,:,:] ** nbr_mask[idx,:,None,None]  # (a,k,y_i,x_{nbr_k})
    mu_exp_2JC = jnp.einsum ('akx,kyx->aky', mu[nbr_idx[idx]], exp_2JC)  # (a,k,y_i)
    S = S * offdiag_mask(N)
    return S[None,:,:] * jnp.exp(h)[None,None,:] * product(mu_exp_2JC,axis=-2,keepdims=True)  # (a,x_i,y_i)

# Returns (M,N,N,N) tensor where entry (j,x_j,x_i,y_i) is the mean-field averaged rate x_i->y_i, conditioned on component nbr_idx[i,j] being in state x_{nbr_idx[i,j]}
# NB only valid for x_i != y_i
def q_bar_cond (i, nbr_idx, nbr_mask, params, mu):
    M = nbr_idx.shape[-1]
    N = mu.shape[-1]
    S = params['S']
    J = params['J']
    h = params['h']
    nonself_nbr_mask = offdiag_mask(M) * jnp.outer(nbr_mask[i],nbr_mask[i])  # (j,k)
    cond_energy = nbr_mask[i,:,None,None] * J[None,:,:]  # (j,x_{nbr_j},y_i)
    exp_2J = jnp.exp (2 * J)  # (N,N)
    exp_2JC = exp_2J[None,None,:,:] ** nonself_nbr_mask[:,:,None,None]  # (j,k,y_i,x_{nbr_k})
    mu_exp_JC = jnp.einsum ('kx,jkyx->jky', mu[nbr_idx[i]], exp_2JC)  # (j,k,y_i)
    S = S * offdiag_mask(N)
    return S[None,None,:,:] * jnp.exp(h)[None,None,None,:] * jnp.exp(-2*cond_energy)[:,:,None,:] * product(mu_exp_JC,axis=-2)[:,None,None,:]  # (j,x_{nbr_j},x_i,y_i)

# Geometrically-averaged mean-field rates for a continuous-time Bayesian network
# Returns (A,N,N) matrix where entry (a,x_{idx[a]},y_{idx[a]}) is geometrically-averaged mean-field rate matrix for component idx[a]
# NB only valid for x_i != y_i
def q_tilde (idx, nbr_idx, nbr_mask, params, mu):
    N = mu.shape[-1]
    S = params['S']
    J = params['J']
    h = params['h']
    mean_energy = jnp.einsum ('akx,ak,yx->ay', mu[nbr_idx[idx,:]], nbr_mask[idx,:], J)  # (a,y_i)
    S = S * offdiag_mask(N)
    return S[None,:,:] * jnp.exp(h+2*mean_energy)[:,None,:]  # (a,x_i,y_i)

# Returns (M,N,N,N) matrix where entry (j,x_j,x_i,y_i) is the geometrically-averaged rate x_i->y_i, conditioned on component nbr_idx[i,j] being in state x_{nbr_idx[i,j]}
# NB only valid for x_i != y_i
def q_tilde_cond (i, nbr_idx, nbr_mask, params, mu):
    M = nbr_idx.shape[-1]
    N = mu.shape[-1]
    S = params['S']
    J = params['J']
    h = params['h']
    nonself_nbr_mask = offdiag_mask(M) * jnp.outer(nbr_mask[i],nbr_mask[i])  # (j,k)
    cond_energy = nbr_mask[i,:,None,None] * J[None,:,:]  # (j,x_{nbr_j},y_i)
    mean_energy = jnp.einsum ('kx,jk,yx->jy', mu[nbr_idx[i]], nonself_nbr_mask, J)  # (j,y_i)
    S = S * offdiag_mask(N)
    return S[None,None,:,:] * jnp.exp(h)[None,None,None,:] * jnp.exp(2*cond_energy)[:,:,None,:] * jnp.exp(2*mean_energy)[:,None,None,:]  # (j,x_{nbr_j},x_i,y_i)

# Rate matrix for a single component, q_{xy} = S_{xy}
# S: (N,N)
# h: (N,)
def q_single (params):
    S = params['S']
    h = params['h']
    N = S.shape[0]
    return S * offdiag_mask(N) * jnp.exp(h)[None,:]

# Amalgamated (joint) rate matrix for all components
# Note this is big: (N^K,N^K)
def q_joint (nbr_idx, nbr_mask, params):
    N = params['S'].shape[0]
    K,M = nbr_idx.shape
    def get_components (x):
        return jnp.array ([(x//(N**j))%N for j in range(K)])
    def get_rate (xs, ys):
        diffs = jnp.where (xs == ys, 0, 1)
        i = jnp.argmax (diffs)
        return jnp.where (jnp.sum(diffs) == 1, q_k(i, xs, ys[i], nbr_idx, nbr_mask, params), 0)
    states = jax.vmap (get_components)(jnp.arange(N**K))
    Q = jax.vmap (lambda x: jax.vmap (lambda y: get_rate(x,y))(states))(states)
    return row_normalise(Q)

# gamma
# Returns (A,N,N) matrix where entry (k,x_{idx[a]},y_{idx[a]}) is the joint probability of transition x_{idx[a]}->y_{idx[a]} for component idx[a]
def gamma (idx, nbr_idx, nbr_mask, params, mu, rho):
    return jnp.einsum ('ax,axy,ay,ay->axy', mu[idx], q_tilde(idx,nbr_idx,nbr_mask,params,mu), rho[idx], 1/mu[idx])

# Returns (N,) vector
def psi (i, nbr_idx, nbr_mask, params, mu, rho):
    gammas = gamma(nbr_idx[i], nbr_idx, nbr_mask, params, mu, rho)  # (M,N,N)
    qbar_cond = q_bar_cond(i,nbr_idx,nbr_mask,params,mu)  # (M,N,N,N)
    qtilde_cond = q_tilde_cond(i,nbr_idx,nbr_mask,params,mu)  # (M,N,N,N)
    log_qtilde_cond = -safe_log (jnp.where (qtilde_cond < 0, 1, qtilde_cond))  # (M,N,N,N)
    return jnp.einsum('jy,jxyz,j->x',mu[nbr_idx[i]],qbar_cond,nbr_mask[i]) + jnp.einsum('jxy,jxyz,j->x',gammas,log_qtilde_cond,nbr_mask[i])

def rho_deriv (i, nbr_idx, nbr_mask, params, mu, rho):
    K = mu.shape[0]
    qbar = q_bar(jnp.array([i]), nbr_idx, nbr_mask, params, mu)[0,:,:]  # (N,N)
    qbar_diag = jnp.einsum ('xy->x', qbar)  # (N,)
    _psi = psi(i, nbr_idx, nbr_mask, params, mu, rho)  # (N,)
    qtilde = q_tilde(jnp.array([i]), nbr_idx, nbr_mask, params, mu)  # (1,N,N)
    rho_deriv_i = -rho[i,:] * (qbar_diag + _psi) - jnp.einsum ('y,xy->x', rho[i,:], qtilde[0,:,:])  # (N,)
    return rho_deriv_i

def mu_deriv (i, nbr_idx, nbr_mask, params, mu, rho):
    K = mu.shape[0]
    _gamma = gamma(jnp.array([i]), nbr_idx, nbr_mask, params, mu, rho)[0,:,:]  # (N,N)
    mu_deriv_i = jnp.einsum('yx->x',_gamma) - jnp.einsum('xy->y',_gamma)  # (N,)
    return mu_deriv_i

def F_deriv (seq_mask, nbr_idx, nbr_mask, params, mu, rho):
    K, N = mu.shape
    idx = jnp.arange(K)
    qbar = q_bar(idx, nbr_idx, nbr_mask, params, mu)  # (K,N,N)
    qtilde = q_tilde(idx, nbr_idx, nbr_mask, params, mu)  # (K,N,N)
    _gamma = gamma (idx, nbr_idx, nbr_mask, params, mu, rho)  # (K,N,N)
    mask = seq_mask[:,None,None] * offdiag_mask(N)[None,:,:]  # (K,N,N)
    log_qtilde = safe_log(jnp.where(mask,qtilde,1))
    gamma_coeff = log_qtilde + 1 + safe_log(mu)[:,:,None] - safe_log(_gamma)
    dF = -jnp.einsum('ix,ixy,ixy->',mu,qbar,mask) + jnp.einsum('ixy,ixy,ixy->',_gamma,gamma_coeff,mask)
    return dF

# Exact posterior for a complete rate matrix
class ExactRho:
    def __init__ (self, q, T, x, y):
        N = q.shape[0]
        assert q.shape == (N,N)
        self.N = N
        self.q = q
        self.T = T
        self.x = x
        self.y = y
        self.exp_qT = expm(q*T) [x, y]

    def evaluate (self, t):
        rho = expm(self.q*(self.T-t)) [:, self.y]  # (N,)
        return rho

class ExactMu (ExactRho):
    def evaluate (self, t):
        rho = super().evaluate (t)
        exp_qt = expm(self.q*t) [self.x, :]  # (N,)
        mu = exp_qt * rho / self.exp_qT
        return mu

# helper to evaluate mu and rho from arrays of Solution-like objects
def eval_mu_rho (mu_solns, rho_solns, t):
    mu = jnp.stack ([mu_soln.evaluate(t) for mu_soln in mu_solns])
    rho = jnp.stack ([rho_soln.evaluate(t) for rho_soln in rho_solns])
    return mu, rho

# wrappers for diffrax
def F_term (t, F_t, args):
    seq_mask, nbr_idx, nbr_mask, params, mu_solns, rho_solns = args
    mu, rho = eval_mu_rho (mu_solns, rho_solns, t)
    return F_deriv (seq_mask, nbr_idx, nbr_mask, params, mu, rho)

def solve_F (seq_mask, nbr_idx, nbr_mask, params, mu_solns, rho_solns, T, rtol=1e-3, atol=1e-6):
    term = diffrax.ODETerm (F_term)
    solver = diffrax.Dopri5()
    controller = diffrax.PIDController (rtol=rtol, atol=atol)
    return diffrax.diffeqsolve (term, solver, t0=0, t1=T, dt0=None, y0=0,
                                args=(seq_mask, nbr_idx, nbr_mask, params, mu_solns, rho_solns),
                                stepsize_controller = controller)

def make_rho_term (i, seq_mask):
    def rho_term (t, rho_i_t, args):
        nbr_idx, nbr_mask, params, mu_solns, rho_solns = args
        mu, rho = eval_mu_rho (mu_solns, rho_solns, t)
        rho = rho.at(i).set(rho_i_t)
        return seq_mask[i] * rho_deriv (i, nbr_idx, nbr_mask, params, mu, rho)
    return rho_term

def solve_rho (i, rho_i_T, seq_mask, nbr_idx, nbr_mask, params, mu_solns, rho_solns, T, rtol=1e-3, atol=1e-6):
    N = params['S'].shape[0]
    term = diffrax.ODETerm (make_rho_term(i, seq_mask))
    solver = diffrax.Dopri5()
    controller = diffrax.PIDController (rtol=rtol, atol=atol)
    return diffrax.diffeqsolve (term, solver, t0=T, t1=0, dt0=None, y0 = rho_i_T,
                                args=(nbr_idx, nbr_mask, params, mu_solns, rho_solns),
                                stepsize_controller = controller,
                                saveat = diffrax.SaveAt(dense=True))

def make_mu_term (i, seq_mask):
    def mu_term (t, mu_i_t, args):
        nbr_idx, nbr_mask, params, mu_solns, rho_solns = args
        mu, rho = eval_mu_rho (mu_solns, rho_solns, t)
        mu = mu.at(i).set(mu_i_t)
        return seq_mask[i] * mu_deriv (i, nbr_idx, nbr_mask, params, mu, rho)
    return mu_term

def solve_mu (i, mu_i_0, seq_mask, nbr_idx, nbr_mask, params, mu_solns, rho_solns, T, rtol=1e-3, atol=1e-6):
    term = diffrax.ODETerm (make_mu_term(i, seq_mask))
    solver = diffrax.Dopri5()
    controller = diffrax.PIDController (rtol=rtol, atol=atol)
    return diffrax.diffeqsolve (term, solver, t0=0, t1=T, dt0=None, y0=mu_i_0,
                                args=(nbr_idx, nbr_mask, params, mu_solns, rho_solns),
                                stepsize_controller = controller,
                                saveat = diffrax.SaveAt(dense=True))

# Calculate the variational likelihood for an endpoint-conditioned continuous-time Bayesian network
# This is a strict lower bound for log P(X_T=ys|X_0=xs,T,params)
def ctbn_variational_log_cond (xs, ys, seq_mask, nbr_idx, nbr_mask, params, T, min_inc=1e-3, max_updates=3):
    K = nbr_idx.shape[0]
    N = params['S'].shape[0]
    params = normalise_ctbn_params (params)
    # initialize component-indexed arrays of mu, rho solutions, using single-component posteriors
    # TODO: replace these with diffrax AbstractPath's generated by diffeqsolve with zeros for mu & rho, to avoid recompilation
    q = q_single (params)
    rho_solns = [ExactRho (q, T, xs[i], ys[i]) for i in range(K)]
    mu_solns = [ExactMu (q, T, xs[i], ys[i]) for i in range(K)]
    # initialize F_current as variational bound for initial mu and rho, and F_prev as -Infinity
    F_prev = -jnp.inf
    F_current = solve_F (seq_mask, nbr_idx, nbr_mask, params, mu_solns, rho_solns, T)
    # create arrays of boundary conditions for mu(0) and rho(T)
    mu_0 = jnp.eye(N)[xs]
    rho_T = jnp.eye(N)[ys]
    # while (F_current - F_prev)/F_prev > minimum relative increase:
    #  loop over component indices i:
    #   solve rho and then mu for component i, using diffrax, and replace single-component posteriors with diffrax Solution's
    #   F_prev <- F_current, F_current <- new variational bound
    def while_cond_fun (args):
        F_prev, (F_current, rho_solns, mu_solns), (F_best, rho_best, mu_best), n_updates = args
        return n_updates < max_updates and F_current > F_prev and jnp.abs((F_current - F_prev) / F_prev) > min_inc
    def while_body_fun (args):
        F_prev, (F_current, rho_solns, mu_solns), (F_best, rho_best, mu_best), n_updates = args
        F_prev = F_best
        F_current = solve_F (seq_mask, nbr_idx, nbr_mask, params, mu_solns, rho_solns, T)
        rho_solns, mu_solns = jax.lax.fori_loop (0, N, loop_body_fun, (rho_solns, mu_solns))
        old_best = lambda: (F_best, rho_best, mu_best)
        new_best = lambda: (F_current, rho_solns, mu_solns)
        F_best, rho_best, mu_best = jax.lax.cond (F_current > F_best, new_best, old_best)
        return F_prev, (F_current, rho_solns, mu_solns), (F_best, rho_best, mu_best), n_updates + 1
    def loop_body_fun (i, args):
        rho_solns, mu_solns = args
        i = N - 1 - i  # finish with component #0
        new_rho_i = solve_rho (i, rho_T[i,:], seq_mask, nbr_idx, nbr_mask, params, mu_solns, rho_solns, T)
        rho_solns = [new_rho_i if k==i else old_rho_i for k, old_rho_i in enumerate(rho_solns)]
        new_mu_i = solve_mu (i, mu_0[i,:], seq_mask, nbr_idx, nbr_mask, params, mu_solns, rho_solns, T)
        mu_solns = [new_mu_i if k==i else old_mu_i for k, old_mu_i in enumerate(mu_solns)]
        return rho_solns, mu_solns
    init_state = F_current, rho_solns, mu_solns
    _F_prev, _final_state, best_state, _final_n_updates = jax.lax.while_loop (while_cond_fun, while_body_fun, (F_prev, init_state, init_state, 0))
    return best_state

# Given sequences xs,ys and contact matrix C, return padded xs,ys along with seq_mask,nbr_idx,nbr_mask
def get_Markov_blankets (xs, ys, C, K=None, M=None):
    K_prepad = len(xs)
    assert len(ys) == K_prepad, "xs and ys must have the same length"
    assert C.shape == (K_prepad,K_prepad), "C must be a square matrix of the same size as xs"
    if K is None:
        K = round_up_to_power (K_prepad)
    nbr_idx_list = [C[i,:].nonzero() for i in range(K_prepad)]
    if M is None:
        M = round_up_to_power (max ([len(nbr_idx) for nbr_idx in nbr_idx_list]))
    else:
        assert M >= max ([len(nbr_idx) for nbr_idx in nbr_idx_list]), "M must be at least as large as the largest number of neighbors"
    seq_mask = jnp.array ([i < K_prepad for i in range(K)])
    nbr_mask = jnp.array ([[1] * len(nbr_idx) + [0] * (M - len(nbr_idx)) for nbr_idx in nbr_idx_list] + [[0] * M] * (K - K_prepad))
    nbr_idx = jnp.array ([nbr_idx + [0] * (M - len(nbr_idx)) for nbr_idx in nbr_idx_list])
    xs = xs + [0] * (K - K_prepad)
    ys = ys + [0] * (K - K_prepad)
    return xs, ys, seq_mask, nbr_idx, nbr_mask

# Weak L2 regularizer for J and h
def ctbn_param_regularizer (params, alpha=1e-4):
    return alpha * (jnp.sum (params['J']**2) + jnp.sum (params['h']**2))

# Log-pseudolikelihood for a continuous-time Bayesian network
def ctbn_pseudo_log_marg (xs, seq_mask, nbr_idx, nbr_mask, params, min_inc=1e-3, max_updates=3):
    K = nbr_idx.shape[0]
    N = params['S'].shape[0]
    params = normalise_ctbn_params (params)
    E_iy = params['h'][None,None,:] + jnp.einsum('ijy,ij->iy',params['J'][nbr_idx,:],nbr_mask)  # (K,N)
    log_Zi = logsumexp (E_iy, axis=-1)  # (K,)
    L_i = E_iy[jnp.arange(K),xs] - log_Zi  # (K,N)
    return jnp.sum (L_i * seq_mask)

# Mean-field approximation to log partition function of continuous-time Bayesian network
def ctbn_mean_field_log_Z (seq_mask, nbr_idx, nbr_mask, params, theta):
    E = jnp.einsum('ix,x->',theta,params['h'][None,:]) + jnp.einsum('ix,ijy,xy,ij->',theta,theta[nbr_idx,:],params['J'],nbr_mask)
    H = -jnp.einsum('ix->',theta * jnp.log(theta))
    return E + H

# Variational lower bound for log partition function of continuous-time Bayesian network
def ctbn_variational_log_Z (seq_mask, nbr_idx, nbr_mask, params, min_inc=1e-3, max_updates=3):
    K = nbr_idx.shape[0]
    N = params['S'].shape[0]
    params = normalise_ctbn_params (params)
    theta = jnp.repeat (jax.nn.softmax (params['h'])[None,:], K)  # (K,N)
    current_log_Z = ctbn_mean_field_log_Z (seq_mask, nbr_idx, nbr_mask, params, theta)
    prev_log_Z = -jnp.inf
    def while_cond_fun (args):
        prev_log_Z, (current_log_Z, current_theta), (best_log_Z, best_theta), n_updates = args
        return n_updates < max_updates and current_log_Z > prev_log_Z and jnp.abs((current_log_Z - prev_log_Z) / prev_log_Z) > min_inc
    def while_body_fun (args):
        prev_log_Z, (current_log_Z, current_theta), (best_log_Z, best_theta), n_updates = args
        prev_log_Z = current_log_Z
        theta = jax.nn.softmax (params['h'][None,:] + 2 * jnp.einsum('ijy,xy,ij->ix',theta[nbr_idx,:],params['J'],nbr_mask))
        current_log_Z = ctbn_mean_field_log_Z (seq_mask, nbr_idx, nbr_mask, params, theta)
        best_theta = jnp.where (current_log_Z > best_log_Z, theta, best_theta)
        best_log_Z = jnp.maximum (current_log_Z, best_log_Z)
        return prev_log_Z, (current_log_Z, theta), (best_log_Z, best_theta), n_updates + 1
    init_state = current_log_Z, theta
    _prev_log_Z, _final_state, best_state, _final_n_updates = jax.lax.while_loop (while_cond_fun, while_body_fun, (prev_log_Z, init_state, init_state, 0))
    return best_state

# Unnormalized log-marginal for a continuous-time Bayesian network
def ctbn_log_marg_unnorm (xs, seq_mask, nbr_idx, nbr_mask, params):
    K = nbr_idx.shape[0]
    N = params['S'].shape[0]
    params = normalise_ctbn_params (params)
    E_i = params['h'][xs] + jnp.einsum('ij,ij->i',params['J'][xs[:,None],xs[nbr_idx]],nbr_mask)  # (K,)
    return jnp.sum (E_i * seq_mask)

# Variational log-marginal for a continuous-time Bayesian network
def ctbn_variational_log_marg (xs, seq_mask, nbr_idx, nbr_mask, params, log_Z=None):
    if log_Z is None:
        log_Z = ctbn_variational_log_Z (seq_mask, nbr_idx, nbr_mask, params)
    log_p = ctbn_log_marg_unnorm (xs, seq_mask, nbr_idx, nbr_mask, params)
    return log_p - log_Z

# Exact log-partition function for a continuous-time Bayesian network
def ctbn_exact_log_Z (seq_mask, nbr_idx, nbr_mask, params, T):
    K = nbr_idx.shape[0]
    N = params['S'].shape[0]
    params = normalise_ctbn_params (params)
    valid_Xs = [xs for xs in np.ndindex(tuple([N]*K)) if np.all(seq_mask * xs == xs)]
    Es = [ctbn_log_marg_unnorm(X,seq_mask,nbr_idx,nbr_mask,params) for X in valid_Xs]
    return logsumexp(Es)

# Exact log-marginal for a continuous-time Bayesian network
def ctbn_exact_log_marg (xs, seq_mask, nbr_idx, nbr_mask, params):
    if log_Z is None:
        log_Z = ctbn_exact_log_Z (seq_mask, nbr_idx, nbr_mask, params)
    log_p = ctbn_log_marg_unnorm (xs, seq_mask, nbr_idx, nbr_mask, params)
    return log_p - log_Z
