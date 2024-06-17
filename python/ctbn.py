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

def rowNormalise (matrix):
    matrix = jnp.abs (matrix)
    return matrix - jnp.diag(jnp.sum(matrix, axis=-1))

# Implements the algorithm from Cohn et al (2010), for protein Potts models parameterized by contact & coupling matrices
# Cohn et al (2010), JMLR 11:93. Mean Field Variational Approximation for Continuous-Time Bayesian Networks

# Model:
# K components, each with N states
# C = symmetric binary contact matrix (K*K). For i!=j, C_ij=C_ji=1 if i and j are in contact, 0 otherwise. C_ii=0
# S = symmetric exchangeability matrix (N*N). For i!=j, S_ij=S_ji=rate of substitution from i<->j if i and j are equiprobable. S_ii = -sum_j S_ij
# J = symmetric coupling matrix (N*N). For i!=j, J_ij=J_ji=interaction strength between components i and j. J_ii = 0
# h = bias vector (N). h_i=bias of state i

def normalise_ctbn_params (params):
    return { 'S' : symmetrise(rowNormalise(params['S'])),
             'J' : symmetrise(params['J']),
             'h' : params['h'] }

# Endpoint-conditioned variational approximation:
# mu = (K,N) matrix of mean-field probabilities
# rho = (K,N) matrix where entry (i,x_i) is the probability of reaching the final state given that component #i is in state x_i

# Rate for substitution x_i->y_i
# i = 1..K
# x = (K,) vector of integers from 0..N-1
# y_i = integer from 0..N-1
def q_k (i, x, y_i, C, params):
    S = params['S']
    J = params['J']
    h = params['h']
    return S[x[i],y_i] * jnp.exp (-h[y_i] - 2*jnp.dot (C[i,:], J[y_i,x]))

# Mean-field averaged rates for a continuous-time Bayesian network
# Returns (K,N,N) matrix where entry (i,x_i,y_i) is mean-field averaged rate matrix for component #i
def q_bar (C, params, mu):
    K, N = mu.shape
    S = params['S']
    J = params['J']
    h = params['h']
    exp_minus_2J = jnp.exp (-2 * J)  # (y_i,x_k)
    exp_minus_2JC = exp_minus_2J[None,None,:,:] ** C[:,:,None,None]  # (i,k,y_i,x_k)
    mu_exp_JC = jnp.einsum ('kx,ikyx->iky', mu, exp_minus_2JC)  # (i,k,y_i)
    S = S * offdiag_mask(N)
    return S[None,:,:] * jnp.exp(-h)[None,None,:] * product(mu_exp_JC,axis=-2,keepdims=True)  # (i,x_i,y_i)

# Returns (K,K,N,N,N) tensor where entry (i,j,x_j,x_i,y_i) is the mean-field averaged rate x_i->y_i, conditioned on component #j being in state x_j
# NB only valid for x_i != y_i
def q_bar_cond (C, params, mu):
    K, N = mu.shape
    S = params['S']
    J = params['J']
    h = params['h']
    nonself_mask = offdiag_mask(K)  # (x,y)
    cond_energy = C[:,:,None,None] * J[None,:,:]  # (i,j,x_j,y_i)
    Ci_mask = jnp.einsum ('jk,ik->ijk', nonself_mask, C)  # (i,j,k)
    exp_minus_2J = jnp.exp (-2 * J)  # (N,N)
    exp_minus_2JC = exp_minus_2J[None,None,None,:,:] ** Ci_mask[:,:,None,None,None]  # (i,j,k,y_i,x_k)
    mu_exp_JC = jnp.einsum ('kx,ijkyx->ijky', mu, exp_minus_2JC)  # (i,j,k,y_i)
    S = S * offdiag_mask(N)
    return S[None,None,None,:,:] * jnp.exp(-h)[None,None,None,None,:] * jnp.exp(-2*cond_energy)[:,:,:,None,:] * product(mu_exp_JC,axis=-2)[:,:,:,None,:]  # (i,j,x_j,x_i,y_i)

# Geometrically-averaged mean-field rates for a continuous-time Bayesian network
# Returns (K,N,N) matrix where entry (i,x_i,y_i) is geometrically-averaged mean-field rate matrix for component #i
# NB only valid for x_i != y_i
def q_tilde (C, params, mu):
    K, N = mu.shape
    S = params['S']
    J = params['J']
    h = params['h']
    mean_energy = jnp.einsum ('kx,ik,yx->iy', mu, C, J)  # (i,y_i)
    S = S * offdiag_mask(N)
    return S[None,:,:] * jnp.exp(-h-2*mean_energy)[None,None,:]  # (i,x_i,y_i)

# Returns (K,K,N,N,N) matrix where entry (i,j,x_j,x_i,y_i) is the geometrically-averaged rate x_i->y_i, conditioned on component #j being in state x_j
# NB only valid for x_i != y_i
def q_tilde_cond (C, params, mu):
    K, N = mu.shape
    S = params['S']
    J = params['J']
    h = params['h']
    nonself_mask = offdiag_mask(K)  # (j,k)
    cond_energy = C[:,:,None,None] * J[None,None,:,:]  # (i,j,x_j,y_i)
    Ci_mask = jnp.einsum ('jk,ik->ijk', nonself_mask, C)  # (i,j,k)
    mean_energy = jnp.einsum ('kx,ijk,yx->ijy', mu, Ci_mask, J)  # (i,j,y_i)
    S = S * offdiag_mask(N)
    return S[None,None,None,:,:] * jnp.exp(-h)[None,None,None,None,:] * jnp.exp(-2*cond_energy)[:,:,:,None,:] * jnp.exp(-2*mean_energy)[:,:,None,None,:]  # (i,j,x_j,x_i,y_i)

# Rate matrix for a single component, q_{xy} = S_{xy}
# S: (N,N)
# h: (N,)
def q_single (params):
    S = params['S']
    h = params['h']
    N = S.shape[0]
    return S * offdiag_mask(N) * jnp.exp(-h)[None,:]

# Amalgamated (joint) rate matrix for all components
# Note this is big: (N^K,N^K)
def q_joint (C, params):
    N = params['S'].shape[0]
    K = C.shape[0]
    def get_components (x):
        return jnp.array ([(x//(N**j))%N for j in range(K)])
    def get_rate (xs, ys):
        diffs = jnp.where (xs == ys, 0, 1)
        i = jnp.argmax (diffs)
        return jnp.where (jnp.sum(diffs) == 1, q_k(i, xs, ys[i], C, params), 0)
    states = jax.vmap (get_components)(jnp.arange(N**K))
    Q = jax.vmap (lambda x: jax.vmap (lambda y: get_rate(x,y))(states))(states)
    return rowNormalise(Q)

# gamma
# Returns (K,N,N) matrix where entry (i,x_i,y_i) is the joint probability of transition x_i->y_i for component #i
def gamma (mu, rho, C, params):
    return jnp.einsum ('ix,ixy,iy,iy->ixy', mu, q_tilde(C,params,mu), rho, 1/mu)

# Returns (K,N) matrix
def psi (mu, rho, C, params):
    _gamma = gamma(mu,rho,C,params)  # (K,N,N)
    qbar_cond = q_bar_cond(C,params,mu)  # (K,K,N,N,N)
    qtilde_cond = q_tilde_cond(C,params,mu)  # (K,K,N,N,N)
    log_qtilde_cond = -safe_log (jnp.where (qtilde_cond < 0, 1, qtilde_cond))  # (K,K,N,N,N)
    return jnp.einsum('jy,jixyz->ix',mu,qbar_cond) + jnp.einsum('jxy,jixyz->ix',_gamma,log_qtilde_cond)

def rho_deriv (i, mu, rho, C, params):
    K = mu.shape[0]
    qbar = q_bar(C,params,mu)  # (K,N,N)
    qbar_diag = jnp.einsum ('kxy->kx', qbar)  # (K,N)
    _psi = psi(mu,rho,C,params)  # (K,N)
    qtilde = q_tilde(C,params,mu)  # (K,N,N)
    rho_deriv_i = -rho[i,:] * (qbar_diag[i,:] + _psi[i,:]) - jnp.einsum ('y,xy->x', rho[i,:], qtilde[i,:,:])  # (N,)
    return rho_deriv_i

def mu_deriv (i, mu, rho, C, params):
    K = mu.shape[0]
    _gamma = gamma(mu,rho,C,params)  # (K,N,N)
    mu_deriv_i = jnp.einsum('yx->x',_gamma[i,:]) - jnp.einsum('xy->y',_gamma[i,:])  # (N,)
    return mu_deriv_i

def F_deriv (mu, rho, C, params):
    N = mu.shape[-1]
    qbar = q_bar(C,params,mu)  # (K,N,N)
    qtilde = q_tilde(C,params,mu)  # (K,N,N)
    _gamma = gamma (mu, rho, C, params)  # (K,N,N)
    mask = offdiag_mask(N)[None,:,:]  # (1,N,N)
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
    mu_solns, rho_solns, C, params = args
    mu, rho = eval_mu_rho (mu_solns, rho_solns, t)
    return F_deriv (mu, rho, C, params)

def solve_F (T, mu_solns, rho_solns, C, params, rtol=1e-3, atol=1e-6):
    term = diffrax.ODETerm (F_term)
    solver = diffrax.Dopri5()
    controller = diffrax.PIDController (rtol=rtol, atol=atol)
    return diffrax.diffeqsolve (term, solver, t0=0, t1=T, dt0=None, y0=0,
                                args=(mu_solns, rho_solns, C, params),
                                stepsize_controller = controller)

def make_rho_term (i):
    def rho_term (t, rho_i_t, args):
        mu_solns, rho_solns, C, params = args
        mu, rho = eval_mu_rho (mu_solns, rho_solns, t)
        rho = rho.at(i).set(rho_i_t)
        return rho_deriv (i, mu, rho, C, params)
    return rho_term

def solve_rho (T, i, rho_i_T, mu_solns, rho_solns, C, params, rtol=1e-3, atol=1e-6):
    N = params['S'].shape[0]
    term = diffrax.ODETerm (make_rho_term(i))
    solver = diffrax.Dopri5()
    controller = diffrax.PIDController (rtol=rtol, atol=atol)
    return diffrax.diffeqsolve (term, solver, t0=T, t1=0, dt0=None, y0 = rho_i_T,
                                args=(mu_solns, rho_solns, C, params),
                                stepsize_controller = controller,
                                saveat = diffrax.SaveAt(dense=True))

def make_mu_term (i):
    def mu_term (t, mu_i_t, args):
        mu_solns, rho_solns, C, params = args
        mu, rho = eval_mu_rho (mu_solns, rho_solns, t)
        mu = mu.at(i).set(mu_i_t)
        return mu_deriv (i, mu, rho, C, params)
    return mu_term

def solve_mu (T, i, mu_i_0, mu_solns, rho_solns, C, params, rtol=1e-3, atol=1e-6):
    term = diffrax.ODETerm (make_mu_term(i))
    solver = diffrax.Dopri5()
    controller = diffrax.PIDController (rtol=rtol, atol=atol)
    return diffrax.diffeqsolve (term, solver, t0=0, t1=T, dt0=None, y0=mu_i_0,
                                args=(mu_solns, rho_solns, C, params),
                                stepsize_controller = controller,
                                saveat = diffrax.SaveAt(dense=True))

# Calculate the variational likelihood for a component in a continuous-time Bayesian network, conditioned on its Markov blanket
def ctbn_component_probability (xs, ys, T, C, params, **kwargs):
    ys_neg0 = ys.at(0).set(-1)
    F, rho_solns, mu_solns = ctbn_variational_fit (xs, ys_neg0, T, C, params, **kwargs)
    return mu_solns[0].evaluate(T)[ys[0]]

# Calculate the best variational fit for an endpoint-conditioned continuous-time Bayesian network
def ctbn_variational_fit (xs, ys, T, C, params, min_inc=1e-3, max_updates=3):
    K = C.shape[0]
    N = params['S'].shape[0]
    params = normalise_ctbn_params (params)
    # initialize component-indexed arrays of mu, rho solutions, using single-component posteriors
    q = q_single (params)
    rho_solns = [ExactRho (q, T, xs[i], ys[i]) for i in range(K)]
    mu_solns = [ExactMu (q, T, xs[i], ys[i]) for i in range(K)]
    # initialize F_current as variational bound for initial mu and rho, and F_prev as -Infinity
    F_prev = -jnp.inf
    F_current = solve_F (T, mu_solns, rho_solns, C, params)
    # create arrays of boundary conditions for mu(0) and rho(T), with y==-1 implying unconstrained rho(T)
    ones_or_onehot = jnp.concatenate([jnp.ones(N)[None,:],jnp.eye(N)])
    mu_0 = jnp.eye(N)[xs]
    rho_T = ones_or_onehot[ys + 1]
    # while (F_current - F_prev)/F_prev > minimum relative increase:
    #  loop over component indices i:
    #   solve rho and then mu for component i, using diffrax, and replace single-component posteriors with diffrax Solution's
    #   F_prev <- F_current, F_current <- new variational bound
    def while_cond_fun (args):
        F_prev, (F_current, rho_solns, mu_solns), (F_best, rho_best, mu_best), n_updates = args
        return n_updates < max_updates and F_current > F_prev and (F_current - F_prev) / F_prev > min_inc
    def while_body_fun (args):
        F_prev, (F_current, rho_solns, mu_solns), (F_best, rho_best, mu_best), n_updates = args
        F_prev = F_best
        F_current = solve_F (T, mu_solns, rho_solns, C, params)
        rho_solns, mu_solns = jax.lax.fori_loop (0, N, loop_body_fun, (rho_solns, mu_solns))
        keep_best = lambda: (F_best, rho_best, mu_best)
        new_best = lambda: (F_current, rho_solns, mu_solns)
        F_best, rho_best, mu_best = jax.lax.cond (F_current > F_best, new_best, keep_best)
        return F_prev, (F_current, rho_solns, mu_solns), (F_best, rho_best, mu_best), n_updates + 1
    def loop_body_fun (i, args):
        rho_solns, mu_solns = args
        i = N - 1 - i  # finish with component #0
        new_rho_i = solve_rho (T, i, rho_T[i,:], mu_solns, rho_solns, C, params)
        rho_solns = [new_rho_i if k==i else old_rho_i for k, old_rho_i in enumerate(rho_solns)]
        new_mu_i = solve_mu (T, i, mu_0[i,:], mu_solns, rho_solns, C, params)
        mu_solns = [new_mu_i if k==i else old_mu_i for k, old_mu_i in enumerate(mu_solns)]
        return rho_solns, mu_solns
    init_state = F_current, rho_solns, mu_solns
    F_prev, _final_state, best_state = jax.lax.while_loop (while_cond_fun, while_body_fun, (F_prev, init_state, init_state, 0))
    return best_state

# Given a contact matrix C for a sequence, return a site-indexed list of tuples:
# - the Markov blanket for site i, plus site i itself (i.e. a list of sites j!=i such that C[i,j] != 0)
# - the contact matrix for that list of sites (i.e. i plus its Markov blanket)
def get_Markov_blankets (C):
    N = C.shape[0]
    C_with_self = jnp.abs(C) + jnp.eye(N)  # has nonzero wherever C does, plus the diagonal
    def get_blanket (i):
        blanket = C_with_self[i,:].nonzero()
        return blanket, C[blanket,:][:,blanket]
    return [get_blanket(i) for i in range(N)]

# Given two aligned (ungapped) sequences, a contact matrix,
# and a site mask (which can be used to mask out sites that are added for padding reasons),
# along with the parameters of a Potts continuous-time Bayes network substitution model,
# return the log of the pseudolikelihood for descendant (sequence #2) given ancestor (sequence #1)
def ctbn_log_pseudolikelihood (xseq, yseq, mask, T, C, params):
    pl_terms = [ctbn_component_probability (xseq[blanket], yseq[blanket], T, C_local, params)
                for (blanket, C_local) in get_Markov_blankets(C)]
    log_pl = jnp.sum (jnp.where(mask, jnp.log(pl_terms), 0))
    return log_pl

# Weak L2 regularizer for J and h
def param_regularizer (params, alpha=1e-4):
    return alpha * (jnp.sum (params['J']**2) + jnp.sum (params['h']**2))
