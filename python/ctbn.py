import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm
import diffrax

# Implements the algorithm from:
# Cohn et al (2010), JMLR 11:93. Mean Field Variational Approximation for Continuous-Time Bayesian Networks

# Model:
# K components, each with N states
# C = symmetric binary contact matrix (K*K). For i!=j, C_ij=C_ji=1 if i and j are in contact, 0 otherwise. C_ii=0
# S = symmetric exchangeability matrix (N*N). For i!=j, S_ij=S_ji=rate of substitution from i<->j if i and j are equiprobable. S_ii = -sum_j S_ij
# J = symmetric coupling matrix (N*N). For i!=j, J_ij=J_ji=interaction strength between components i and j. J_ii = 0
# h = bias vector (N). h_i=bias of state i

# Endpoint-conditioned variational approximation:
# mu = (K,N) matrix of mean-field probabilities
# rho = (K,N) matrix where entry (i,x_i) is the probability of reaching the final state given that component #i is in state x_i

# Rate for substitution x_i->y_i
# i = 1..K
# x = (K,) vector of integers from 0..N-1
# y_i = integer from 0..N-1
def q_k (i, x, y_i, C, S, J, h):
    return S[x[i],y_i] * jnp.exp (-h[y_i] - jnp.dot (C[i,:], J[y_i,x]))

def product(x,axis=None,keepdims=False):
    return jnp.exp(jnp.sum(safe_log(x),axis=axis,keepdims=keepdims))

def offdiag_mask (N):
    return jnp.ones((N,N)) - jnp.eye(N)

smallest_float32 = jnp.finfo('float32').smallest_normal

def safe_log (x):
    return jnp.log (jnp.where (x == 0, smallest_float32, x))

# Mean-field averaged rates for a continuous-time Bayesian network
# Returns (K,N,N) matrix where entry (i,x_i,y_i) is mean-field averaged rate matrix for component #i
def q_bar (C, S, J, h, mu):
    K, N = mu.shape
    exp_minus_2J = jnp.exp (-2 * J)  # (y_i,x_k)
    exp_minus_2JC = exp_minus_2J[None,None,:,:] ** C[:,:,None,None]  # (i,k,y_i,x_k)
    mu_exp_JC = jnp.einsum ('kx,ikyx->iky', mu, exp_minus_2JC)  # (i,k,y_i)
    S = S * offdiag_mask(N)
    return S[None,:,:] * jnp.exp(-h)[None,None,:] * product(mu_exp_JC,axis=-2,keepdims=True)  # (i,x_i,y_i)

# Returns (K,K,N,N,N) tensor where entry (i,j,x_j,x_i,y_i) is the mean-field averaged rate x_i->y_i, conditioned on component #j being in state x_j
# NB only valid for x_i != y_i
def q_bar_cond (C, S, J, h, mu):
    K, N = mu.shape
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
def q_tilde (C, S, J, h, mu):
    K, N = mu.shape
    mean_energy = jnp.einsum ('kx,ik,yx->iy', mu, C, J)  # (i,y_i)
    S = S * offdiag_mask(N)
    return S[None,:,:] * jnp.exp(-h-2*mean_energy)[None,None,:]  # (i,x_i,y_i)

# Returns (K,K,N,N,N) matrix where entry (i,j,x_j,x_i,y_i) is the geometrically-averaged rate x_i->y_i, conditioned on component #j being in state x_j
# NB only valid for x_i != y_i
def q_tilde_cond (C, S, J, h, mu):
    K, N = mu.shape
    nonself_mask = offdiag_mask(K)  # (j,k)
    cond_energy = C[:,:,None,None] * J[None,None,:,:]  # (i,j,x_j,y_i)
    Ci_mask = jnp.einsum ('jk,ik->ijk', nonself_mask, C)  # (i,j,k)
    mean_energy = jnp.einsum ('kx,ijk,yx->ijy', mu, Ci_mask, J)  # (i,j,y_i)
    S = S * offdiag_mask(N)
    return S[None,None,None,:,:] * jnp.exp(-h)[None,None,None,None,:] * jnp.exp(-2*cond_energy)[:,:,:,None,:] * jnp.exp(-2*mean_energy)[:,:,None,None,:]  # (i,j,x_j,x_i,y_i)

# Rate matrix for a single component, q_{xy} = S_{xy}
def q_single (S, h):
    N = S.shape[0]
    return S * offdiag_mask(N) * jnp.exp(-h)[None,:]

# gamma
# Returns (K,N,N) matrix where entry (i,x_i,y_i) is the joint probability of transition x_i->y_i for component #i
def gamma (mu, rho, C, S, J, h):
    return jnp.einsum ('ix,ixy,iy,iy->ixy', mu, q_tilde(C,S,J,h,mu), rho, 1/mu)

# Returns (K,N) matrix
def psi (mu, rho, C, S, J, h):
    _gamma = gamma(mu,rho,C,S,J,h)  # (K,N,N)
    qbar_cond = q_bar_cond(C,S,J,h,mu)  # (K,K,N,N,N)
    qtilde_cond = q_tilde_cond(C,S,J,h,mu)  # (K,K,N,N,N)
    log_qtilde_cond = -safe_log (jnp.where (qtilde_cond < 0, 1, qtilde_cond))  # (K,K,N,N,N)
    return jnp.einsum('jy,jixyz->ix',mu,qbar_cond) + jnp.einsum('jxy,jixyz->ix',_gamma,log_qtilde_cond)

def rho_deriv (i, mu, rho, C, S, J, h):
    K = mu.shape[0]
    qbar = q_bar(C,S,J,h,mu)  # (K,N,N)
    qbar_diag = jnp.einsum ('kxy->kx', qbar)  # (K,N)
    _psi = psi(mu,rho,C,S,J,h)  # (K,N)
    qtilde = q_tilde(C,S,J,h,mu)  # (K,N,N)
    rho_deriv_i = -rho[i,:] * (qbar_diag[i,:] + _psi[i,:]) - jnp.einsum ('y,xy->x', rho[i,:], qtilde[i,:,:])  # (N,)
    return rho_deriv_i

def mu_deriv (i, mu, rho, C, S, J, h):
    K = mu.shape[0]
    _gamma = gamma(mu,rho,C,S,J,h)  # (K,N,N)
    mu_deriv_i = jnp.einsum('yx->x',_gamma[i,:]) - jnp.einsum('xy->y',_gamma[i,:])  # (N,)
    return mu_deriv_i

def F_deriv (mu, rho, C, S, J, h):
    N = mu.shape[-1]
    qbar = q_bar(C,S,J,h,mu)  # (K,N,N)
    qtilde = q_tilde(C,S,J,h,mu)  # (K,N,N)
    _gamma = gamma (mu, rho, C, S, J, h)  # (K,N,N)
    mask = offdiag_mask(N)[None,:,:]  # (1,N,N)
    log_qtilde = safe_log(jnp.where(mask,qtilde,1))
    gamma_coeff = log_qtilde + 1 + safe_log(mu)[:,:,None] - safe_log(_gamma)
    dF = -jnp.einsum('ix,ixy,ixy->',mu,qbar,mask) + jnp.einsum('ixy,ixy,ixy->',_gamma,gamma_coeff,mask)
    return dF

# Exact posterior for a single component with no interactions
class SingleComponentPosteriorRho:
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

class SingleComponentPosteriorMu (SingleComponentPosteriorRho):
    def evaluate (self, t):
        rho = super().evaluate (t)
        exp_qt = expm(self.q*t) [self.x, :]  # (N,)
        mu = exp_qt * rho / self.exp_qT
        return mu

# Calculate the variational bound for a continuous-time Bayesian network
def ctbnVariationalBound (xs, ys, T, C, S, J, h):
    # initialize component-indexed arrays of mu, rho solutions, using single-component posteriors
    # initialize F_current as variational bound for initial mu and rho, and F_prev as -Infinity
    # create an array of boundary conditions for rho(T), with component #0 having unconstrained rho(T)
    # while (F_current - F_prev)/F_prev > minimum relative increase:
    #  loop over component indices i:
    #   solve rho and then mu for component i, using diffrax, and replace single-component posteriors with diffrax Solution's
    #   F_prev <- F_current, F_current <- new variational bound
    # return mu^0_y(T), i.e. the variational probability that component #0 is in state y at time T
    pass
