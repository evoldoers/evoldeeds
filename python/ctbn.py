import jax
import jax.numpy as jnp
import diffrax

# Model:
# K components, each with N states
# C = symmetric binary contact matrix (K*K). For i!=j, C_ij=C_ji=1 if i and j are in contact, 0 otherwise. C_ii=0
# S = symmetric exchangeability matrix (N*N). For i!=j, S_ij=S_ji=rate of substitution from i<->j if i and j are equiprobable. S_ii = -sum_j S_ij
# J = symmetric coupling matrix (N*N). For i!=j, J_ij=J_ji=interaction strength between components i and j. J_ii = 0
# h = bias vector (N). h_i=bias of state i

# Rate for substitution x_i->y_i
# i = 1..K
# x = (K,) vector of integers from 0..N-1
# y_i = integer from 0..N-1
def q_k (i, x, y_i, C, S, J, h):
    return S[x[i],y_i] * jnp.exp (-h[y_i] - jnp.dot (C[i,:], J[y_i,x]))

def product(x,axis=None,keepdims=False):
    return jnp.exp(jnp.sum(jnp.log(x),axis=axis,keepdims=keepdims))

def offdiag_mask (N):
    return jnp.ones((N,N)) - jnp.eye(N)

smallest_float32 = jnp.finfo('float32').smallest_normal

# Mean-field averaged rates for a continuous-time Bayesian network
# mu = (K,N) matrix of mean-field probabilities
# Returns (K,N,N) matrix where entry (i,x_i,y_i) is mean-field averaged rate matrix for component #i
# NB only valid for x_i != y_i
def q_bar (i, C, S, J, h, mu):
    exp_minus_2J = jnp.exp (-2 * J)  # (y_i,x_k)
    exp_minus_2JC = exp_minus_2J[None,None,:,:] ** C[:,:,None,None]  # (i,k,y_i,x_k)
    mu_exp_JC = jnp.einsum ('kx,ikyx->iky', mu, exp_minus_2JC)  # (i,k,y_i)
    return S[None,:,:] * jnp.exp(-h)[None,None,:] * product(mu_exp_JC,axis=-2,keepdims=True)  # (i,x_i,y_i)

# Returns (K,K,N,N,N) tensor where entry (i,j,x_j,x_i,y_i) is the mean-field averaged rate x_i->y_i, conditioned on component #j being in state x_j
# NB only valid for x_i != y_i
def q_bar_cond (C, S, J, h, mu):
    K = mu.shape[0]
    cond_mask = offdiag_mask(K)  # (j,k)
    cond_energy = C[:,:,None,None] * J[None,:,:]  # (i,j,x_j,y_i)
    Ci_mask = jnp.einsum ('jk,ik->ijk', cond_mask, C)  # (i,j,k)
    exp_minus_2J = jnp.exp (-2 * J)  # (N,N)
    exp_minus_2JC = exp_minus_2J[None,None,None,:,:] ** Ci_mask[:,:,None,None,None]  # (i,j,k,y_i,x_k)
    mu_exp_JC = jnp.einsum ('kx,ijkyx->ijky', mu, exp_minus_2JC)  # (i,j,k,y_i)
    return S[None,None,None,:,:] * jnp.exp(-h)[None,None,None,None,:] * jnp.exp(-2*cond_energy)[:,:,:,None,:] * product(mu_exp_JC,axis=-2)[:,:,:,None,:]  # (i,j,x_j,x_i,y_i)

# Geometrically-averaged mean-field rates for a continuous-time Bayesian network
# Returns (K,N,N) matrix where entry (i,x_i,y_i) is geometrically-averaged mean-field rate matrix for component #i
# NB only valid for x_i != y_i
def q_tilde (C, S, J, h, mu):
    mean_energy = jnp.einsum ('kx,ik,yx->iy', mu, C, J)  # (i,y_i)
    return S[None,:,:] * jnp.exp(-h-2*mean_energy)[None,None,:]  # (i,x_i,y_i)

# Returns (K,K,N,N,N) matrix where entry (i,j,x_j,x_i,y_i) is the geometrically-averaged rate x_i->y_i, conditioned on component #j being in state x_j
# NB only valid for x_i != y_i
def q_tilde_cond (C, S, J, h, mu):
    K = mu.shape[0]
    cond_mask = offdiag_mask(K)  # (j,k)
    cond_energy = C[:,:,None,None] * J[None,None,:,:]  # (i,j,x_j,y_i)
    Ci_mask = jnp.einsum ('jk,ik->ijk', cond_mask, C)  # (i,j,k)
    mean_energy = jnp.einsum ('kx,ijk,yx->ijy', mu, Ci_mask, J)  # (i,j,y_i)
    return S[None,None,None,:,:] * jnp.exp(-h)[None,None,None,None,:] * jnp.exp(-2*cond_energy)[:,:,:,None,:] * jnp.exp(-2*mean_energy)[:,:,None,None,:]  # (i,j,x_j,x_i,y_i)

# gamma
# Returns (K,N,N) matrix where entry (i,x_i,y_i) is the joint probability of transition x_i->y_i for component #i
def gamma (mu, rho, C, S, J, h):
    return jnp.einsum ('ix,ixy,iy,iy->ixy', mu, q_bar(C,S,J,h,mu), rho, 1/mu)

# Returns (K,N) matrix
def psi (mu, rho, C, S, J, h):
    _gamma = gamma(mu,rho,C,S,J,h)  # (K,N,N)
    _q_bar_cond = q_bar_cond(C,S,J,h,mu)  # (K,K,N,N,N)
    _q_tilde_cond = q_tilde_cond(C,S,J,h,mu)  # (K,K,N,N,N)
    log_q_tilde_cond = -jnp.log (jnp.where (_q_tilde_cond < 0, 1, _q_tilde_cond))  # (K,K,N,N,N)
    return jnp.einsum('jy,jixyz->ix',mu,_q_bar_cond) + jnp.einsum('jxy,jixyz->ix',_gamma,log_q_tilde_cond)


# Calculate the variational bound for a continuous-time Bayesian network
def ctbnVariationalBound ():
    pass