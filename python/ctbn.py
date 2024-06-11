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

def product(x,axis=None):
    return jnp.exp(jnp.sum(jnp.log(x),axis=axis))

# Mean-field averaged rates for a continuous-time Bayesian network
# mu = (K,N) matrix of mean-field probabilities
# Returns (N,N) mean-field averaged rate matrix for component #i
def q_bar (i, C, S, J, h, mu):
    Ci = C[i,:]  # (k,)
    exp_minus_2J = jnp.exp (-2 * J)  # (y_i,x_k)
    exp_minus_2JC = exp_minus_2J[None,:,:] ** Ci[:,None,None]  # (k,y_i,x_k)
    mu_exp_JC = jnp.einsum ('kx,kyx->ky', mu, exp_minus_2JC)  # (k,y_i)
    return S * jnp.exp(-h)[None,:] * product(mu_exp_JC,axis=-2)[None,:]  # (x_i,y_i)

# Returns (K,N,N,N) matrix where entry (j,x_j,x_i,y_i) is the mean-field averaged rate x_i->y_i, conditioned on component #j being in state x_j
def q_bar_cond (i, C, S, J, h, mu):
    K = mu.shape[0]
    cond_mask = jnp.ones((K,K)) - jnp.eye(K)  # (j,k)
    cond_energy = C[i,:,None,None] * J[None,:,:]  # (j,x_j,y_i)
    Ci_mask = jnp.einsum ('jk,k->jk', cond_mask, C[i,:])  # (j,k)
    exp_minus_2J = jnp.exp (-2 * J)  # (N,N)    
    exp_minus_2JC = exp_minus_2J[None,None,:,:] ** Ci_mask[:,:,None,None]  # (j,k,y_i,x_k)
    mu_exp_JC = jnp.einsum ('kx,jkyx->jky', mu, exp_minus_2JC)  # (j,k,y_i)
    return S[None,None,:,:] * jnp.exp(-h)[None,None,None,:] * jnp.exp(-2*cond_energy)[:,:,None,:] * product(mu_exp_JC,axis=-2)[:,:,None,:]  # (j,x_j,x_i,y_i)

# Returns (K,N,N) matrix where entry (j,x_j,x_i) is the mean-field averaged negative exit rate from x_i, conditioned on component #j being in state x_j
def q_bar_cond_diag (i, C, S, J, h, mu):
    K = mu.shape[0]
    cond_mask = jnp.ones((K,K)) - jnp.eye(K)  # (j,k)
    cond_energy = C[i,:,None,None] * J[None,:,:]  # (j,x_j,y_i)
    Ci_mask = jnp.einsum ('jk,k->jk', cond_mask, C[i,:])  # (j,k)
    exp_minus_2J = jnp.exp (-2 * J)  # (N,N)    
    exp_minus_2JC = exp_minus_2J[None,None,:,:] ** Ci_mask[:,:,None,None]  # (j,k,y_i,x_k)
    mu_exp_JC = jnp.einsum ('kx,jkyx->jky', mu, exp_minus_2JC)  # (j,k,y_i)
    return jnp.diag(S)[None,None,:] * jnp.exp(-h)[None,None,:] * jnp.exp(-2*cond_energy) * product(mu_exp_JC,axis=-2)  # (j,x_j,x_i)


# Geometrically-averaged mean-field rates for a continuous-time Bayesian network
# Returns (N,N) matrix
def q_tilde (i, C, S, J, h, mu):
    mean_energy = jnp.einsum ('kx,k,yx->y', mu, C[i,:], J)  # (y_i,)
    return S * jnp.exp(-h-2*mean_energy)[None,:]  # (x_i,y_i)

# Returns (K,N,N,N) matrix where entry (j,x_j,x_i,y_i) is the geometrically-averaged rate x_i->y_i, conditioned on component #j being in state x_j
def q_tilde_cond (i, C, S, J, h, mu):
    K = mu.shape[0]
    cond_mask = jnp.ones((K,K)) - jnp.eye(K)  # (j,k)
    cond_energy = C[i,:,None,None] * J[None,:,:]  # (j,x_j,y_i)
    Ci_mask = jnp.einsum ('jk,k->jk', cond_mask, C[i,:])  # (j,k)
    mean_energy = jnp.einsum ('kx,jk,yx->jy', mu, Ci_mask, J)  # (j,y_i)
    return S[None,None,:,:] * jnp.exp(-h)[None,None,None,:] * jnp.exp(-2*cond_energy)[:,:,None,:] * jnp.exp(-2*mean_energy)[:,None,None,:]  # (j,x_j,x_i,y_i)

# gamma
def gamma (i, mu, rho, C, S, J, h):
    return jnp.einsum ('x,xy,y,x->xy', mu[i,:], q_bar(i,C,S,J,h,mu), rho[i,:], 1/mu[i,:])


# Calculate the variational bound for a continuous-time Bayesian network
def ctbnVariationalBound ():
    pass