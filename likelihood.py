import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm

# Compute substitution log-likelihood of a multiple sequence alignment and phylogenetic tree by pruning
# Parameters:
#  - alignment: (N,L) integer tokens. L is the length of the alignment (#cols), N is the number of sequences (#rows). A token of -1 indicates a gap.
#  - distanceToParent: (N,) floats, distance to parent node
#  - parentIndex: (N,) integers, index of parent node. Nodes are sorted in preorder so parentIndex[i] < i for all i. parentIndex[0] = -1
#  - subRate: (A,A) substitution rate matrix
#  - rootFreq: (A,) root frequencies (typically equilibrium frequencies for substitution rate matrix)

def pruneLogLike (alignment, distanceToParent, parentIndex, subRate, rootFreq):
    assert alignment.ndim == 2
    assert subRate.ndim == 2
    N, L = alignment.shape
    A = subRate.shape[0]
    assert distanceToParent.shape == (N,)
    assert parentIndex.shape == (N,)
    assert rootFreq.shape == (A,)
    assert subRate.shape == (A,A)
    assert alignment.dtype == jnp.int32
    assert parentIndex.dtype == jnp.int32
    assert jnp.all(alignment >= -1)
    assert jnp.all(alignment < A)
    assert jnp.all(distanceToParent >= 0)
    assert jnp.all(parentIndex[1:] >= -1)
    assert jnp.all(parentIndex < jnp.arange(N))
    # Ensure that the substitution rate matrix is normalized
    subRate = jnp.maximum(subRate, 0)
    subRate = subRate - jnp.diag(jnp.sum(subRate, axis=1))
    rootFreq = jnp.maximum(rootFreq, 0)
    rootFreq = rootFreq / jnp.sum(rootFreq)
    # Compute transition matrices per branch
    subMatrix = expm (jnp.einsum('ij,n->nij', subRate, distanceToParent))
    # Initialize pruning matrix
    tokenLookup = jnp.concatenate([jnp.ones(A)[None,:],jnp.eye(A)])
    likelihood = tokenLookup[alignment + 1]  # (N,L,A)
    logNorm = jnp.zeros(L)
    # Compute log-likelihood for all columns in parallel by iterating over nodes in postorder
    for child in range(A-1,0,-1):
        parent = parentIndex[child]
        likelihood = likelihood.at[parent,:,:].multiply (jnp.einsum('ij,lj->li', subMatrix[child], likelihood[child,:,:]))
        maxLike = jnp.max(likelihood[parent,:,:], axis=-1)  # (L,)
        likelihood = likelihood.at[parent,:,:].divide (maxLike[:,None])
        logNorm = logNorm + jnp.log(maxLike)
    logNorm = logNorm + jnp.log(jnp.einsum('li,i->l', likelihood.at[0,:,:], rootFreq))  # (L,)
    return logNorm
