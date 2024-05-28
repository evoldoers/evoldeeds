import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm

import h20
import km03

def alignmentIsValid (alignment, alphabetSize):
    assert jnp.all(alignment >= -1)
    assert jnp.all(alignment < alphabetSize)

def treeIsValid (distanceToParent, parentIndex, alignmentRows):
    assert jnp.all(distanceToParent >= 0)
    assert jnp.all(parentIndex[1:] >= -1)
    assert jnp.all(parentIndex <= jnp.arange(alignmentRows))

# Compute substitution log-likelihood of a multiple sequence alignment and phylogenetic tree by pruning
# Parameters:
#  - alignment: (R,C) integer tokens. C is the length of the alignment (#cols), R is the number of sequences (#rows). A token of -1 indicates a gap.
#  - distanceToParent: (R,) floats, distance to parent node
#  - parentIndex: (R,) integers, index of parent node. Nodes are sorted in preorder so parentIndex[i] <= i for all i. parentIndex[0] = -1
#  - subRate: (*H,A,A) substitution rate matrix/matrices. Leading H axes (if any) are "hidden" substitution rate categories, A is alphabet size
#  - rootProb: (*H,A) root frequencies (typically equilibrium frequencies for substitution rate matrix)
# To pad rows, set parentIndex[paddingRow:] = arange(paddingRow,R)
# To pad columns, set alignment[paddingRow,paddingCol:] = -1

def subLogLike (alignment, distanceToParent, parentIndex, subRate, rootProb):
    subMatrix = computeSubMatrixForTimes (distanceToParent, subRate)
    return subLogLikeForMatrices (alignment, parentIndex, subMatrix, rootProb)

def transLogLike (transCounts, distanceToParent, indelParams, alphabet):
    transMats = computeTransMatForTimes (distanceToParent, indelParams, alphabet)
    return transLogLikeForTransMats (transCounts, transMats)

def computeSubMatrixForTimes (distanceToParent, subRate):
    assert distanceToParent.ndim == 1
    assert subRate.ndim >= 2
    R, = distanceToParent.shape
    *H, A = subRate.shape[0:-1]
    assert subRate.shape == (*H,A,A)
    # Compute transition matrices per branch
    subMatrix = expm (jnp.einsum('...ij,r->...rij', subRate, distanceToParent))  # (*H,R,A,A)
    return subMatrix

defaultDiscretizationParams = (1e-3, 10, 400)  # tMin, tMax, nSteps
def getDiscretizedTimes (discretizationParams=defaultDiscretizationParams):
    return jnp.concat ([jnp.array([0]), jnp.geomspace (*discretizationParams)])

def computeSubMatrixForDiscretizedTimes (subRate, discretizationParams=defaultDiscretizationParams):
    t = getDiscretizedTimes (discretizationParams)
    discreteTimeSubMatrix = computeSubMatrixForTimes (t, subRate)  # (*H,T,A,A)
    return discreteTimeSubMatrix

def discretizeBranchLength (t, discretizationParams=defaultDiscretizationParams):
    tMin, tMax, nSteps = discretizationParams
    return jnp.where (t == 0,
                      0,
                      jnp.digitize (jnp.clip (t, tMin, tMax), jnp.geomspace (tMin, tMax, nSteps)))

def getSubMatrixForDiscretizedBranchLengths (discretizedTimes, discreteTimeSubMatrix):
    subMatrix = discreteTimeSubMatrix[...,discretizedTimes,:,:]  # (*H,R,A,A)
    return subMatrix

def logTransMat (transMat):
    return jnp.log (jnp.maximum (transMat, h20.smallest_float32))

def logRootTransMat():
    return logTransMat (h20.dummyRootTransitionMatrix())

def computeTransMatForDiscretizedTimes (indelParams, alphabet, discretizationParams=defaultDiscretizationParams, useKM03=False):
    td = getDiscretizedTimes (discretizationParams)
    if useKM03:
        transMat = h20.transitionMatrixForTimes(td,indelParams,alphabetSize=len(alphabet),transitionMatrix=km03.transitionMatrix)
    else:
        transMat = h20.transitionMatrixForMonotonicTimes(td,indelParams,alphabetSize=len(alphabet))
    return logTransMat (transMat)

def getTransMatForDiscretizedTimes (discretizedTimes, discreteTimeTransMat):
    assert discreteTimeTransMat.ndim >= 3
    assert len(discretizedTimes) > 1
    branches = discreteTimeTransMat[...,discretizedTimes[1:],:,:]  # (...categories...,T-1,A,A)
    root = logRootTransMat() * jnp.ones_like(branches[...,0:1,:,:])  # (...categories...,1,A,A)
    return jnp.concatenate ([root, branches], axis=-3)

def computeTransMatForTimes (ts, indelParams, alphabet):
    branches = jnp.stack ([h20.transitionMatrix(t,indelParams,alphabetSize=len(alphabet)) for t in ts[1:]], axis=0)
    return jnp.concatenate ([logRootTransMat()[None,:,:], logTransMat(branches)], axis=0)

def transLogLikeForTransMats (transCounts, transMats):
    assert transCounts.shape == transMats.shape, "transCounts.shape = %s, transMats.shape = %s" % (transCounts.shape, transMats.shape)  # (...categories...,T,A,A)
#    jax.debug.print('transCounts={}, transMats={}', transCounts, transMats)
    trans_ll = transCounts * transMats
    trans_ll = jnp.sum (trans_ll, axis=(-1,-2))
    return trans_ll  # (...categories...,rows)

def subLogLikeForMatrices (alignment, parentIndex, subMatrix, rootProb, maxChunkSize = 128):
    assert alignment.ndim == 2
    assert subMatrix.ndim >= 3
    *H, R, A = subMatrix.shape[0:-1]
    C = alignment.shape[-1]
    assert parentIndex.shape == (R,)
    assert rootProb.shape == (*H,A)
    assert subMatrix.shape == (*H,R,A,A)
    assert alignment.dtype == jnp.int32
    assert parentIndex.dtype == jnp.int32
    # If too big, split into chunks
    if C > maxChunkSize:
#        jax.debug.print('Splitting %d x %d alignment into %d chunks of size %d x %d' % (R,C,C//maxChunkSize,R,maxChunkSize))
        return jnp.concatenate ([subLogLikeForMatrices (alignment[:,i:i+maxChunkSize], parentIndex, subMatrix, rootProb) for i in range(0,C,maxChunkSize)], axis=-1)
    # Initialize pruning matrix
    tokenLookup = jnp.concatenate([jnp.ones(A)[None,:],jnp.eye(A)])
    likelihood = tokenLookup[alignment + 1]  # (R,C,A)
    if len(H) > 0:
        likelihood = jnp.expand_dims (likelihood, jnp.arange(len(H)))  # (*ones_like(H),R,C,A)
        likelihood = jnp.repeat (likelihood, repeats=jnp.array(H), axis=jnp.arange(len(H)))  # (*H,R,C,A)
    logNorm = jnp.zeros(*H,C)  # (*H,C)
    # Compute log-likelihood for all columns in parallel by iterating over nodes in postorder
    for child in range(R-1,0,-1):
        parent = parentIndex[child]
        likelihood = likelihood.at[...,parent,:,:].multiply (jnp.einsum('...ij,...cj->...ci', subMatrix[...,child,:,:], likelihood[...,child,:,:]))
        maxLike = jnp.max(likelihood[...,parent,:,:], axis=-1)  # (*H,C)
        likelihood = likelihood.at[...,parent,:,:].divide (maxLike[...,None])  # guard against underflow
        logNorm = logNorm + jnp.log(maxLike)
    logNorm = logNorm + jnp.log(jnp.einsum('...ci,...i->...c', likelihood[...,0,:,:], rootProb))  # (*H,C)
    return logNorm

def padDimension (len, multiplier):
    return jnp.where (multiplier == 2,  # handle this case specially to avoid precision errors
                      1 << (len-1).bit_length(),
                      int (jnp.ceil (multiplier ** jnp.ceil (jnp.log(len) / jnp.log(multiplier)))))

def padAlignment (alignment, parentIndex, distanceToParent, transCounts, nRows: int = None, nCols: int = None, colMultiplier = 2, rowMultiplier = 2):
    unpaddedRows, unpaddedCols = alignment.shape
    if nCols is None:
        nCols = padDimension (unpaddedCols, colMultiplier)
    if nRows is None:
        nRows = padDimension (unpaddedRows, rowMultiplier)
    # To pad rows, set parentIndex[paddingRow:] = arange(paddingRow,R) and pad alignment and distanceToParent with any value (-1 and 0 for predictability),
    # and pad transCounts with zeros
    if nRows > unpaddedRows:
        alignment = jnp.concatenate ([alignment, -1*jnp.ones((nRows - unpaddedRows, unpaddedCols), dtype=alignment.dtype)], axis=0)
        distanceToParent = jnp.concatenate ([distanceToParent, jnp.zeros(nRows - unpaddedRows, dtype=distanceToParent.dtype)], axis=0)
        parentIndex = jnp.concatenate ([parentIndex, jnp.arange(unpaddedRows,nRows, dtype=parentIndex.dtype)], axis=0)
        transCounts = jnp.concatenate ([transCounts, jnp.zeros((nRows - unpaddedRows, *transCounts.shape[1:]), dtype=transCounts.dtype)], axis=0)
    # To pad columns, set alignment[paddingRow,paddingCol:] = -1
    if nCols > unpaddedCols:
        alignment = jnp.concatenate ([alignment, -1*jnp.ones((nRows, nCols - unpaddedCols), dtype=alignment.dtype)], axis=1)
    return alignment, parentIndex, distanceToParent, transCounts

def normalizeSubRate (subRate):
    subRate = jnp.abs (subRate)
    return subRate - jnp.diag(jnp.sum(subRate, axis=-1))

def zeroDiagonal (matrix):
    return matrix - jnp.diag(jnp.diag(matrix))

def logitsToProbs (logits):
    return jax.nn.softmax(logits)

def probsToLogits (probs):
    return jnp.log (probs)

def logitToProb (logit):
    return 1 / (1 + jnp.exp (logit))

def probToLogit (prob):
    return jnp.log (1/prob - 1)

def normalizeRootProb (rootProb):
    return rootProb / jnp.sum(rootProb)

def normalizeSubModel (subRate, rootProb):
    return normalizeSubRate(subRate), normalizeRootProb(rootProb)

def parametricSubModel (subRate, rootLogits):
    return normalizeSubRate(subRate), logitsToProbs(rootLogits)

def exchangeabilityMatrixToSubMatrix (exchangeRate, rootProb):
    sqrtRootProb = jnp.sqrt(rootProb)
    return jnp.einsum('i,...ij,j->...ij', 1/sqrtRootProb, exchangeRate, sqrtRootProb)

def subMatrixToExchangeabilityMatrix (subMatrix, rootProb):
    sqrtRootProb = jnp.sqrt(rootProb)
    return jnp.einsum('i,...ij,j->...ij', sqrtRootProb, subMatrix, 1/sqrtRootProb)

def symmetrizeSubRate (matrix):
    return normalizeSubRate (0.5 * (matrix + matrix.swapaxes(-1,-2)))

def parametricReversibleSubModel (subRate, rootLogits):
    rootProb = logitsToProbs (rootLogits)
    subRate = exchangeabilityMatrixToSubMatrix (symmetrizeSubRate(subRate), rootProb)
    return subRate, rootProb

def parametricIndelModel (lam, mu, x_logits, y_logits):
    return jnp.abs(lam), jnp.abs(mu), logitToProb(x_logits), logitToProb(y_logits)

def indelModelToParams (lam, mu, x, y):
    return lam, mu, probToLogit(x), probToLogit(y)

def createGGIModelFactory (subModelFactory):
    def parametricGGIModel (params):
        subRate, rootProb = subModelFactory (params['subrate'], params['root'])
        indelParams = parametricIndelModel (*params['indels'])
        return subRate, rootProb, indelParams
    return parametricGGIModel

def parseHistorianParams (params):
    alphabet = params['alphabet']
    def parseSubRate (p):
        subRate = jnp.array([[p['subrate'].get(i,{}).get(j,0) for j in alphabet] for i in alphabet], dtype=jnp.float32)
        rootProb = jnp.array([p['rootprob'].get(i,0) for i in alphabet], dtype=jnp.float32)
        subRate, rootProb = normalizeSubModel (subRate, rootProb)
        return subRate, rootProb
    if 'mixture' in params or 'coltype' in params:
        mixture = [parseSubRate(p) for p in params.get('mixture',params.get('coltype'))]
    else:
        mixture = [parseSubRate(params)]
    def parseIndelParams (cpt):
        return tuple(cpt.get(name,0) for name in ['insrate','delrate','insextprob','delextprob'])
    alignTypes = params.get ('aligntype', [params])
    indelParams = [parseIndelParams(t) for t in alignTypes]
    alnTypeLogits = jnp.array([t.get('weight',0) for t in alignTypes])   # famTypeLogits[i] \propto log P(alignType=i)
    colTypeLogits = jnp.stack([jnp.array([t.get('coltypeweight',jnp.zeros(len(mixture))) for t in alignTypes])], axis=0)  # colTypeLogits[i,j] \propto log P(columnType=j | alignType=i)
    colShape = jnp.stack([jnp.array([t.get('colshape',jnp.ones(len(mixture))) for t in alignTypes])], axis=0)  # colShape[i,j] = shape param for gamma distribution of colType j in alnType i (scale=shape assumed)
    quantiles = params.get('quantiles',1)  # number of quantiles (rate classes) for column gamma distributions
    return alphabet, mixture, indelParams, alnTypeLogits, colTypeLogits, colShape, quantiles

def toHistorianParams (alphabet, mixture, indelParams, alnTypeLogits, colTypeLogits, colShape, nQuantiles):
    def toSubRate (subRate, rootProb):
        return { 'subrate': {alphabet[i]: {alphabet[j]: float(subRate[i,j]) for j in range(len(alphabet)) if i != j} for i in range(len(alphabet))},
                 'rootprob': {alphabet[i]: float(rootProb[i]) for i in range(len(alphabet))} }
    def toIndelParams (indelParams):
        return {name: float(param) for name,param in zip(['insrate','delrate','insextprob','delextprob'],indelParams)}
    nColTypes = len(mixture)
    nAlignTypes = len(alnTypeLogits)
    assert alnTypeLogits.shape == (nAlignTypes,)
    assert colTypeLogits.shape == (nAlignTypes,nColTypes)
    assert colShape.shape == (nAlignTypes,nColTypes)
    # use backwardly-compatible format if possible, so Historian can read it
    if nAlignTypes == 1 and nQuantiles == 1:
        if nColTypes == 1:
            return {**toSubRate(*mixture[0]), **toIndelParams(indelParams[0])}
        else:
            return {'mixture': [toSubRate(*m) for m in mixture],
                     **toIndelParams(indelParams[0])}
    return {'coltype': [toSubRate(*m) for m in mixture],
            'aligntype': [{'weight': float(alnTypeLogits[i]),
                           'colshape': [float(s) for s in colShape[i,:]],
                           'coltypeweight': [float(w) for w in colTypeLogits[i,:]],
                           **toIndelParams(indelParams[i])} for i in range(nAlignTypes)],
            'quantiles': nQuantiles}
