import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm

from tensorflow_probability.substrates.jax.distributions import Gamma

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

def transLogLike (transCounts, distanceToParent, indelParams, alphabetSize = 20, useKM03 = False):
    nRows = distanceToParent.shape[0]
    assert transCounts.shape == (nRows, 3, 3)
    transMats = computeTransMatForTimes (distanceToParent, indelParams, alphabetSize=alphabetSize, useKM03=useKM03)
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

def logTransMat (transMat):
    return jnp.log (jnp.maximum (transMat, h20.smallest_float32))

def logRootTransMat():
    return logTransMat (h20.dummyRootTransitionMatrix())

def computeTransMatForTimes (ts, indelParams, alphabetSize=20, useKM03=False):
    transitionMatrix = km03.transitionMatrix if useKM03 else h20.transitionMatrix
    branches = jnp.stack ([transitionMatrix(t,indelParams,alphabetSize=alphabetSize) for t in ts[1:]], axis=0)
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
        likelihood += jnp.zeros((*H,R,C,A))
    logNorm = jnp.zeros((*H,C))  # (*H,C)
    # Compute log-likelihood for all columns in parallel by iterating over nodes in postorder
    postorderBranches = (jnp.arange(R-1,0,-1),  # child indices
                         jnp.flip(parentIndex[1:]), # parent indices
                         jnp.flip(jnp.moveaxis(subMatrix,-3,0)[1:,...],axis=0))  # substitution matrices
    (likelihood, logNorm), _dummy = jax.lax.scan (computeLogLikeForBranch, (likelihood, logNorm), postorderBranches)
    logNorm = logNorm + jnp.log(jnp.einsum('...ci,...i->...c', likelihood[...,0,:,:], rootProb))  # (*H,C)
    return logNorm

def computeLogLikeForBranch (vars, branch):
    likelihood, logNorm = vars
    child, parent, subMatrix = branch
    likelihood = likelihood.at[...,parent,:,:].multiply (jnp.einsum('...ij,...cj->...ci', subMatrix, likelihood[...,child,:,:]))
    maxLike = jnp.max(likelihood[...,parent,:,:], axis=-1)  # (*H,C)
    likelihood = likelihood.at[...,parent,:,:].divide (maxLike[...,None])  # guard against underflow
    logNorm = logNorm + jnp.log(maxLike)
    return (likelihood, logNorm), None

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
    return jax.nn.softmax(logits,axis=-1)

def probsToLogits (probs):
    return jnp.log (probs)

def logitToProb (logit):
    return 1 / (1 + jnp.exp (logit))

def probToLogit (prob):
    return jnp.log (1/jnp.clip(prob,0,1) - 1)

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
    return jnp.array ([jnp.abs(lam), jnp.abs(mu), logitToProb(x_logits), logitToProb(y_logits)])

def indelModelToLogits (lam, mu, x, y):
    return jnp.array ([lam, mu, probToLogit(x), probToLogit(y)])

def createGGIModelFactory (subModelFactory, nQuantiles):
    def parametricGGIModel (params):
        mixture = [subModelFactory(p['subrate'],p['rootlogits']) for p in params['subs']]
        subRate = jnp.stack([m[0] for m in mixture], axis=0)  # (nColTypes,A,A)
        colShape = params['colshape']
        colGammas = [Gamma(s,s) for s in colShape]
        quantileProbs = (2*jnp.arange(nQuantiles) + 1) / (2*nQuantiles)
        colQuantiles = jnp.stack ([jnp.array ([g.quantile(p) for g in colGammas]) for p in quantileProbs], axis=0)  # (nQuantiles,nColTypes)
        colQuantiles /= jnp.mean(colQuantiles,axis=0,keepdims=True)  # maintain E[rate]=1
        subRate = jnp.einsum ('cij,qc->qcij', subRate, colQuantiles)  # (nQuantiles,nColTypes,A,A)
        rootProb = jnp.stack([m[1] for m in mixture], axis=0)  # (nColTypes,A)
        rootProb = jnp.repeat (rootProb[None,:,:], nQuantiles, axis=0)  # (nQuantiles,nColTypes,A)
        indelParams = [parametricIndelModel(*p) for p in params['indels']]  # (nAlignTypes,4)
        alnTypeWeight = logitsToProbs(params['alntypelogits'])  # (nAlignTypes,)
        colTypeWeight = logitsToProbs(params['coltypelogits'])  # (nAlignTypes,nColTypes)
        return subRate, rootProb, indelParams, alnTypeWeight, colTypeWeight, colQuantiles
    return parametricGGIModel

def parseHistorianParams (params):
    alphabet = params['alphabet']
    def parseSubRate (p):
        subRate = jnp.array([[p['subrate'].get(i,{}).get(j,0) for j in alphabet] for i in alphabet], dtype=jnp.float32)
        rootProb = jnp.array([p['rootprob'].get(i,0) for i in alphabet], dtype=jnp.float32)
        subRate, rootProb = normalizeSubModel (subRate, rootProb)
        return subRate, rootProb
    if 'mixture' in params or 'coltype' in params:
        mixtureJson = params.get('mixture',params.get('coltype'))  # backward compatible name for coltype is 'mixture'
        mixture = [parseSubRate(p) for p in mixtureJson]
        colShape = jnp.array([p.get('shape',1) for p in mixtureJson], dtype=jnp.float32)
    else:
        mixture = [parseSubRate(params)]
        colShape = jnp.array([params.get('shape',1)], dtype=jnp.float32)
    def parseIndelParams (cpt):
        return tuple(cpt.get(name,0) for name in ['insrate','delrate','insextprob','delextprob'])
    alignTypes = params.get ('aligntype', [params])
    indelParams = [parseIndelParams(t) for t in alignTypes]
    alnTypeLogits = jnp.log(jnp.array([t.get('weight',1) for t in alignTypes], dtype=jnp.float32))   # alnTypeLogits[i] \propto log P(alignType=i)
    colTypeLogits = jnp.log(jnp.stack([jnp.array(t.get('coltypeweight',jnp.ones(len(mixture), dtype=jnp.float32))) for t in alignTypes], axis=0))  # colTypeLogits[i,j] \propto log P(columnType=j | alignType=i)
    quantiles = params.get('quantiles',1)  # number of quantiles (rate classes) for column gamma distributions
    return alphabet, mixture, indelParams, alnTypeLogits, colTypeLogits, colShape, quantiles

def toHistorianParams (alphabet, params, sub_model_factory, nQuantiles):
    def toSubRate (shape, subParams):
        subRate, rootProb = sub_model_factory(subParams['subrate'], subParams['rootlogits'])
        json = { 'subrate': {alphabet[i]: {alphabet[j]: float(subRate[i,j]) for j in range(len(alphabet)) if i != j} for i in range(len(alphabet))},
                 'rootprob': {alphabet[i]: float(rootProb[i]) for i in range(len(alphabet))} }
        if shape is not None:
            json['shape'] = float(shape)
        return json
    def toIndelParams (indelParams_logits):
        indelParams = parametricIndelModel(*indelParams_logits)
        return {name: float(param) for name,param in zip(['insrate','delrate','insextprob','delextprob'],indelParams)}
    nColTypes, nAlignTypes = params['coltypelogits'].shape
    # use backwardly-compatible format if possible, so Historian can read it
    if nAlignTypes == 1 and nQuantiles == 1:
        if nColTypes == 1:
            json = {**toSubRate(None, params['subs'][0]),
                    **toIndelParams(params['indels'][0])}
        else:
            json = {'coltype': [toSubRate(None, params['subs'][n]) for n in range(nColTypes)],
                    **toIndelParams(params['indels'][0])}
    else:
        alnTypeProbs = jax.nn.softmax(params['alntypelogits'])
        colTypeProbs = jax.nn.softmax(params['coltypelogits'], axis=-1)
        json = {'coltype': [toSubRate(params['colshape'][n],params['subs'][n]) for n in range(nColTypes)],
                'aligntype': [{'weight': float(alnTypeProbs[i]),
                            'coltypeweight': [float(w) for w in colTypeProbs[i,:]],
                            **toIndelParams(params['indels'][i])} for i in range(nAlignTypes)],
                'quantiles': nQuantiles}
    json['alphabet'] = alphabet
    return json
