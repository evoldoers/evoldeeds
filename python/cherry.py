from functools import partial

import numpy as np

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import einops

import cigartree
import likelihood

def pickCherries (parentIndex, distanceToParent):
    assert len(parentIndex) == len(distanceToParent), "Parent index and distance to parent must have the same length"
    assert sum(1 for n,p in enumerate(parentIndex) if p > n) == 0, "Parent index must be sorted in preorder"
    assert list(n for n,p in enumerate(parentIndex) if p < 0) == [0], "There must be exactly one root node"
    assert sum(1 for d in distanceToParent if d < 0) == 0, "All distances must be nonnegative"
    # find leaves
    nodes = len(parentIndex)
    nChildren = [0] * nodes
    for parent in parentIndex:
        if parent >= 0:
            nChildren[parent] += 1  # padding nodes with parentIndex[n]=n will be flagged as having one child here, and so excluded from leaves, which is what we want
    leaves = [i for i,n in enumerate(nChildren) if n == 0]
    # for each pair of leaves, find MRCA and thereby distance between leaves
    available = [True] * nodes
    def leafPairs():
        for ni,i in enumerate(leaves):
            if available[i]:
                for j in leaves[ni+1:]:
                    if available[j]:
                        ia, ja, = i, j
                        dij = 0
                        while ia != ja:
                            if ia > ja:
                                dij += distanceToParent[ia]
                                ia = parentIndex[ia]
                            else:
                                dij += distanceToParent[ja]
                                ja = parentIndex[ja]
                        if dij > 0:  # avoid zero-length branches
                            yield (i,j), dij
                        else:
                            available[j] = False  # remove duplicates
    lp = list(leafPairs()).sort (key=lambda x: x[1])
    # return a unique partition of nonduplicate leaves into pairs with their distances, preferring closer pairs
    def cherryPairs():
        for (i,j), dij in lp:
            if available[i] and available[j]:
                available[i] = False
                available[j] = False
                yield (i,j), dij
    return cherryPairs()

@jax.jit
def getPosteriorWeights (data, model, alphabetSize=20, useKM03=False):
    seqs, parentIndex, distanceToParent, transCounts = data
    subRate, rootProb, indelParams, alnTypeWeight, colTypeWeight, colQuantiles = model
    alphabetSize = subRate.shape[-1]
    logQuantiles = jnp.log(len(colQuantiles))
    colTypeLogWeight = jnp.log(colTypeWeight)
    alnTypeLogWeight = jnp.log(alnTypeWeight)
    qtc_ll = likelihood.subLogLike (seqs, distanceToParent, parentIndex, subRate, rootProb)  # (nQuantiles, nColTypes, nCols)
    aqtc_ll = qtc_ll[None,...] + colTypeLogWeight[:,None,:,None] - logQuantiles  # (nAlignTypes, nQuantiles, nColTypes, nCols)
    atc_ll = logsumexp(aqtc_ll,axis=1)  # (nAlignTypes, nColTypes, nCols)
    at_ll = jnp.sum(atc_ll,axis=-1)  # (nAlignTypes, nColTypes)
    a_ll = logsumexp(at_ll,axis=-1)  # (nAlignTypes,)
    a_ll += jnp.array ([jnp.sum (likelihood.transLogLike (transCounts, distanceToParent, p, alphabetSize=alphabetSize, useKM03=useKM03)) for p in indelParams])  # (nAlignTypes,)
    a_ll += alnTypeLogWeight  # (nAlignTypes,)
    a_pp = jax.nn.softmax(a_ll)  # (nAlignTypes,)
    atc_pp = jax.nn.softmax(atc_ll,axis=1)  # (nAlignTypes, nColTypes, nCols).  atc_pp[a,t,c] = P(column c has type t given alignment type is a)
    at_count = jnp.einsum ('atc->at', atc_pp)  # (nAlignTypes, nCols).  at_count[a,c] = E[number of type c columns for alignment type a]
    tc_pp = jnp.einsum ('a,atc->tc', a_pp, atc_pp)  # (nColTypes, nCols).  tc_pp[t,c] = P(column c has type t)
    qtc_pp = tc_pp[None,...] * jax.nn.softmax(qtc_ll,axis=0)  # (nQuantiles, nColTypes, nCols).  qtc_pp[q,t,c] = P(column c has type t and rate quantile q)
    return a_pp, at_count, qtc_pp

defaultLogBase = 10**.1
def discretizeTime (t, logBase=defaultLogBase):
    return jnp.round (jnp.log(t) / jnp.log(logBase))

def undiscretizeTime (t, logBase=defaultLogBase):
    return jnp.power (logBase, t)

def getPosteriorCounts (dataset, params, model_factory, logBase=defaultLogBase, **kwargs):
    model = model_factory (params)
    subRate, _rootProb, _indelParams, _alnTypeWeight, colTypeWeight, colQuantiles = model
    nAlignTypes, nColTypes = colTypeWeight.shape
    nQuantiles = len(colQuantiles)
    alphabetSize = subRate.shape[-1]
    a_count = np.zeros (nAlignTypes)
    at_count = np.zeros ((nAlignTypes, nColTypes))
    rootCount = np.zeros ((nQuantiles, nColTypes, alphabetSize))
    count_by_t = {}  # count_by_t[discretizedTime] = (subRateCount, transCounts)
    for data in dataset:
        seqs, parentIndex, distanceToParent, transCounts = data
        _nRows, nCols = seqs.shape
        a_pp, at_c, qtc_pp = getPosteriorWeights (data, model, **kwargs)
        a_count += a_pp
        at_count += at_c
        for (i,j), dij in pickCherries (parentIndex, distanceToParent):
            t = discretizeTime (dij, logBase=logBase)
            subRateCount, transCount = count_by_t.get(t, (np.zeros((nQuantiles, nColTypes, alphabetSize, alphabetSize)),
                                                          np.zeros((nAlignTypes, 3, 3))))

            _pairGapSize, pairTransCount = cigartree.countGapSizesInTokenizedPairwiseAlignment (seqs[i,:], seqs[j,:])
            pairTransCount = (pairTransCount + pairTransCount.transpose()) / 2
            for a in range(nAlignTypes):
                transCount[a,:,:] += pairTransCount * a_pp[a]

            for col in range(nCols):
                ci = seqs[i,col]
                cj = seqs[j,col]
                if ci >= 0 or cj >= 0:
                    for quantile in range(nQuantiles):
                        for colType in range(nColTypes):
                            count = qtc_pp[quantile,colType,col] / 2
                            if ci >= 0:
                                rootCount[quantile,colType,ci] += count
                            if cj >= 0:
                                rootCount[quantile,colType,cj] += count
                            if ci >= 0 and cj >= 0:
                                subRateCount[colType,ci,cj] += count
                                subRateCount[colType,cj,ci] += count
            
            count_by_t[t] = subRateCount, transCount

    ts = sorted(count_by_t.keys())  # (nDiscretizedTimes,)
    subRateCount = np.stack ([count_by_t[t][1] for t in ts], axis=0)  # (nDiscretizedTimes, nQuantiles, nColTypes, alphabetSize, alphabetSize)
    transCount = np.stack ([count_by_t[t][2] for t in ts], axis=0)  # (nDiscretizedTimes, nAlignTypes, 3, 3)

    return a_count, at_count, ts, rootCount, subRateCount, transCount

def createCompositeSubLoss (model_factory):
    def compositeSubLoss (params, ts, rootCount, subRateCount):
        subRate, rootProb, _indelParams, _alnTypeWeight, _colTypeWeight, _colQuantiles = model_factory (params)
        subMatrix = likelihood.computeSubMatrixForTimes (ts, subRate)  # (nQuantiles, nColTypes, nDiscretizedTimes, alphabetSize, alphabetSize)
        subMatrix = einops.rearrange (subMatrix, 'q c t i j -> t q c i j')
        return -(subRateCount * jnp.log(subMatrix) + rootCount * jnp.log(rootProb))
    return compositeSubLoss

def createCompositeIndelLoss (model_factory, **kwargs):
    def compositeIndelLoss (params, ts, transCount):
        subRate, _rootProb, indelParams, alnTypeWeight, _colTypeWeight, _colQuantiles = model_factory (params)
        nAlignTypes = alnTypeWeight.shape[0]
        alphabetSize = subRate.shape[-1]
        loss = 0
        for a in range(nAlignTypes):
            loss -= jnp.sum (likelihood.transLogLike (transCount[:,a,:,:], ts, indelParams, alphabetSize=alphabetSize, **kwargs))
        return loss
    return compositeIndelLoss