import logging
from copy import deepcopy
import numpy as np
from tqdm import tqdm

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import einops

import cigartree
import likelihood
import h20
import km03
from optimize import optimize, optimize_generic

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
    lp = sorted(leafPairs(), key=lambda x: x[1])
    # return a unique partition of nonduplicate leaves into pairs with their distances, preferring closer pairs
    def cherryPairs():
        for (i,j), dij in lp:
            if available[i] and available[j]:
                available[i] = False
                available[j] = False
                yield (i,j), dij
    return cherryPairs()

def getPosteriorWeights (data, model, alphabetSize=20, useKM03=False):
    seqs, parentIndex, distanceToParent, transCounts = data
    subRate, rootProb, indelParams, alnTypeWeight, colTypeWeight, colQuantiles = model
    alphabetSize = subRate.shape[-1]
    logQuantiles = jnp.log(len(colQuantiles))
    colTypeLogWeight = jnp.log(colTypeWeight)
    alnTypeLogWeight = jnp.log(alnTypeWeight)
    colMask = jnp.where(jnp.all(seqs < 0, axis=0), 0, 1)  # (nCols,)
    qtc_ll = likelihood.subLogLike (seqs, distanceToParent, parentIndex, subRate, rootProb)  # (nQuantiles, nColTypes, nCols)
    aqtc_ll = qtc_ll[None,...] + colTypeLogWeight[:,None,:,None] - logQuantiles  # (nAlignTypes, nQuantiles, nColTypes, nCols)
    atc_ll = logsumexp(aqtc_ll,axis=1)  # (nAlignTypes, nColTypes, nCols)
    at_ll = jnp.einsum('atc->at', atc_ll)  # (nAlignTypes, nColTypes)
    a_ll = logsumexp(at_ll,axis=-1)  # (nAlignTypes,)
    a_ll += jnp.array ([jnp.sum (likelihood.transLogLike (transCounts, distanceToParent, p, alphabetSize=alphabetSize, useKM03=useKM03)) for p in indelParams])  # (nAlignTypes,)
    a_ll += alnTypeLogWeight  # (nAlignTypes,)
    ll = jnp.sum(a_ll)
    a_pp = jax.nn.softmax(a_ll)  # (nAlignTypes,)
    atc_pp = jax.nn.softmax(atc_ll,axis=1)  # (nAlignTypes, nColTypes, nCols).  atc_pp[a,t,c] = P(column c has type t given alignment type is a)
    at_count = jnp.einsum ('atc,c->at', atc_pp, colMask)  # (nAlignTypes, nColTypes).  at_count[a,c] = E[number of type c columns for alignment type a]
    tc_pp = jnp.einsum ('a,atc->tc', a_pp, atc_pp)  # (nColTypes, nCols).  tc_pp[t,c] = P(column c has type t)
    qtc_pp = tc_pp[None,...] * jax.nn.softmax(qtc_ll,axis=0)  # (nQuantiles, nColTypes, nCols).  qtc_pp[q,t,c] = P(column c has type t and rate quantile q)
    return a_pp, at_count, qtc_pp, ll

defaultLogBase = 10**.1
def discretizeTime (t, logBase=defaultLogBase):
    return jnp.round (jnp.log(t) / jnp.log(logBase))

def undiscretizeTime (t, logBase=defaultLogBase):
    return jnp.power (logBase, t)

def countWeightedCherries (data, alphabetSize, a_pp = None, qtc_pp = None, rootCount=None, count_by_t=None, logBase=defaultLogBase):
    seqs, parentIndex, distanceToParent, _transCounts = data
    _nRows, nCols = seqs.shape
    if a_pp is not None:
        nAlignTypes = len(a_pp)
    else:
        nAlignTypes = 1
        a_pp = jnp.ones (nAlignTypes)
    if qtc_pp is not None:
        nQuantiles, nColTypes, _ = qtc_pp.shape
    else:
        nQuantiles = 1
        nColTypes = 1
        qtc_pp = jnp.ones ((nQuantiles, nColTypes, nCols))
    if rootCount is None:
        rootCount = np.zeros ((nQuantiles, nColTypes, alphabetSize))
    if count_by_t is None:
        count_by_t = {}  # count_by_t[discretizedTime] = (subRateCount, transCounts)
    for (i,j), dij in pickCherries (parentIndex, distanceToParent):
        t = discretizeTime (dij, logBase=logBase).item()
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
                            subRateCount[quantile,colType,ci,cj] += count
                            subRateCount[quantile,colType,cj,ci] += count
        
        count_by_t[t] = subRateCount, transCount

    return rootCount, count_by_t

def cherryCountsToArrays (count_by_t):
    dts = sorted(count_by_t.keys())
    ts = undiscretizeTime (np.array (dts))  # (nDiscretizedTimes,)
    subRateCount = np.stack ([count_by_t[dt][0] for dt in dts], axis=0)  # (nDiscretizedTimes, nQuantiles, nColTypes, alphabetSize, alphabetSize)
    transCount = np.stack ([count_by_t[dt][1] for dt in dts], axis=0)  # (nDiscretizedTimes, nAlignTypes, 3, 3)
    return ts, subRateCount, transCount

def getDatasetCounts (dataset, alphabetSize, logBase=defaultLogBase):
    logging.warning("Preprocessing alignment dataset")
    for data in tqdm(dataset):
        rootCount, count_by_t = countWeightedCherries (data, alphabetSize, logBase=logBase)
        ts, subRateCount, transCount = cherryCountsToArrays (count_by_t)
        yield rootCount, count_by_t, ts, subRateCount, transCount

def getPosteriorCounts (dataset, params, model_factory, getPosteriorWeights=getPosteriorWeights, logBase=defaultLogBase, **kwargs):
    model = model_factory (params)
    subRate, _rootProb, _indelParams, _alnTypeWeight, colTypeWeight, colQuantiles = model
    nAlignTypes, nColTypes = colTypeWeight.shape
    nQuantiles = len(colQuantiles)
    alphabetSize = subRate.shape[-1]
    a_count = np.zeros (nAlignTypes)
    at_count = np.zeros ((nAlignTypes, nColTypes))
    rootCount = np.zeros ((nQuantiles, nColTypes, alphabetSize))
    count_by_t = {}  # count_by_t[discretizedTime] = (subRateCount, transCounts)
    ll = 0.
    for data in tqdm(dataset):
        a_pp, at_c, qtc_pp, a_ll = getPosteriorWeights (data, model, **kwargs)
        a_count += a_pp
        at_count += at_c
        ll += a_ll
        rootCount, count_by_t = countWeightedCherries (data, alphabetSize, a_pp=a_pp, qtc_pp=qtc_pp, rootCount=rootCount, count_by_t=count_by_t, logBase=logBase)
    ts, subRateCount, transCount = cherryCountsToArrays (count_by_t)
    return a_count, at_count, ts, rootCount, subRateCount, transCount, ll

def selectAlignAndColType (params, nType):
    params = deepcopy(params)
    params['alntypelogits'] = jnp.zeros(1)
    params['coltypelogits'] = jnp.zeros((1,1))
    params['subs'] = [params['subs'][nType]]
    params['indels'] = [params['indels'][nType]]
    params['colshape'] = [params['colshape'][nType]]
    return params

def createCompositeSubLoss (model_factory):
    def compositeSubLoss (params, ts, rootCount, subRateCount):
        subRate, rootProb, _indelParams, _alnTypeWeight, _colTypeWeight, _colQuantiles = model_factory (params)
        subMatrix = likelihood.computeSubMatrixForTimes (ts, subRate)  # (nQuantiles, nColTypes, nDiscretizedTimes, alphabetSize, alphabetSize)
        subMatrix = einops.rearrange (subMatrix, 'q c t i j -> t q c i j')
        return -jnp.sum(subRateCount * jnp.log(subMatrix)) - jnp.sum(rootCount * jnp.log(rootProb))
    return compositeSubLoss

def createCompositeIndelLoss (model_factory, useKM03=False, **kwargs):
    transitionMatrix = km03.transitionMatrix if useKM03 else h20.transitionMatrix
    def compositeIndelLoss (params, ts, transCount):
        subRate, _rootProb, indelParams, alnTypeWeight, _colTypeWeight, _colQuantiles = model_factory (params)
        nAlignTypes = alnTypeWeight.shape[0]
        alphabetSize = subRate.shape[-1]
        loss = 0.
        for a in range(nAlignTypes):
            logTransMat = jnp.log (jnp.stack ([transitionMatrix(t,indelParams[a],alphabetSize=alphabetSize) for t in ts], axis=0))
            loss -= jnp.sum (transCount[:,a,:,:] * logTransMat)
        return loss
    return compositeIndelLoss

def createCompositeAlignmentLogLike (sub_loss, indel_loss):
    def compositeAlignmentLogLike (params, ts, rootCount, subRateCount, transCount):
        return -(sub_loss (params, ts, rootCount, subRateCount) + indel_loss (params, ts, transCount))
    return compositeAlignmentLogLike

def optimizeByPhyloExpectationCompositeMaximization (dataset, model_factory, params, use_jit=True, useKM03=False, init_lr=None, show_grads=None, verbosity=1, **kwargs):
    if use_jit:
        value_and_grad = lambda f: jax.jit (jax.value_and_grad (f))
        getPosteriorWeights_jit = jax.jit (getPosteriorWeights, static_argnames=('alphabetSize', 'useKM03'))
    else:
        value_and_grad = jax.value_and_grad
        getPosteriorWeights_jit = getPosteriorWeights
    sub_loss_value_and_grad = value_and_grad (createCompositeSubLoss (model_factory))
    trans_loss_value_and_grad = value_and_grad (createCompositeIndelLoss (model_factory, useKM03=useKM03))
    optax_args = dict((k,v) for k,v in [('init_lr',init_lr),('show_grads',show_grads)] if v is not None)
    def take_step (params, nStep):
        logging.warning("E-step %d: computing posterior counts" % (nStep+1))
        a_count, at_count, ts, rootCount, subRateCount, transCount, ll = getPosteriorCounts(dataset, params, model_factory, getPosteriorWeights=getPosteriorWeights_jit, useKM03=useKM03)
        logging.warning ("M-step %d: optimizing composite likelihoods (actual loss %f)" % (nStep+1, -ll))
        sub_loss_vg_bound = lambda params: sub_loss_value_and_grad (params, ts, rootCount, subRateCount)
        trans_loss_vg_bound = lambda params: trans_loss_value_and_grad (params, ts, transCount)
        params, _sub_ll = optimize (sub_loss_vg_bound, params, prefix=f"EM step {nStep+1}, substitution params iteration ", verbose=verbosity>1, **optax_args, **kwargs)
        #*stuff, new_a_ll = getPosteriorWeights_jit (dataset[0], model_factory(params))
        #logging.warning('a_ll = %f, new_a_ll = %f' % (ll,new_a_ll))
        params, _trans_ll = optimize (trans_loss_vg_bound, params, prefix=f"EM step {nStep+1}, indel params iteration ", verbose=verbosity>1, **optax_args, **kwargs)
        params['alntypelogits'] = jnp.log (a_count / jnp.sum(a_count))
        params['coltypelogits'] = jnp.log (at_count / jnp.sum(at_count,axis=-1,keepdims=True))
        return params, -ll
    return optimize_generic (take_step, params, prefix="EM step ", verbose=verbosity>0, **kwargs)

def optimizeByCompositeExpectationMaximization (dataset, model_factory, params, use_jit=True, useKM03=False, init_lr=None, show_grads=None, verbosity=1, **kwargs):
    jit = jax.jit if use_jit else lambda f: f
    sub_loss = createCompositeSubLoss (model_factory)
    trans_loss = createCompositeIndelLoss (model_factory, useKM03=useKM03)
    sub_loss_value_and_grad = jit (jax.value_and_grad (sub_loss))
    trans_loss_value_and_grad = jit (jax.value_and_grad (trans_loss))
    align_loglike = jit (createCompositeAlignmentLogLike (sub_loss, trans_loss))
    optax_args = dict((k,v) for k,v in [('init_lr',init_lr),('show_grads',show_grads)] if v is not None)
    alphabetSize = params['subs'][0]['subrate'].shape[-1]
    nQuantiles = len(params['colshape'])
    nAlignTypes = params['alntypelogits'].shape[0]
    assert nQuantiles==1, 'Composite EM currently only supports one quantile (FIXME)'  # could fix by prescaling columns by estimated rate, as in CherryML paper
    assert params['coltypelogits'] == jnp.log(jnp.eye(nAlignTypes)), 'Composite EM requires a one-to-one mapping between alignments & columns'
    nColTypes = nAlignTypes
    datasetCounts = list(getDatasetCounts (dataset, alphabetSize))
    def take_step (params, nStep):
        rootCount = np.zeros ((nQuantiles, nColTypes, alphabetSize))
        count_by_t = {}  # count_by_t[discretizedTime] = (subRateCount, transCounts)
        logging.warning("E-step %d: computing posterior alignment weights" % (nStep+1))
        ll = 0.
        a_count = np.zeros (nAlignTypes)
        a_logprior = params['alntypelogits'] - logsumexp(params['alntypelogits'])
        for rc, cbt, ts, subRateCount, transCount in tqdm(datasetCounts):
            a_ll = jnp.array ([align_loglike (selectAlignAndColType(params,n), ts, rc, subRateCount, transCount) for n in range(nAlignTypes)])
            a_pp = jax.nn.softmax(a_ll)
            a_count += a_pp
            ll += logsumexp (a_ll + a_logprior)
            for dt in cbt:
                subRateCount_dt_for_aln, transCount_dt_for_aln = cbt[dt]
                subRateCount_dt, transCount_dt = count_by_t.get (dt, (np.zeros((nQuantiles, nColTypes, alphabetSize, alphabetSize)),
                                                                      np.zeros((nAlignTypes, 3, 3))))
                subRateCount_dt += subRateCount_dt_for_aln * a_pp[None,:,None,None]
                transCount_dt += transCount_dt_for_aln * a_pp[:,None,None]
                count_by_t[dt] = subRateCount_dt, transCount_dt
        ts, subRateCount, transCount = cherryCountsToArrays (count_by_t)
        logging.warning ("M-step %d: optimizing composite likelihoods (total loss %f)" % (nStep+1, -ll))
        sub_loss_vg_bound = lambda params: sub_loss_value_and_grad (params, ts, rootCount, subRateCount)
        trans_loss_vg_bound = lambda params: trans_loss_value_and_grad (params, ts, transCount)
        params, _sub_ll = optimize (sub_loss_vg_bound, params, prefix=f"EM step {nStep+1}, substitution params iteration ", verbose=verbosity>1, **optax_args, **kwargs)
        params, _trans_ll = optimize (trans_loss_vg_bound, params, prefix=f"EM step {nStep+1}, indel params iteration ", verbose=verbosity>1, **optax_args, **kwargs)
        params['alntypelogits'] = jnp.log (a_count / jnp.sum(a_count))
        return params, -ll
    return optimize_generic (take_step, params, prefix="EM step ", verbose=verbosity>0, **kwargs)