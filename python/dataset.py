import os
import glob

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

import logging

import cigartree
import likelihood

def loadTreeAndAlignment (treeFilename, alignFilename, alphabet):
    with open(treeFilename, 'r') as f:
        treeStr = f.read()
    with open(alignFilename, 'r') as f:
        alignStr = f.read()

    ct = cigartree.makeCigarTree (treeStr, alignStr)
    seqs, _nodeName, distanceToParent, parentIndex, transCounts = cigartree.getHMMSummaries (treeStr, alignStr, alphabet)

    unpaddedRows, unpaddedCols = seqs.shape
    seqs, parentIndex, distanceToParent, transCounts = likelihood.padAlignment (seqs, parentIndex, distanceToParent, transCounts)
    paddedRows, paddedCols = seqs.shape
    logging.warning("Padded alignment %s from %d x %d to %d x %d" % (os.path.basename(alignFilename), unpaddedRows, unpaddedCols, paddedRows, paddedCols))

    return seqs, parentIndex, distanceToParent, transCounts

def loadMultipleTreesAndAlignments (treeDir, alignDir, alphabet, families = None, limit = None, treeSuffix = '.nh', alignSuffix = '.aa.fasta'):
    if families is not None:
        treeFiles = ['%s/%s%s' % (treeDir,f,treeSuffix) for f in families]
    else:
        treeFiles = glob.glob('%s/*%s' % (treeDir,treeSuffix))
        families = [os.path.basename(f).replace(treeSuffix,'') for f in treeFiles]
    alignFiles = ['%s/%s%s' % (alignDir,f,alignSuffix) for f in families]
    for n, (treeFile, alignFile) in enumerate(zip(treeFiles, alignFiles)):
        if limit is not None and n >= limit:
            break
        logging.warning("Reading #%d: tree %s and alignment %s" % (n+1, treeFile, alignFile))
        yield loadTreeAndAlignment (treeFile, alignFile, alphabet)

def loadTreeFamData (treeFamDir, alphabet, **kwargs):
    return loadMultipleTreesAndAlignments (treeFamDir, treeFamDir, alphabet, **kwargs)

def createLossFunction (dataset, model_factory, includeSubs = True, includeIndels = True, useKM03 = False):
    def loss (params):
        subRate, rootProb, indelParams, alnTypeWeight, colTypeWeight, colQuantiles = model_factory (params)
        alphabetSize = subRate.shape[-1]
        logQuantiles = jnp.log(len(colQuantiles))
        colTypeLogWeight = jnp.log(colTypeWeight)
        alnTypeLogWeight = jnp.log(alnTypeWeight)
        l_total = 0.
        for seqs, parentIndex, distanceToParent, transCounts in dataset:
            if includeSubs:
                sub_ll = likelihood.subLogLike (seqs, distanceToParent, parentIndex, subRate, rootProb)  # (nQuantiles, nColTypes, nCols)
                sub_ll = jnp.sum(sub_ll,axis=-1)  # (nQuantiles, nColTypes)
                sub_ll = logsumexp(sub_ll,axis=0) - logQuantiles  # (nColTypes,)
                sub_ll = colTypeLogWeight + sub_ll[None,:]  # (nAlignTypes, nColTypes)
                sub_ll = logsumexp(sub_ll, axis=-1)  # (nAlignTypes,)
            else:
                sub_ll = 0.
            if includeIndels:
                trans_ll = jnp.array ([jnp.sum (likelihood.transLogLike (transCounts, distanceToParent, p, alphabetSize=alphabetSize, useKM03=useKM03)) for p in indelParams])  # (nAlignTypes,)
            else:
                trans_ll = 0.
            jax.debug.print("sub_ll={} trans_ll={}", sub_ll, trans_ll)
            l_total -= logsumexp(alnTypeLogWeight + trans_ll + sub_ll)
        return l_total
    return loss
