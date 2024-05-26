import os
import glob

import jax.numpy as jnp

import logging

import cigartree
import likelihood

def loadTreeAndAlignment (treeFilename, alignFilename, alphabet):
    with open(treeFilename, 'r') as f:
        treeStr = f.read()
    with open(alignFilename, 'r') as f:
        alignStr = f.read()

    ct = cigartree.makeCigarTree (treeStr, alignStr)
    seqs, _nodeName, distanceToParent, parentIndex, _transCounts = cigartree.getHMMSummaries (treeStr, alignStr, alphabet)

    discretizedDistanceToParent = likelihood.discretizeBranchLength (distanceToParent)

    seqs, parentIndex, discretizedDistanceToParent = likelihood.padAlignment (seqs, parentIndex, discretizedDistanceToParent)
    return seqs, parentIndex, discretizedDistanceToParent

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

def createLossFunction (dataset, model_factory):
    def loss (params):
        subRate, rootProb = model_factory (params['subrate'], params['root'])
        discSubMatrix = likelihood.computeSubMatrixForDiscretizedTimes (subRate)
        ll = 0.
        for seqs, parentIndex, discretizedDistanceToParent in dataset:
            sub_ll = likelihood.subLogLikeForMatrices (seqs, parentIndex,
                                                        discSubMatrix[...,discretizedDistanceToParent,:,:],
                                                        rootProb)
            ll = ll - jnp.sum(sub_ll)
        return ll
    return loss
