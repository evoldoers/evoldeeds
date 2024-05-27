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
    seqs, _nodeName, distanceToParent, parentIndex, transCounts = cigartree.getHMMSummaries (treeStr, alignStr, alphabet)

    discretizedDistanceToParent = likelihood.discretizeBranchLength (distanceToParent)

    unpaddedRows, unpaddedCols = seqs.shape
    seqs, parentIndex, discretizedDistanceToParent, transCounts = likelihood.padAlignment (seqs, parentIndex, discretizedDistanceToParent, transCounts)
    paddedRows, paddedCols = seqs.shape
    logging.warning("Padded alignment %s from %d x %d to %d x %d" % (os.path.basename(alignFilename), unpaddedRows, unpaddedCols, paddedRows, paddedCols))
    return seqs, parentIndex, discretizedDistanceToParent, transCounts

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

def createLossFunction (dataset, model_factory, alphabet, useKM03 = False):
    def loss (params):
        subRate, rootProb, indelParams = model_factory (params)
        discSubMatrix = likelihood.computeSubMatrixForDiscretizedTimes (subRate)
        discTransMat = likelihood.computeTransMatForDiscretizedTimes (indelParams, alphabet, useKM03=useKM03)
        ll = 0.
        for seqs, parentIndex, discretizedDistanceToParent, transCounts in dataset:
            trans_ll = likelihood.transLogLikeForTransMats (transCounts,
                                                            likelihood.getTransMatForDiscretizedTimes (discretizedDistanceToParent, discTransMat))
            sub_ll = likelihood.subLogLikeForMatrices (seqs, parentIndex,
                                                       likelihood.getSubMatrixForDiscretizedBranchLengths (discretizedDistanceToParent, discSubMatrix),
                                                       rootProb)
            ll = ll - jnp.sum(sub_ll) - jnp.sum(trans_ll)
        return ll
    return loss
