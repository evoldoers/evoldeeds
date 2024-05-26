import os
import glob

import jax.numpy as jnp

import cigartree
import likelihood

def loadTreeAndAlignment (treeFilename, alignFilename, alphabet):
    with open(treeFilename, 'r') as f:
        treeStr = f.read()
    with open(alignFilename, 'r') as f:
        alignStr = f.read()

    ct = cigartree.makeCigarTree (treeStr, alignStr)
    seqs, _nodeName, distanceToParent, parentIndex, _transCounts = cigartree.getHMMSummaries (treeStr, alignStr, alphabet)

    discretizedDistanceToParent = cigartree.discretizeBranchLengths (ct, distanceToParent)

    return seqs, parentIndex, discretizedDistanceToParent

def loadMultipleTreesAndAlignments (treeDir, alignDir, alphabet, treeSuffix = '.nh', alignSuffix = '.aa.fasta'):
    treeFiles = glob.glob('%s/*%s' % (treeDir,treeSuffix))
    families = [os.path.basename(f).replace(treeSuffix,'') for f in treeFiles]
    alignFiles = ['%s/%s%s' % (alignDir,f,alignSuffix) for f in families]
    for treeFile, alignFile in zip(treeFiles, alignFiles):
        yield loadTreeAndAlignment (treeFile, alignFile, alphabet)

def loadTreeFamData (treeFamDir, alphabet, **kwargs):
    return loadMultipleTreesAndAlignments (treeFamDir, treeFamDir, alphabet, **kwargs)

def createLossFunction (dataset):
    def loss (params):
        subRate = params['subrate']
        rootProb = params['rootprob']
        subRate, rootProb = likelihood.normalizeSubModel (subRate, rootProb)
        discSubMatrix = likelihood.computeSubMatrixForDiscretizedTimes (subRate)
        return sum (-jnp.sum (likelihood.subLogLikeForMatrices (seqs, parentIndex,
                                                                likelihood.computeSubMatrixForDiscretizedBranchLengths (distanceToParent, discSubMatrix),
                                                                rootProb) for seqs, parentIndex, distanceToParent in dataset))
    return loss
