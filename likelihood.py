import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm

import newick

# Compute substitution log-likelihood of a multiple sequence alignment and phylogenetic tree by pruning
# Parameters:
#  - alignment: (R,C) integer tokens. C is the length of the alignment (#cols), R is the number of sequences (#rows). A token of -1 indicates a gap.
#  - distanceToParent: (R,) floats, distance to parent node
#  - parentIndex: (R,) integers, index of parent node. Nodes are sorted in preorder so parentIndex[i] <= i for all i. parentIndex[0] = -1
#  - subRate: (*H,A,A) substitution rate matrix/matrices. Leading H axes (if any) are "hidden" substitution rate categories, A is alphabet size
#  - rootProb: (*H,A) root frequencies (typically equilibrium frequencies for substitution rate matrix)
# To pad rows, set parentIndex[paddingRow:] = arange(paddingRow,R)
# To pad columns, set alignment[paddingRow,paddingCol:] = -1

def pruneLogLike (alignment, distanceToParent, parentIndex, subRate, rootProb):
    assert alignment.ndim == 2
    assert subRate.ndim >= 2
    R, C = alignment.shape
    *H, A = subRate.shape[0:-1]
    assert distanceToParent.shape == (R,)
    assert parentIndex.shape == (R,)
    assert rootProb.shape == (*H,A)
    assert subRate.shape == (*H,A,A)
    assert alignment.dtype == jnp.int32
    assert parentIndex.dtype == jnp.int32
    assert jnp.all(alignment >= -1)
    assert jnp.all(alignment < A)
    assert jnp.all(distanceToParent >= 0)
    assert jnp.all(parentIndex[1:] >= -1)
    assert jnp.all(parentIndex <= jnp.arange(R))
    # Compute transition matrices per branch
    subMatrix = expm (jnp.einsum('...ij,r->...rij', subRate, distanceToParent))  # (*H,R,A,A)
    # Initialize pruning matrix
    tokenLookup = jnp.concatenate([jnp.ones(A)[None,:],jnp.eye(A)])
    likelihood = jnp.repeat (jnp.expand_dims (tokenLookup[alignment + 1], jnp.arange(len(H))), repeats=H, axis=0)  # (*H,R,C,A)
    logNorm = jnp.zeros(*H,C)  # (*H,C)
    # Compute log-likelihood for all columns in parallel by iterating over nodes in postorder
    for child in range(A-1,0,-1):
        parent = parentIndex[child]
        likelihood = likelihood.at[...,parent,:,:].multiply (jnp.einsum('...ij,...cj->...ci', subMatrix[...,child,:,:], likelihood[...,child,:,:]))
        maxLike = jnp.max(likelihood[...,parent,:,:], axis=-1)  # (*H,C)
        likelihood = likelihood.at[...,parent,:,:].divide (maxLike[...,None])
        logNorm = logNorm + jnp.log(maxLike)
    logNorm = logNorm + jnp.log(jnp.einsum('...ci,...i->...c', likelihood.at[:,0,:,:], rootProb))  # (*H,C)
    return logNorm

def padDimension (unpaddedVal, multiplier):
    if multiplier == 2:  # avoid doubling length due to precision errors
        paddedVal = 1 << (unpaddedVal-1).bit_length()
    else:
        paddedVal = int (np.ceil (multiplier ** np.ceil (np.log(unpaddedVal) / np.log(multiplier))))
    return paddedVal

def padAlignment (alignment, distanceToParent, parentIndex, nRows: int = None, nCols: int = None, colMultiplier = 2, rowMultiplier = 2):
    unpaddedRows, unpaddedCols = alignment.shape
    if nCols is None:
        nCols = padDimension (unpaddedCols, colMultiplier)
    if nRows is None:
        nRows = padDimension (unpaddedRows, rowMultiplier)
    # To pad rows, set parentIndex[paddingRow:] = arange(paddingRow,R) and pad alignment and distanceToParent with any value (-1 and 0 for predictability)
    if nRows > unpaddedRows:
        alignment = jnp.stack ([alignment, -1*jnp.ones((nRows - unpaddedRows, unpaddedCols))], axis=0)
        distanceToParent = jnp.stack ([distanceToParent, jnp.zeros(nRows - unpaddedRows)], axis=0)
        parentIndex = jnp.stack ([parentIndex, jnp.arange(unpaddedRows,nRows)], axis=0)
    # To pad columns, set alignment[paddingRow,paddingCol:] = -1
    if nCols > unpaddedCols:
        alignment = jnp.stack ([alignment, -1*jnp.ones((nRows, nCols - unpaddedCols))], axis=1)
    return alignment, distanceToParent, parentIndex

def parseNewick (newickStr):
    root = newick.loads(newickStr)[0]
    nodes = [n for n in root.walk()]
    parentIndex = jnp.array([nodes.index(n.parent) if n.parent is not None else -1 for n in nodes], dtype=jnp.int32)
    distanceToParent = jnp.array([n.length for n in nodes], dtype=jnp.float32)
    nodeName = [n.name for n in nodes]
    return parentIndex, distanceToParent, nodeName

def parseFasta (fastaStr, requireFlush = True):
    lines = fastaStr.splitlines()
    seqByName = {}
    seqNames = []
    name = None
    for line in lines:
        if line.startswith('>'):
            name = line[1:].split()[0]
            seqNames.append(name)
            seqByName[name] = ''
        else:
            seqByName[name] += line
    if requireFlush:
        seqLengths = [len(s) for s in seqByName.values()]
        assertSame (seqLengths, "Sequences are supposed to be the same length, but are not")
    return seqNames, seqByName

def assertSame (l, error):
    if len(l) > 0:
        assert len([n for n in l if n != l[0]]) == 0, error

def orderAlignmentSeqs (seqByName, nodeName, parentIndex, gapChar = '-', wildChar = 'x'):
    seqs = [seqByName.get(n,None) if n is not None else None for n in nodeName]
    seqLengths = [len(s) for s in seqs if s is not None]
    assertSame (seqLengths, "Alignment rows are supposed to be the same length, but are not")
    assert len(seqLengths) > 0, "Must have at least one sequence in alignment"
    nRows = len(seqs)
    nCols = seqLengths[0]
    gapRow = gapChar * nCols
    seqs = [s if s is not None else gapRow for s in seqs]
    children = [[] for _ in range(nRows)]
    for row in range(nRows):
        if parentIndex[row] >= 0:
            children[parentIndex[row]].append (row)
    for col in range(nCols):
        nUngappedDescendants = [0] * nRows
        for row in range(nRows-1,-1,-1):
            isLeaf = len(children[row]) == 0
            if isLeaf and seqs[row][col] != gapChar:
                nUngappedDescendants[row] = 1
            else:
                nUngappedDescendants[row] = sum(nUngappedDescendants[c] for c in children[row])
        for row in range(nRows):
            isInternal = len(children[row]) > 0
            if isInternal and seqs[row][col] == gapChar and len([c for c in children[row] if nUngappedDescendants[c] > 0]) > 1:
                seqs[row][col] = wildChar
    return seqs

def getExpandedCigarsFromAlignment (seqs, parentIndex, gapChar = '-'):
    assert len(seqs) > 0, "Alignment is empty"
    nRows = len(seqs)
    nCols = len(seqs[0])
    def getCigarChar (parentChar, childChar):
        if parentChar == gapChar:
            if childChar == gapChar:
                return None
            return 'I'
        else:
            if childChar == gapChar:
                return 'D'
            return 'M'
    def getExpandedCigarString (parentRow, childRow):
        return ''.join ([c for c in [getCigarChar(parentRow[n],childRow[n]) for n in range(nCols)] if c is not None])
    return [getExpandedCigarString(seqs[parentIndex[r]] if r >= 0 else gapChar*nCols,seqs[r]) for r in range(nRows)]

def compressCigarString (stateStr):
    s = None
    n = 0
    cigar = ''
    for c in stateStr:
        if s != c:
            if s is not None:
                cigar += str(n) + s
                s = c
                n = 0
        n = n + 1
    if s is not None:
        cigar += str(n) + s
    return cigar

def tokenizeAlignment (seqs, alphabet):
    return jnp.array([[alphabet.index(c) if c in alphabet else -1 for c in seq] for seq in seqs], dtype=jnp.int32)

def parseHistorianParams (params):
    alphabet = params['alphabet']
    def parseSubRate (p):
        subRate = jnp.array([[p['subRate'].get(i,{}).get(j,0) for j in alphabet] for i in alphabet], dtype=jnp.float32)
        subRate = subRate - jnp.diag(jnp.sum(subRate, axis=-1))
        rootProb = jnp.array([p['rootProb'].get(i,0) for i in alphabet], dtype=jnp.float32)
        rootProb = rootProb / jnp.sum(rootProb)
        return subRate, rootProb
    if 'mixture' in params:
        mixture = [parseSubRate(p) for p in params['mixture']]
    else:
        mixture = [parseSubRate(params)]
    indelParams = (params.get(name,0) for name in ['insrate','delrate','insextprob','delextprob'])
    return alphabet, mixture, indelParams
