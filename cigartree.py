import numpy as np
import jax.numpy as jnp
import newick

def parseNewick (newickStr):
    root = newick.loads(newickStr)[0]
    nodes = [n for n in root.walk()]
    parentIndex = jnp.array([nodes.index(n.ancestor) if n.ancestor is not None else -1 for n in nodes], dtype=jnp.int32)
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

def getNumCols (seqs):
    seqLengths = [len(s) for s in seqs if s is not None]
    assertSame (seqLengths, "Alignment rows are supposed to be the same length, but are not")
    assert len(seqLengths) > 0, "Must have at least one sequence in alignment"
    return seqLengths[0]

def orderAlignmentSeqs (seqByName, nodeName, gapChar = '-'):
    seqs = [seqByName.get(n,None) if n is not None else None for n in nodeName]
    nCols = getNumCols(seqs)
    gapRow = gapChar * nCols
    seqs = [s if s is not None and len(s) > 0 else gapRow for s in seqs]
    return seqs

def getChildren (parentIndex):
    nRows = len(parentIndex)
    children = [[] for _ in range(nRows)]
    for row in range(nRows):
        if parentIndex[row] >= 0:
            children[parentIndex[row]].append (row)
    return children

def addPathsToMRCAs (seqs, parentIndex, gapChar = '-', wildChar = 'x'):
    nRows = len(seqs)
    nCols = getNumCols(seqs)
    children = getChildren(parentIndex)
    seqs = [list(s) for s in seqs]
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
    seqs = [''.join(s) for s in seqs]
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
    return [getExpandedCigarString(seqs[parentIndex[r]] if r > 0 else gapChar*nCols,seqs[r]) for r in range(nRows)]

def countGapSizes (expandedCigars):
    def countCigar (excig):
        nInsertions = 0
        nDeletions = 0
        gapSizeCount = {}
        transitionCount = np.zeros((3,3))
        def countGapSize():
            nonlocal gapSizeCount, nDeletions, nInsertions
            gapSize = "%s %s" % (nDeletions, nInsertions)
            gapSizeCount[gapSize] = gapSizeCount.get(gapSize,0) + 1
        stateIndex = lambda c: 'MID'.index(c)
        prev = stateIndex('M')
        for c in excig:
            if c == 'I':
                nInsertions = nInsertions + 1
            elif c == 'D':
                nDeletions = nDeletions + 1
            else:
                countGapSize()
                nInsertions = nDeletions = 0
            state = stateIndex(c)
            transitionCount[prev][state] = transitionCount[prev][state] + 1
            prev = state
        countGapSize()
        transitionCount[prev][stateIndex('M')] = transitionCount[prev][stateIndex('M')] + 1
        return gapSizeCount, transitionCount
    counts = [countCigar(x) for x in expandedCigars]
    return tuple([c[i] for c in counts] for i in range(2))  # gapSizeCount, transitionCount

def compressCigarString (stateStr):
    s = None
    n = 0
    cigar = []
    for c in stateStr:
        if s != c:
            if s is not None:
                cigar.append ("%d%s" % (n,s))
            s = c
            n = 0
        n = n + 1
    if s is not None:
        cigar.append ("%d%s" % (n,s))
    return ''.join(cigar)

def makeCigarTree (newickStr, fastaStr, gapChar = '-'):
    parentIndex, distanceToParent, nodeName = parseNewick (newickStr)
    _seqNames, seqByName = parseFasta (fastaStr)
    seqs = orderAlignmentSeqs (seqByName, nodeName)
    seqs = addPathsToMRCAs (seqs, parentIndex, gapChar)
    expandedCigars = getExpandedCigarsFromAlignment (seqs, parentIndex, gapChar)
    cigars = [compressCigarString(x) for x in expandedCigars]
    children = getChildren (parentIndex)
    def makeNode(n):
        node = {}
        id = nodeName[n]
        if id:
            node['id'] = id
        if n > 0:
            node['distance'] = float(distanceToParent[n])
        node['cigar'] = cigars[n]
        if id and id in seqByName:
            node['seq'] = seqByName[id].replace(gapChar,'')
        if len(children[n]) > 0:
            node['child'] = [makeNode(c) for c in children[n]]
        return node
    return makeNode(0)

def getHMMSummaries (newickStr, fastaStr, alphabet, gapChar = '-'):
    parentIndex, distanceToParent, nodeName = parseNewick (newickStr)
    _seqNames, seqByName = parseFasta (fastaStr)
    seqs = orderAlignmentSeqs (seqByName, nodeName)
    seqs = addPathsToMRCAs (seqs, parentIndex)
    seqs = [s.lower() for s in seqs]
    expandedCigars = getExpandedCigarsFromAlignment (seqs, parentIndex)
    _gapSizes, transCounts = countGapSizes(expandedCigars)
    transCounts = jnp.stack (transCounts, axis=0)
    seqs = tokenizeAlignment (seqs, alphabet)
    return seqs, distanceToParent, parentIndex, transCounts


def tokenizeAlignment (seqs, alphabet):
    return jnp.array([[alphabet.index(c) if c in alphabet else -1 for c in seq] for seq in seqs], dtype=jnp.int32)
