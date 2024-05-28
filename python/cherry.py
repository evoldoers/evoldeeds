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
    def leafPairs():
        for ni,i in enumerate(leaves):
            for j in leaves[ni+1:]:
                ia, ja, = i, j
                dij = 0
                while ia != ja:
                    if ia > ja:
                        dij += distanceToParent[ia]
                        ia = parentIndex[ia]
                    else:
                        dij += distanceToParent[ja]
                        ja = parentIndex[ja]
                yield (i,j), dij
    lp = list(leafPairs()).sort (key=lambda x: x[1])
    # return a unique partition of leaves into pairs with their distances, preferring closer pairs
    available = [True] * nodes
    def cherryPairs():
        for (i,j), dij in lp:
            if available[i] and available[j]:
                available[i] = False
                available[j] = False
                yield (i,j), dij
    return cherryPairs()
