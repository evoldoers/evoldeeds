def pickCherries (parentIndex, distanceToParent):
    # find leaves
    nodes = len(parentIndex)
    nChildren = [0] * nodes
    for parent in parentIndex:
        if parent >= 0:
            nChildren[parent] += 1
    leaves = [i for i,n in enumerate(nChildren) if n == 0]
    # for each leaf, find distance to each ancestor
    distance = [[0] * nodes for _ in range(nodes)]
    for leaf in leaves:
        ancestor = leaf
        d = 0
        while ancestor >= 0:
            distance[leaf][ancestor] = d
            d += distanceToParent[ancestor]
            ancestor = parentIndex[ancestor]
    # for each pair of leaves, find MRCA and thereby distance between leaves
    def leafPairs():
        for ni,i in enumerate(leaves):
            for j in leaves[ni+1:]:
                ia, ja, = i, j
                while ia != ja:
                    if ia > ja:
                        ia = parentIndex[ia]
                    else:
                        ja = parentIndex[ja]
                dij = distance[i][ia] + distance[j][ia]
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
