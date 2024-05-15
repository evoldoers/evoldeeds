# Compute substitution log-likelihood of a multiple sequence alignment and phylogenetic tree
# Parameters:
#  - alignment: (L,N) integer tokens. L is the length of the alignment, N is the number of sequences. A token of -1 indicates a gap.
#  - distanceToParent: vector of N floats, distance to parent node
#  - childIndex: (N,2) integers, index of left and right child node. Nodes are sorted in preorder so childIndex[i,j] > i for all i,j