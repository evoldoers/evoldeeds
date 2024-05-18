// A history tree is a recursive structure of nodes. It may contain the following fields:
//  - distance: the distance from this node to its parent (not present for the root node)
//  - cigar: the CIGAR string for the alignment from the parent to this node (root node must contain all inserts)
//  - child: an array of child nodes (not present for leaf nodes)
//  - seq: the sequence at this node (may be omitted under some circumstances)
//  - id: a unique identifier for this node (may be omitted unless needed)
//  - n: the preorder index of this node (may be omitted, will be calculated as needed)

// Expand a history tree to a multiple alignment and a tree
const cigarRegex = /^(\d+[MID])*$/;
const cigarGroupRegex = /(\d+)([MID])/g;
const expandHistory = (rootNode, seqById, gap, wildcard) => {
    gap = gap || '-';
    wildcard = wildcard || '*';
    // Build a list of nodes in preorder
    let nodeList = [], nodeParentIndex = [], nodeChildIndex = [], distanceToParent = [], nodeById = {};
    const visitSubtree = (node, parentIndex) => {
        const nodeIndex = nodeList.length;
        node.n = nodeIndex;
        if ('id' in node) {
            if (node.id in nodeById)
                throw new Error("Duplicate node ID: " + node.id);
            nodeById[node.id] = node;
        }
        nodeList.push(node);
        nodeParentIndex.push(parentIndex);
        distanceToParent.push(node.distance || 0);
        nodeChildIndex.push([]);
        if (node.child)
            nodeChildIndex[nodeIndex] = node.child.map((child) => visitSubtree(child,nodeIndex));
        return nodeIndex;
    }
    visitSubtree(rootNode,undefined);
    const nRows = nodeList.length;
    // Expand the cigar strings for each node
    const nodeName = (n) => nodeList[n].id || ('#' + n);
    const expandedCigar = nodeList.map((node, n) => {
        if (typeof(node.cigar) !== "string")
            throw new Error("CIGAR string missing from node " + nodeName(n));
        if (!cigarRegex.test(node.cigar))
            throw new Error("Invalid CIGAR string in node " + nodeName(n));
        return node.cigar.replace(cigarGroupRegex, (_g,n,c) => c.repeat(parseInt(n)));
    });
    // Construct the multiple alignment by maintaining a cursor for each node (i.e. row), pointing to the next character in its expanded CIGAR string.
    // At each step, find the highest-numbered row whose next character is an I, then advance its cursor and all its child,
    // recursing on each subtree until a descendant's next character is a D.
    let nextCigarPos = expandedCigar.map(() => 0);
    let nextSeqPos = nodeList.map(node => 0);
    let alignment = nodeList.map(node => '');
    let leavesByColumn = [], internalsByColumn = [], rootByColumn = [], branchesByColumn = [];
    let nColumns = 0;
    const getSequence = (node) => (seqById && 'id' in node) ? seqById[node.id] : node.seq;
    const advanceCursor = (row, leaves, internals, branches, isDelete) => {
        nextCigarPos[row]++;
        if (!isDelete) {
            const node = nodeList[row];
            const sequence = getSequence(node);
            if (typeof(sequence) === 'string') {
                if (nextSeqPos[row] >= sequence.length)
                    throw new Error("Sequence ended prematurely in node " + nodeName(row));
                const nextChar = sequence[nextSeqPos[row]];
                if (nextChar === gap)
                    throw new Error("Gap character found in sequence of node " + nodeName(row));
                alignment[row] += nextChar;
                nextSeqPos[row]++;                
            } else
                alignment[row] += wildcard;
            if (nodeChildIndex[row].length === 0)
                leaves.push(row);
            else {
                branches[row] = nodeChildIndex[row].filter ((childRow) => {
                        if (nextCigarPos[childRow] >= expandedCigar[childRow].length)
                            throw new Error("CIGAR string ended prematurely in node " + nodeName(childRow));
                        const childCigarChar = expandedCigar[childRow][nextCigarPos[childRow]];
                        if (childCigarChar === 'I')
                            throw new Error("Insertion in child node " + nodeName(childRow) + " when match or deletion expected");
                        const childIsDelete = childCigarChar === 'D';
                        advanceCursor (childRow, leaves, internals, branches, childIsDelete);
                        return !childIsDelete;
                    });
                internals.push(row);  // push at end so as to sort in postorder
            }
        }
    };
    while (true) {
        let nextInsertRow = nextCigarPos.length - 1;
        while (nextInsertRow >= 0 && (nextCigarPos[nextInsertRow] >= expandedCigar[nextInsertRow].length || expandedCigar[nextInsertRow][nextCigarPos[nextInsertRow]] !== 'I'))
            nextInsertRow--;
        if (nextInsertRow < 0)
            break;
        let leaves = [], internals = [], branches = Array.from({length: nRows}, () => null);
        advanceCursor(nextInsertRow,leaves,internals,branches,false);
        rootByColumn.push(nextInsertRow);
        internalsByColumn.push(internals);
        leavesByColumn.push(leaves);
        branchesByColumn.push(branches);
        ++nColumns;
        alignment = alignment.map ((row) => row.length < nColumns ? row + gap : row);
    }
    // verify that all cursors reached the end of the respective expanded CIGAR strings (and sequences, if supplied)
    const badCigarRow = nextCigarPos.findIndex ((pos,row) => pos < expandedCigar[row].length);
    if (badCigarRow >= 0)
        throw new Error("CIGAR string not fully processed in node " + nodeName(badCigarRow) + " (position " + nextCigarPos[badCigarRow] + " of " + expandedCigar[badCigarRow] + ")");
    const badSeqRow = nextSeqPos.findIndex ((pos,row) => typeof(getSequence(nodeList[row]))==='string' && pos < getSequence(nodeList[row]).length);
    if (badSeqRow >= 0)
        throw new Error("Sequence not fully processed in node " + nodeName(badCigarRow) + " (position " + nextSeqPos[badSeqRow] + " of " + getSequence(nodeList[badSeqRow]) + ")");
    // return
    return {alignment, gap, wildcard, expandedCigar, nRows, nColumns, leavesByColumn, internalsByColumn, rootByColumn, nodeList, nodeParentIndex, distanceToParent, nodeChildIndex, nodeById};
};

// Verify that there is a one-to-one mapping between leaf nodes and sequences in a separate sequence dataset.
// Also check that no nodes specify their own sequences.
const doLeavesMatchSequences = (expandedHistory, seqById) => {
    if (expandedHistory.nodeList.some(node => 'seq' in node))
        return false;
    const leafNodes = expandedHistory.nodeList.filter(node => !node.child);
    if (leafNodes.some(node => !('id' in node)))
        return false;
    const leafIds = new Set(leafNodes.map(node => node.id));
    const seqIds = new Set(Object.keys(seqById));
    const missingSeqs = Array.from(leafIds).filter(id => !seqIds.has(id));
    const missingNodes = Array.from(seqIds).filter(id => !leafIds.has(id));
    return missingSeqs.length === 0 && missingNodes.length === 0;
}

const countGapSizes = (expandedCigar) => {
    const counts = expandedCigar.map ((excig) => {
        let nInsertions = 0, nDeletions = 0, gapSizeCount = {}, transitionCount = [[0,0,0],[0,0,0],[0,0,0]];
        const countGapSize = () => {
            gapSize = nDeletions + ' ' + nInsertions;
            if (gapSize in gapSizeCount)
                gapSizeCount[gapSize]++;
            else
                gapSizeCount[gapSize] = 1;
        };
        const stateIndex = (c) => 'MID'.indexOf(c);
        let prev = stateIndex('M');
        for (let pos = 0; pos < excig.length; pos++) {
            const c = excig[pos];
            if (c === 'I')
                nInsertions++;
            else if (c === 'D')
                nDeletions++;
            else {
                countGapSize();
                nInsertions = nDeletions = 0;
            }
            const state = stateIndex(c);
            transitionCount[prev][state]++;
            prev = state;
        }
        countGapSize();
        transitionCount[prev][stateIndex('M')]++;
        return { gapSizeCount, transitionCount };
    });
    return Object.fromEntries (['gapSizeCount','transitionCount'].map ((key,i) => [key,counts.map (count => count[key])]));
};

module.exports = { expandHistory, doLeavesMatchSequences, countGapSizes };
