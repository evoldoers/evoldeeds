import assert from 'node:assert/strict';
import { parse_newick } from 'biojs-io-newick';

const sum = (arr) => arr.reduce((a,b) => a+b, 0);

// A cigar tree is a recursive structure of nodes. It may contain the following fields:
//  - distance: the distance from this node to its parent (not present for the root node)
//  - cigar: the CIGAR string for the alignment from the parent to this node (root node must contain all inserts)
//  - child: an array of child nodes (not present for leaf nodes)
//  - seq: the sequence at this node (may be omitted under some circumstances, e.g. internal nodes)
//  - id: a unique identifier for this node (may be omitted unless needed)

// Expand a history tree to a multiple alignment and a tree
const cigarRegex = /^(\d+[MID])*$/;
const cigarGroupRegex = /(\d+)([MID])/g;
const canonicalGapChar = '-';
const canonicalWildChar = '*';
const allGapChars = '-.';
export const expandCigarTree = (rootNode, seqById, gap = canonicalGapChar, wildcard = canonicalWildChar) => {
    // Build a list of nodes in preorder
    let nodeList = [], parentIndex = [], childIndex = [], distanceToParent = [], nodeById = {};
    const visitSubtree = (node, pi) => {
        const nodeIndex = nodeList.length;
        if ('id' in node) {
            if (node.id in nodeById)
                throw new Error("Duplicate node ID: " + node.id);
            nodeById[node.id] = node;
        }
        nodeList.push(node);
        parentIndex.push(pi);
        distanceToParent.push(node.distance || 0);
        childIndex.push([]);
        if (node.child)
            childIndex[nodeIndex] = node.child.map((child) => visitSubtree(child,nodeIndex));
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
            if (childIndex[row].length === 0)
                leaves.push(row);
            else {
                branches[row] = childIndex[row].filter ((childRow) => {
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
        assert (alignment.filter((row)=>row.length!==nColumns).length === 0)
        assert (leaves.filter((r)=>alignment[r][-1]===gap).length === 0)
    }
    assert (leavesByColumn.filter((leaves,col)=>leaves.filter((leaf)=>alignment[leaf][col]==='-').length).length===0)
    // verify that all cursors reached the end of the respective expanded CIGAR strings (and sequences, if supplied)
    const badCigarRow = nextCigarPos.findIndex ((pos,row) => pos < expandedCigar[row].length);
    if (badCigarRow >= 0)
        throw new Error("CIGAR string not fully processed in node " + nodeName(badCigarRow) + " (position " + nextCigarPos[badCigarRow] + " of " + expandedCigar[badCigarRow] + ")");
    const badSeqRow = nextSeqPos.findIndex ((pos,row) => typeof(getSequence(nodeList[row]))==='string' && pos < getSequence(nodeList[row]).length);
    if (badSeqRow >= 0)
        throw new Error("Sequence not fully accounted for in node " + nodeName(badCigarRow) + " (position " + nextSeqPos[badSeqRow] + " of " + getSequence(nodeList[badSeqRow]) + ")");
    // return
    return {alignment, gap, wildcard, expandedCigar, nRows, nColumns, leavesByColumn, internalsByColumn, branchesByColumn, rootByColumn, nodeList, parentIndex, distanceToParent, childIndex, nodeById};
};

// Verify that there is a one-to-one mapping between leaf nodes and sequences in a separate sequence dataset.
// Also check that no nodes specify their own sequences.
export const doLeavesMatchSequences = (expandedHistory, seqById) => {
    if (expandedHistory.nodeList.some(node => 'seq' in node)) {
        return false;
    }
    const leafNodes = expandedHistory.nodeList.filter(node => !node.child);
    if (leafNodes.some(node => !('id' in node))) {
        return false;
    }
    const leafIds = new Set(leafNodes.map(node => node.id));
    const seqIds = new Set(Object.keys(seqById));
    const missingSeqs = Array.from(leafIds).filter(id => !seqIds.has(id));
    const missingNodes = Array.from(seqIds).filter(id => !leafIds.has(id));
    return missingSeqs.length === 0 && missingNodes.length === 0;
}

export const countGapSizes = (expandedCigar) => {
    const counts = expandedCigar.map ((excig) => {
        let nInsertions = 0, nDeletions = 0, gapSizeCounts = {};
        let startState = 0;  // S
        let endState = 1;  // M
        const countGapSize = () => {
            const gapSize = startState + ' ' + endState + ' ' + nDeletions + ' ' + nInsertions;
            if (gapSize in gapSizeCounts)
                gapSizeCounts[gapSize]++;
            else
                gapSizeCounts[gapSize] = 1;
        };
        for (let pos = 0; pos < excig.length; pos++) {
            const c = excig[pos];
            if (c === 'I')
                nInsertions++;
            else if (c === 'D')
                nDeletions++;
            else {
                countGapSize();
                nInsertions = nDeletions = 0;
                startState = 1;  // M
            }
        }
        endState = 4;  // E
        countGapSize();
        return gapSizeCounts;
    });
    return counts;
};

const parseNewick = (newickStr) => {
    const tree = parse_newick(newickStr);
    let nodeName = [], parentIndex = [], distanceToParent = [];
    const visitNode = (node, p) => {
        const n = nodeName.length;
        nodeName.push (node.name || undefined);
        parentIndex.push (p);
        distanceToParent.push (node.length || node.branch_length || 0);
        node.children?.forEach ((child) => visitNode(child,n));
    };
    visitNode (tree, -1);
    return { parentIndex, distanceToParent, nodeName };
};

const defaultParseFastaOpts = { requireFlush: false, forceLowerCase: false, removeGaps: false, alphabet: undefined };
export const parseFasta = (fastaStr, opts = {}) => {
    opts = { ...defaultParseFastaOpts, ...opts };
    const lines = fastaStr.split('\n');
    let seqByName = {}, seqNames = [], name;
    lines.forEach ((line) => {
        if (line[0] === '>') {
            name = line.substr(1).split(' ')[0];
            seqNames.push(name);
            seqByName[name] = '';
        } else {
            if (opts?.forceLowerCase)
                line = line.toLowerCase();
            line = line.replaceAll(new RegExp(`[${allGapChars}]`,'g'), opts?.removeGaps ? '' : canonicalGapChar);
            seqByName[name] += line;
        }
    });
    if (opts?.requireFlush) {
        const seqLengths = Object.values(seqByName).map((s) => s.length);
        assertSame (seqLengths, "Rows in an alignment are supposed to be the same length, but these sequences have different lengths");
    }
    if (opts?.alphabet) {
        const alphabet = (opts?.forceLowerCase && opts.alphabet?.toLowerCase()) || opts.alphabet;
        Object.keys(seqByName).forEach((id) => {
            const seq = seqByName[id];
            for (const c of seq) {
                if (!alphabet.includes(c)) {
                    throw new Error(`Sequence '${id}' contains character '${c}' not in alphabet '${alphabet}'`);
                }
            }
        });
    }
    return { seqNames, seqByName };
};

const assertSame = (l, error) => {
    if (l.length > 0)
        assert (l.filter((n) => n != l[0]).length === 0, error);
};

const getNumCols = (seqs) => {
    const seqLengths = seqs.filter((s) => typeof(s) !== 'undefined').map((s) => s.length);
    assertSame (seqLengths, "Alignment rows are supposed to be the same length, but are not");
    assert (seqLengths.length > 0, "Must have at least one sequence in alignment");
    return seqLengths[0];
};

const orderAlignmentSeqs = (seqByName, nodeName, gapChar = '-') => {
    let seqs = nodeName.map((n) => (n && seqByName[n]) || undefined);
    const nCols = getNumCols(seqs);
    const gapRow = gapChar.repeat(nCols);
    seqs = seqs.map ((s) => s || gapRow);
    return seqs;
};

const getChildren = (parentIndex) => {
    const nRows = parentIndex.length;
    let children = Array.from({length:nRows}).map(() => []);
    parentIndex.forEach ((p,c) => {
        if (p >= 0)
            children[p].push(c);
    })
    return children;
};

const addPathsToMRCAs = (seqs, parentIndex, gapChar = '-', wildChar = 'x') => {
    const nRows = seqs.length;
    const nCols = getNumCols(seqs);
    const children = getChildren(parentIndex);
    seqs = seqs.map ((s) => s.split(''));
    for (let col = 0; col < nCols; ++col) {
        let nUngappedDescendants = new Array(nRows).fill(0);
        for (let row = nRows-1; row >= 0; --row)
            nUngappedDescendants[row] = (children[row].length === 0
                                         ? (seqs[row][col] === gapChar ? 0 : 1)
                                         : sum(children[row].map((c) => nUngappedDescendants[c])));
        let mrca;
        for (let row = nRows-1; row >= 0; --row)
            if (nUngappedDescendants[row] === nUngappedDescendants[0]) {
                mrca = row;
                break;
            }
        for (let row = mrca; row < nRows; ++row) {
            const isInternal = children[row].length;
            if (isInternal && seqs[row][col] === gapChar && nUngappedDescendants[row] > 0)
                seqs[row][col] = wildChar;
        }
    }
    seqs = seqs.map ((s) => s.join(''));
    return seqs;
};

const getExpandedCigarsFromAlignment = (seqs, parentIndex, gapChar = '-') => {
    assert (seqs.length > 0, "Alignment is empty");
    const nRows = seqs.length;
    const nCols = seqs[0].length;
    const getCigarChar = (parentChar, childChar) => {
        if (parentChar === gapChar) {
            if (childChar === gapChar)
                return undefined;
            return 'I';
        } else {
            if (childChar == gapChar)
                return 'D'
            return 'M'
        }
    };
    const getExpandedCigarString = (parentRow, childRow) => {
        return Array.from({length:nCols},(_,n)=>getCigarChar(parentRow[n],childRow[n])).filter((c)=>c).join('');
    };
    return Array.from({length:nRows},(_,r)=> getExpandedCigarString (r > 0 ? seqs[parentIndex[r]] : gapChar.repeat(nCols),seqs[r]));
};

const compressCigarString = (stateStr) => {
    let s, n = 0, cigar = [];
    stateStr.split('').forEach ((c) => {
        if (s != c) {
            if (s)
                cigar.push (String(n) + s);
            s = c;
            n = 0;
        }
        n = n + 1;
    });
    if (s)
        cigar.push (String(n) + s);
    return cigar.join('');
};

const defaultMakeCigarTreeOpts = { forceLowerCase: true, gapChar: '-', omitSeqs: false };
export const makeCigarTree = (newickStr, fastaStr, opts = {}) => {
    const { forceLowerCase, gapChar, omitSeqs } = { ...defaultMakeCigarTreeOpts, ...opts };
    const { parentIndex, distanceToParent, nodeName } = parseNewick (newickStr);
    const { seqByName: gappedSeqByName } = parseFasta (fastaStr, { requireFlush: true, forceLowerCase });
    const seqByName = {};
    Object.keys(gappedSeqByName).forEach ((id) => {
        seqByName[id] = gappedSeqByName[id].replaceAll(gapChar,'');
    });
    let seqs = orderAlignmentSeqs (gappedSeqByName, nodeName);
    seqs = addPathsToMRCAs (seqs, parentIndex, gapChar);
    const expandedCigars = getExpandedCigarsFromAlignment (seqs, parentIndex, gapChar);
    const cigars = expandedCigars.map (compressCigarString);
    const children = getChildren (parentIndex);
    const makeNode = (n) => {
        let node = {};
        const id = nodeName[n];
        if (id)
            node['id'] = id;
        if (n > 0)
            node['distance'] = distanceToParent[n];
        node['cigar'] = cigars[n];
        if (!omitSeqs && id && id in gappedSeqByName)
            node['seq'] = seqByName[id];
        if (children[n].length > 0)
            node['child'] = children[n].map(makeNode);
        return node;
    };
    const cigarTree = makeNode(0);
    return { cigarTree, seqByName };
};

export const getSeqByName = (cigarTree) => {
    const seqByName = {};
    const traverse = (node) => {
        if ('id' in node && 'seq' in node) {
            seqByName[node.id] = node.seq;
        }
        if (node.child) {
            node.child.forEach(traverse);
        }
    };
    traverse(cigarTree);
    return seqByName;
};
