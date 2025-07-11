import * as math from 'mathjs';
import { calcTkf92EqmProbs, tkf92RootTransitionMatrix, tkf92BranchTransitionMatrix } from './tkf92.js';
import { expandCigarTree, countGapSizes, doLeavesMatchSequences } from './cigartree.js';
import { logHypergeom2F1, logSumExp } from './mathutil.js';

// Binomial coefficient using gamma functions
const log_binom = (x, y) => math.lgamma(x+1) - math.lgamma(y+1) - math.lgamma(x-y+1);

export const sum = (arr) => arr.reduce((a,b) => a+b, 0);

export const logsumexp = (arr) => arr.filter((x)=>x>-Infinity).reduce((a,b) => Math.max(a,b) + Math.log(1 + Math.exp(-Math.abs(a-b))),-Infinity);

// Convert from expectations (mean eqm sequence length, mean insertion event length, mean lifespan of residue) to GGI params (insertion & deletion rates & extension probs)
// We are making use of the following identities:
//  1/inslen = 1 - insextprob
//  1/seqlen = delextprob/insextprob + 1
//  1/reslife = delrate / (1 - delextprob)
//  insrate * delextprob * (1 - insextprob) = delrate * insextprob * (1 - delextprob)
export const expectationsToGgiParams = (seqlen, inslen, reslife) => {
    const insextprob = 1 - 1 / inslen;  // GGI parameter x
    const delextprob = insextprob * (seqlen + 1) / seqlen;  // GGI parameter y
    const delrate = (1 / reslife) * (1 - delextprob);  // GGI parameter mu_0
    const insrate = delrate * insextprob * (1 - delextprob) / (delextprob * (1 - insextprob));   // GGI parameter lambda_0
    return { insrate, delrate, insextprob, delextprob };
};


// Returns the (log of the) probability of seeing a particular size of gap,
// by combinatorially summing over all paths with the right number of I's and D's.
// Internal gaps: startState = endState = 1 (Match)
// Gaps at the start: startState = 0 (Start), endState = 1 (Match)
// Gaps at the end: startState = 1 (Match), endState = 4 (End)
// No aligned positions: startState = 0 (Start), endState = 4 (End)
const logGapProb = (nDeletions, nInsertions, transmat, startState, endState) => {
  const insState = 2, delState = 3;
  const rowIndices = [startState, insState, delState];
  const colIndices = [endState, insState, delState];
  const [[a,b,c],[f,g,h],[p,q,r]] = rowIndices.map ((i) => colIndices.map ((j) => transmat[i][j]));
  const z = h*q/(g*r);

  const log = Math.log;

  if (nDeletions === 0) {
    return nInsertions === 0 ? log(a) : log(b*f) + (nInsertions-1) * log(g);
  } else if (nInsertions === 0)
    return log(c*p) + (nDeletions-1) * log(r);

  return (nInsertions-1) * log(g) + (nDeletions-1) * log(r) + logSumExp(
              [(nInsertions > 1 ? logHypergeom2F1(1-nDeletions,2-nInsertions,2,z) + log(z*(nInsertions-1)*b*r*f) : -Infinity),
               (logHypergeom2F1(1-nDeletions,1-nInsertions,1,z) + log (b*h*p + c*q*f)),
               (nDeletions > 1 ? logHypergeom2F1(2-nDeletions,1-nInsertions,2,z) + log(z*(nDeletions-1)*c*g*p) : -Infinity)]);
};

// Compute the indel log-likelihood for a multiple sequence alignment
export const logBranchAlignProbs = (gapSizeCounts, distanceToParent, ggiParams) => {
    const branchTransMatrices = makeBranchTransMatrices (distanceToParent, ggiParams);
    const loglike = branchTransMatrices.map ((transmat, i) => {
        const gapSizeCountForNode = gapSizeCounts[i] || {};
        return Object.keys(gapSizeCountForNode).reduce((sum, size, _i) => {
            const [startState, endState, nDeletions, nInsertions] = size.split(' ').map(Number);
            const n = gapSizeCountForNode[size], lp = logGapProb(nDeletions, nInsertions, transmat, startState, endState);
            const g = n * lp;
            return sum + g;
        }, 0);
    });
    return loglike;
};

// Compute the substitution log-likelihood for a multiple sequence alignment
// Input parameters:
//  alignment: array of strings, each string representing a gapped sequence
//  distanceToParent: array of distances to parent for each node
//  leavesByColumn: array of leaf indices for each column
//  internalsByColumn: array of internal node indices for each column, sorted in postorder
//  branchesByColumn: column-indexed array of node-indexed arrays of active (ungapped) child node indices
//  alphabet: array of characters representing the alphabet
//  subRate: substitution rate matrix
//  rootProb: root frequencies (typically equilibrium frequencies for substitution rate matrix)
export const subLogLike = (alignment, distanceToParent, leavesByColumn, internalsByColumn, branchesByColumn, alphabet, rootProb, branchProbMatrixConfig, gapChar = '-') => {
    const branchProbMatrix = makeBranchSubstMatrices (distanceToParent, branchProbMatrixConfig);
    const branchLogProbMatrix = branchProbMatrix.map ((m) => math.map (m, (x) => Math.log(Math.max(Number.MIN_VALUE,x))));
    const rootLogProb = math.map (math.matrix(rootProb), Math.log);
    const nColumns = leavesByColumn.length;
    const nRows = alignment.length;
    const nTokens = alphabet.length;
    const tokenIndices = Array.from({length:nTokens}, (_,i) => i);
    let logF = Array.from({length:nRows}, () => Array.from({length:nTokens}, () => -Infinity));
    const colLogProb = Array.from({length:nColumns}, (_,col) => {
        const internals = internalsByColumn[col];
        const leaves = leavesByColumn[col];
        const branches = branchesByColumn[col];
        const root = internals.length > 0 ? internals[internals.length-1] : leaves[0];
        leaves.forEach((leaf) => {
            const char = alignment[leaf][col];
            if (char === gapChar) throw new Error ("unexpected gap at row " + leaf + " col " + col + ': ' + alignment)
            const token = alphabet.indexOf(char);
            tokenIndices.forEach((i) => { logF[leaf][i] = i === token ? 0 : -Infinity; });
        });
        internals.forEach((node) => {
            tokenIndices.forEach((i) => {
                logF[node][i] = sum (branches[node].map((child) => logsumexp (tokenIndices.map((j) => branchLogProbMatrix[child].get([i,j]) + logF[child][j]))));
            });
        });
        const clp_lse = tokenIndices.map((i) => rootLogProb.get([i]) + logF[root][i]);
        const clp = logsumexp (clp_lse);
        return clp;
    });
    return colLogProb;
};

const makeBranchSubstMatrices = (distanceToParent, branchProbMatrixConfig) => {
    let branchProbMatrix;
    if ('evals' in branchProbMatrixConfig) {
        const { evecs_l, evals, evecs_r } = branchProbMatrixConfig;
        const evecsLMatrix = math.matrix (evecs_l);
        const evalsVector = math.matrix (evals);
        const evecsRMatrix = math.matrix (evecs_r);
        branchProbMatrix = distanceToParent.map (d => math.multiply(evecsRMatrix,
                                                                    math.diag (math.map (math.multiply (evalsVector, d), math.exp)),
                                                                    evecsLMatrix));
    } else if ('subRate' in branchProbMatrixConfig) {
        const { subRate } = branchProbMatrixConfig;
        const subRateMatrix = math.matrix(subRate);
        branchProbMatrix = distanceToParent.map (d => math.expm(math.multiply(subRateMatrix,d)));
    }
    return branchProbMatrix;
};

const makeBranchTransMatrices = (distanceToParent, ggiParams) => {
    distanceToParent = distanceToParent.slice(1);
    const branchTransMatrix = distanceToParent.map ((t) => tkf92BranchTransitionMatrix(t,ggiParams));
    return [tkf92RootTransitionMatrix(ggiParams)].concat (branchTransMatrix);
};

export const historyScore = (history, seqById, modelJson) => {
    const expandedHistory = expandCigarTree (history, seqById);
    if (!doLeavesMatchSequences (expandedHistory, seqById))
        throw new Error ("History does not match sequences");

    const { alignment, expandedCigar, distanceToParent, leavesByColumn, internalsByColumn, branchesByColumn } = expandedHistory;
    const gapSizeCounts = countGapSizes (expandedCigar);

    const { alphabet, seqlen, inslen, reslife, evecs_l, evals, evecs_r, root } = modelJson;

    const alphArray = alphabet.split('');
    const subll_by_col = subLogLike (alignment, distanceToParent, leavesByColumn, internalsByColumn, branchesByColumn, alphArray, root, { evecs_l, evals, evecs_r });
    const subll_total = sum (subll_by_col);

    const ggiParams = expectationsToGgiParams (seqlen, inslen, reslife);
    const gapll_by_branch = logBranchAlignProbs (gapSizeCounts, distanceToParent, ggiParams);
    const gapll_total = sum (gapll_by_branch);

    // Subtract null model log likelihood
    const tokens = alphabet.split('');
    let nRes = Object.fromEntries (tokens.map((c) => [c,0])), nStarts = 0, nEnds = 0, nExtends = 0, nEmpties = 0;
    Object.values(seqById).forEach ((seq) => {
        seq.split('').forEach((c) => ++nRes[c]);
        if (seq.length > 0) {
            ++nStarts;
            ++nEnds;
            nExtends += seq.length - 1;
        } else
            ++nEmpties;
    });
    const subll_null = tokens.reduce ((ll, c, n) => ll + nRes[c] * Math.log(root[n]), 0);

    const { kappa, nu } = calcTkf92EqmProbs(ggiParams);
    const gapll_null = nStarts * Math.log(kappa) + nEmpties * Math.log(1 - kappa) + nExtends * Math.log(nu) + nEnds * Math.log(1 - nu);

    const score = subll_total + gapll_total - subll_null - gapll_null;
    
    return score;
};
