import * as math from 'mathjs';
import { transitionMatrix, dummyRootTransitionMatrix, normalizeTransMatrix, smallTimeTransitionMatrixFromCounts } from './h20.js';
import { tkf92approxTransitionMatrix } from './tkf92.js';
import { expandCigarTree, countGapSizes, doLeavesMatchSequences } from './cigartree.js';

// Binomial coefficient using gamma functions
const log_binom = (x, y) => lgamma(x+1) - lgamma(y+1) - lgamma(x-y+1);

export const sum = (arr) => arr.reduce((a,b) => a+b, 0);

export const logsumexp = (arr) => arr.filter((x)=>x>-Infinity).reduce((a,b) => Math.max(a,b) + Math.log(1 + Math.exp(-Math.abs(a-b))),-Infinity);

// Returns the (log of the) probability of seeing a particular size of gap
const gapProb = (nDeletions, nInsertions, transmat) => {
  const [[a,b,c],[f,g,h],[p,q,r]] = transmat;
  const log = Math.log;
  const Ck = (k) => {
    const logbinom = log_binom(nDeletions>k ? nDeletions-1 : k,k-1) + log_binom(nInsertions>k ? nInsertions-1 : k,k-1);  // guard against out-of-range errors
    const log_arg = b*(nInsertions-k)*(r*f*k + h*p*(nDeletions-k)) + c*(nDeletions-k)*(g*p*k + q*f*(nInsertions-k));  // guard against log(0)
    return Math.exp (k * log(h*q/(g*r)) + logbinom - 2*log(k) + log(log_arg>0 ? log_arg : 1));
  };
  return nDeletions == 0
         ? (nInsertions == 0
             ? log(a)
             : log(b) + (nInsertions - 1)*log(g) + log(f))
         : (nInsertions == 0
             ? log(c) + (nDeletions - 1)*log(r) + log(p)
             : (nDeletions - 1)*log(g) + (nInsertions-1)*log(r)
              + log (b*h*p + c*q*f + sum (Array.from({length:nDeletions+nInsertions}, (_,k) => (k<nInsertions || k<nDeletions) && k<=nInsertions && k<=nDeletions ? Ck(k) : 0))));
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
    const branchProbMatrix = makeBranchProbMatrices (distanceToParent, branchProbMatrixConfig);
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

const makeBranchProbMatrices = (distanceToParent, branchProbMatrixConfig) => {
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

export const transLogLike = (transCounts, distanceToParent, branchTransConfig) => {
    const transMat = makeBranchTransMatrices (distanceToParent, branchTransConfig);
    const logTransMat = transMat.map ((mt) => mt.map((mti) => mti.map((mtij) => Math.log (Math.max (mtij, Number.MIN_VALUE)))));
    const transll = transCounts.map ((mt,t) => mt.map((mti,i) => mti.map((mtij,j) => mtij * logTransMat[t][i][j])));
    return transll.map ((mt) => sum (mt.map(sum)));
};

const makeBranchTransMatrices = (distanceToParent, branchTransConfig) => {
    let branchTransMatrix;
    distanceToParent = distanceToParent.slice(1);
    if ('steps' in branchTransConfig) {
        branchTransMatrix = distanceToParent.map ((t)=> piecewiseBranchTransMatrix(t,branchTransConfig));
    } else if ('poly' in branchTransConfig) {
        const { poly, tmin, tmax } = branchTransConfig;
        branchTransMatrix = distanceToParent.map ((t)=> polyBranchTransMatrix(t,poly,tmin,tmax));
    } else if ('alphabet' in branchTransConfig) {
        const { indelParams, alphabet } = branchTransConfig;
        branchTransMatrix = distanceToParent.map ((t) => transitionMatrix(t,indelParams,alphabet.length));
    } else {
        const { lmxy } = branchTransConfig;
        branchTransMatrix = distanceToParent.map ((t) => tkf92approxTransitionMatrix(t,lmxy));
    }
    return [dummyRootTransitionMatrix()].concat (branchTransMatrix);
};

const piecewiseBranchTransMatrix = (t, branchTransConfig) => {
    const { t1, tmax, steps, lmxy, abuq } = branchTransConfig;
    const logRatioPerStep = Math.log(tmax/t1) / steps;
    const step = 1 + Math.ceil (Math.log(t/t1) / logRatioPerStep);
    let abuq_t;
    if (step > steps)
        abuq_t = abuq[abuq.length - 1];
    else {
        const t_step = t1 * Math.exp ((step - 1) * logRatioPerStep), t_prev = step > 1 ? (t1 * Math.exp ((step - 2) * logRatioPerStep)) : 0;
        const dt_rel = (t - t_prev) / (t_step - t_prev);
        abuq_t = abuq[step].map ((abuq_step_n, n) => abuq[step-1][n] * (1 - dt_rel) + abuq_step_n * dt_rel);
    }
    return smallTimeTransitionMatrixFromCounts (t, lmxy, abuq_t);
};

const polyBranchTransMatrix = (t, coeffs, tmin, tmax) => {
    t = Math.min (tmax, Math.max (tmin, t));
    const deg = coeffs[0][0].length;
    const tPow = Array.from ({length:deg},(_,n)=>t**n);
    const trans = coeffs.map ((coeffs_i) => coeffs_i.map ((coeffs_ij) => coeffs_ij.reduce ((val, coeff, n) => val + coeff*tPow[n], 0)));
    return normalizeTransMatrix(trans);
};

const normalizeSubModel = (subRate, rootProb) => {
    const A = rootProb.length;
    const norm = sum(rootProb);
    for (let i = 0; i < A; ++i) {
        subRate[i][i] -= sum(subRate[i]);
        rootProb[i] /= norm;
    }
    return { subRate, rootProb };
};

export const parseHistorianParams = (params) => {
    const alphabet = params['alphabet'];
    const alphArray = alphabet.split('');
    const parseSubRate = (p) => {
        let sr = alphArray.map ((i) => alphArray.map ((j) => (p['subrate'][i] || {})[j] || 0));
        let rp = alphArray.map ((i) => p['rootprob'][i] || 0);
        const { subRate, rootProb } = normalizeSubModel (sr, rp);
        return { subRate, rootProb };
    };
    const mixture = 'mixture' in params ? params['mixture'].map(parseSubRate): [parseSubRate(params)];
    const indelParams = ['insrate','delrate','insextprob','delextprob'].map ((key) => params[key] || 0);
    return { alphabet, mixture, indelParams };
};

export const historyScore = (history, seqById, modelJson) => {
    const expandedHistory = expandCigarTree (history, seqById);
    if (!doLeavesMatchSequences (expandedHistory, seqById))
        throw new Error ("History does not match sequences");

    const { alignment, expandedCigar, distanceToParent, leavesByColumn, internalsByColumn, branchesByColumn } = expandedHistory;
    const { transCounts } = countGapSizes (expandedCigar);

    const { alphabet, hmm, mixture } = modelJson;

    const { evecs_l, evals, evecs_r, root } = mixture[0];
    const subll = subLogLike (alignment, distanceToParent, leavesByColumn, internalsByColumn, branchesByColumn, alphabet, root, { evecs_l, evals, evecs_r });
    const subll_total = sum (subll);

    const transll = transLogLike (transCounts, distanceToParent, hmm);
    const transll_total = sum (transll);

    const tokens = alphabet.split('');
    let resFreq = Object.fromEntries (tokens.map((c) => [c,0]));
    Object.values(seqById).forEach ((seq) => seq.split('').forEach((c) => ++resFreq[c]));
    const null_total = tokens.reduce ((ll, c, n) => ll + resFreq[c] * Math.log(root[n]), 0);

    const score = subll_total + transll_total - null_total;

    return score;
};
