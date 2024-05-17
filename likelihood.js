import math from 'mathjs';

// Binomial coefficient using gamma functions
const log_binom = (x, y) => lgamma(x+1) - lgamma(y+1) - lgamma(x-y+1);

const sum = (arr) => arr.reduce((a,b) => a+b, 0);

const logsumexp = (arr) => arr.reduce((a,b) => Math.min(a,b) + Math.log(1 + Math.exp(Math.abs(a-b))));

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
const subLogLike = (alignment, distanceToParent, leavesByColumn, internalsByColumn, branchesByColumn, alphabet, subRate, rootProb) => {
    const subRateMatrix = math.matrix(subRate);
    const branchLogProbMatrix = distanceToParent.map (d => math.map(math.expm(subRateMatrix.multiply(-d)), Math.log));
    const rootLogProb = math.map (math.vector(rootProb), Math.log);
    const nColumns = leavesByColumn.length;
    const nRows = alignment.length;
    const nTokens = alphabet.length;
    const tokenIndices = Array.from({length:nTokens}, (_,i) => i);
    let logF = Array.from({length:nRows}, () => Array.from({length:nTokens}, () => -Infinity));
    const colLogProb = Array.from({length:nColumns}, (_,col) => {
        const internals = internalsByColumn[col];
        if (internals.length == 0)
            return 0;
        const root = internals[0];
        const branches = branchesByColumn[col];
        const leaves = leavesByColumn[col];
        leaves.forEach((leaf) => {
            const char = alignment[leaf][col];
            const token = alphabet.indexOf(char);
            tokenIndices.forEach((i) => { logF[leaf][col][i] = i === token ? 0 : -Infinity; });
        });
        internals.forEach((node) => {
            tokenIndices.forEach((i) => {
                logF[node][i] = sum (branches[node].map((child) => logsumexp (tokenIndices.map((j) => branchLogProbMatrix[child][i][j] + logF[child][j]))));
            });
        });
        return logsumexp (tokenIndices.map((i) => rootLogProb[i] + logF[root][i]));
    });
    return sum(colLogProb);
};