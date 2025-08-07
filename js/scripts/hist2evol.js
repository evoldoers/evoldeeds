// Script to convert Historian-format model parameters to EvolDeeds format.
// Includes pre-diagonalization of substitution model, and specification of indel model parameters in terms of expectations
// (mean sequence length, mean insertion event length, mean lifespan of a residue).

import * as math from 'mathjs';
import fs from 'fs';

if (process.argv.length != 3) {
    console.error('Usage: ' + process.argv[1] + ' historian_model.json');
    process.exit(1);
}

const [ modelFilename ] = process.argv.slice(2);
const histModel = JSON.parse (fs.readFileSync(modelFilename).toString());
const { insrate, delrate, insextprob, delextprob, alphabet, subrate, rootprob } = histModel;

// Calculate GGI model expectations
const inslen = 1 / (1 - insextprob);  // expected insertion length
const seqlen = 1 / (delextprob / insextprob - 1);  // expected sequence length
const reslife = (1 - delextprob) / delrate;  // expected lifespan of a residue

// Get equilibrium residue frequencies as vector
const alphVec = alphabet.split('');
const rootVec = alphVec.map ((i) => rootprob[i] || 0);

// Diagonalize substitution model
// subrate[i][j] is the rate from i to j, where i and j are characters from alphabet.
// We want to convert this into a square matrix, then diagonalize it,
// returning eigenvectors (evals), left (evecs_l) and right (evecs_r) eigenvectors.
alphVec.forEach((i) => alphVec.forEach((j) => {
    if (i != j)
        subrate[i][i] = (subrate[i][i] || 0) - subrate[i][j];
}));
const subrateMatrix = alphVec.map((i) => alphVec.map((j) => subrate[i][j] || 0));
const { values: evals, eigenvectors: evecs } = math.eigs(math.matrix(subrateMatrix));

const evecs_l = math.matrix(evecs.map(e => e.vector));
const evecs_r = math.inv(evecs_l);

// const evals_diag = math.diag(evals);
// const R = math.multiply(evecs_r, evals_diag, evecs_l);
// const R_err = math.subtract (R, subrateMatrix);

// console.warn('expm',math.expm(subrateMatrix).toString());
// console.warn('diag',math.multiply(evecs_r,math.expm(math.diag(evals)),evecs_l).toString());
// process.exit(1);

const evolModel = { alphabet, seqlen, inslen, reslife,
    root: rootVec, evecs_l: evecs_l.toArray(), evecs_r: evecs_r.toArray(), evals: evals.toArray() };

console.log (JSON.stringify (evolModel));

