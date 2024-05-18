
import fs from 'fs';
import { makeCigarTree, expandCigarTree, getHMMSummaries } from './cigartree.js';
import { parseHistorianParams, subLogLike, sum } from './likelihood.js';
import { transitionMatrix, dummyRootTransitionMatrix } from './h20.js';

if (process.argv.length != 5) {
    console.error('Usage: ' + process.argv[1] + ' tree.nh align.fa model.json');
    process.exit(1);
}

const [ treeFilename, alignFilename, modelFilename ] = process.argv.slice(2);
const treeStr = fs.readFileSync(treeFilename).toString();
const alignStr = fs.readFileSync(alignFilename).toString();
const modelJson = JSON.parse (fs.readFileSync(modelFilename).toString());

const ct = makeCigarTree (treeStr, alignStr);
const { leavesByColumn, internalsByColumn, branchesByColumn } = expandCigarTree (ct);

const { alphabet, mixture, indelParams } = parseHistorianParams (modelJson);
const { seqs, distanceToParent, parentIndex, transCounts } = getHMMSummaries (treeStr, alignStr, alphabet);

const { subRate, rootProb } = mixture[0];
const subll = subLogLike (seqs, distanceToParent, leavesByColumn, internalsByColumn, branchesByColumn, alphabet, subRate, rootProb);
console.warn({subll})
const subll_total = sum (subll);

let transMat = [dummyRootTransitionMatrix()].concat (distanceToParent.slice(1).map((t)=>transitionMatrix(t,indelParams,alphabet.length)));
transMat = transMat.map ((mt) => mt.map((mti) => mti.map((mtij) => Math.log (Math.max (mtij, Number.MIN_VALUE)))));
const transll = transCounts.map ((mt,t) => mt.map((mti,i) => mti.map((mtij,j) => mtij * transMat[t][i][j])));
const transll_total = sum (transll.map ((mt) => sum (mt.map(sum))));

console.log (JSON.stringify({'loglike':{'subs':subll_total,'indels':transll_total}, 'cigartree': ct}));

