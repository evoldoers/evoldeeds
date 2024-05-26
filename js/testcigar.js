
import fs from 'fs';
import { makeCigarTree, expandCigarTree, countGapSizes } from './cigartree.js';
import { parseHistorianParams, subLogLike, transLogLike, sum } from './likelihood.js';

if (process.argv.length != 5) {
    console.error('Usage: ' + process.argv[1] + ' model.json tree.nh align.fa');
    process.exit(1);
}

const [ modelFilename, treeFilename, alignFilename ] = process.argv.slice(2);
const modelJson = JSON.parse (fs.readFileSync(modelFilename).toString());
const treeStr = fs.readFileSync(treeFilename).toString();
const alignStr = fs.readFileSync(alignFilename).toString();

const ct = makeCigarTree (treeStr, alignStr);
const { alignment, expandedCigar, distanceToParent, leavesByColumn, internalsByColumn, branchesByColumn } = expandCigarTree (ct);
const lcAlignment = alignment.map ((s) => s.toLowerCase());

const { alphabet, mixture, indelParams } = parseHistorianParams (modelJson);
const { transCounts } = countGapSizes (expandedCigar);

const { subRate, rootProb } = mixture[0];
const subll = subLogLike (lcAlignment, distanceToParent, leavesByColumn, internalsByColumn, branchesByColumn, alphabet, rootProb, { subRate });
const subll_total = sum (subll);

const transll = transLogLike (transCounts, distanceToParent, { indelParams, alphabet });
const transll_total = sum (transll);

console.log (JSON.stringify({'loglike':{'subs':subll_total,'indels':transll_total}, 'cigartree': ct}));

