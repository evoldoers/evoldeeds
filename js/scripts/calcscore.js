
import fs from 'fs';
import { makeCigarTree } from '../cigartree.js';
import { historyScore } from '../likelihood.js';

if (process.argv.length != 5) {
    console.error('Usage: ' + process.argv[1] + ' model.json tree.nh align.fa');
    process.exit(1);
}

const [ modelFilename, treeFilename, alignFilename ] = process.argv.slice(2);
const modelJson = JSON.parse (fs.readFileSync(modelFilename).toString());
const treeStr = fs.readFileSync(treeFilename).toString();
const alignStr = fs.readFileSync(alignFilename).toString();

const { cigarTree, seqByName } = makeCigarTree (treeStr, alignStr, { omitSeqs: true });

const score = historyScore (cigarTree, seqByName, modelJson);

console.log (JSON.stringify({cigarTree, score}));

