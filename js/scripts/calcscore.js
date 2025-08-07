
import fs from 'fs';
import path from 'path';
import { makeCigarTree } from '../cigartree.js';
import { historyScore } from '../likelihood.js';
import Getopt from 'node-getopt';

const getopt = new Getopt([
    ['c', 'cigartree=FILE', 'Specify a CIGAR tree JSON file, instead of tree and alignment files'],
    ['h', 'help', 'display this help'],
]).bindHelp();

getopt.setHelp(`\nUsage: ${path.basename(process.argv[1])} [options] model.json tree.nh align.fa\n\nOptions:\n[[OPTIONS]]\n`);

const opt = getopt.parse(process.argv.slice(2));
if (opt.argv.length != (opt.options.cigartree ? 1 : 3)) {
    getopt.showHelp();
    process.exit(1);
}

let score = undefined;
let cigarTree = undefined;

if (opt.options.cigartree) {

    const cigarTreeFile = opt.options.cigartree;
    const modelFilename = opt.argv[0];
    if (!fs.existsSync(cigarTreeFile)) {
        console.error(`CIGAR tree file ${cigarTreeFile} does not exist.`);
        process.exit(1);
    }
    if (!fs.existsSync(modelFilename)) {
        console.error(`Model file ${modelFilename} does not exist.`);
        process.exit(1);
    }

    const cigarTreeJson = JSON.parse(fs.readFileSync(cigarTreeFile, 'utf8'));
    const modelJson = JSON.parse(fs.readFileSync(modelFilename, 'utf8'));

    cigarTree = cigarTreeJson;
    score = historyScore(cigarTree, null, modelJson);
    
} else {
    const [ modelFilename, treeFilename, alignFilename ] = process.argv.slice(2);

    if (!fs.existsSync(modelFilename)) {
        console.error(`Model file ${modelFilename} does not exist.`);
        process.exit(1);
    }
    if (!fs.existsSync(treeFilename)) {
        console.error(`Tree file ${treeFilename} does not exist.`);
        process.exit(1);
    }
    if (!fs.existsSync(alignFilename)) {
        console.error(`Alignment file ${alignFilename} does not exist.`);
        process.exit(1);
    }

    const modelJson = JSON.parse (fs.readFileSync(modelFilename).toString());
    const treeStr = fs.readFileSync(treeFilename).toString();
    const alignStr = fs.readFileSync(alignFilename).toString();

    const ct = makeCigarTree (treeStr, alignStr, { omitSeqs: true });

    cigarTree = ct.cigarTree;
    score = historyScore (ct.cigarTree, ct.seqByName, modelJson);
}

console.log (JSON.stringify({cigarTree, score}));

