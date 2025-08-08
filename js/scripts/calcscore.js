
import fs from 'fs';
import path from 'path';
import { makeCigarTree } from '../cigartree.js';
import { historyScore } from '../likelihood.js';
import Getopt from 'node-getopt';

const getopt = new Getopt([
    ['c', 'cigartree', 'Specify a CIGAR tree JSON file, instead of tree and alignment files'],
    ['p', 'paramtree', 'Specify a single JSON file containing cigartree and params'],
    ['h', 'help', 'display this help'],
]).bindHelp();

const prog = path.basename(process.argv[1]);
getopt.setHelp(`\nUsage:\n ${prog} [options] model.json tree.nh align.fa\n ${prog} --cigartree model.json cigartree.json\n cat cigartree-with-params.json | ${prog} --paramtree\n\nOptions:\n[[OPTIONS]]\n`);

const opt = getopt.parse(process.argv.slice(2));

if (opt.options.cigartree && opt.options.paramtree) {
    console.error('Cannot use both --cigartree and --paramtree options together');
    process.exit(1);
}

function expectArgs(n, underTheseCircumstances) {
    if (opt.argv.length !== n) {
        console.error(`Expected ${n} arguments ${underTheseCircumstances}, but got ${opt.argv.length}`);
        getopt.showHelp();
        process.exit(1);
    }
}

let score = undefined;
let cigarTree = undefined;

if (opt.options.paramtree) {

    expectArgs(0, "with --paramtree option");
    const parsed = JSON.parse(fs.readFileSync(process.stdin.fd, 'utf8'));
    cigarTree = parsed.cigartree;

    score = historyScore(cigarTree, null, parsed.params);

} else if (opt.options.cigartree) {

    expectArgs(1, "with --cigartree option");
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
    expectArgs(3, "when neither --cigartree nor --paramtree is specified");
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

console.log (score);

