import fs from 'fs';
import { makeCigarTree } from '../cigartree.js';
import Getopt from 'node-getopt';

const getopt = new Getopt([
    ['o', 'omitSeqs', 'Omit leaf node sequences from output'],
]);

const opt = getopt.parse(process.argv.slice(2));

if (opt.argv.length != 2) {
    console.error('Usage: ' + process.argv[1] + ' [options] tree.nh align.fa');
    getopt.showHelp();
    process.exit(1);
}

const omitSeqs = opt.options.omitSeqs ?? true;
const [treeFilename, alignFilename] = opt.argv;

const treeStr = fs.readFileSync(treeFilename).toString();
const alignStr = fs.readFileSync(alignFilename).toString();

const { cigarTree, seqByName } = makeCigarTree(treeStr, alignStr, { omitSeqs });

console.log(JSON.stringify(cigarTree));

