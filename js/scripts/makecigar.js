import fs from 'fs';
import path from 'path';
import { makeCigarTree } from '../cigartree.js';
import Getopt from 'node-getopt';

const getopt = new Getopt([
    ['o', 'omitSeqs', 'Omit leaf node sequences from output'],
]);

getopt.setHelp(`\nUsage: ${path.basename(process.argv[1])} [options] tree.nh align.fa\n\nOptions:\n[[OPTIONS]]\n`);

const opt = getopt.parse(process.argv.slice(2));
if (opt.argv.length != 2) {
    getopt.showHelp();
    process.exit(1);
}

const omitSeqs = !!opt.options.omitSeqs;
const [treeFilename, alignFilename] = opt.argv;

const treeStr = fs.readFileSync(treeFilename).toString();
const alignStr = fs.readFileSync(alignFilename).toString();

const { cigarTree, seqByName } = makeCigarTree(treeStr, alignStr, { omitSeqs });

console.log(JSON.stringify(cigarTree));

