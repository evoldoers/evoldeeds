import fs from 'fs';
import { makeCigarTree } from '../cigartree.js';
import Getopt from 'node-getopt';

const scriptName = process.argv[1].split(/[/\\]/).pop();

// Define CLI options and usage
const getopt = new Getopt([
    ['o', 'omitSeqs', 'Omit leaf node sequences from CIGAR tree'],
    ['h', 'help', 'Display this help'],
])
.setHelp(`
Convert a Newick tree and FASTA sequence alignment into a (JSON) CIGAR tree.

Usage: node ${scriptName} [options] tree.nh align.fa

Options:
[[OPTIONS]]
`);

// Parse options
const opt = getopt.parse(process.argv.slice(2));

// Show help if requested
if (opt.options.help) {
    getopt.showHelp();
    process.exit(0);
}

// Validate positional arguments
if (opt.argv.length !== 2) {
    console.error('Error: You must provide exactly two input files: tree.nh and align.fa.\n');
    getopt.showHelp();
    process.exit(1);
}

const [treeFilename, alignFilename] = opt.argv;
const omitSeqs = opt.options.omitSeqs ?? true;

// Read input files
const treeStr = fs.readFileSync(treeFilename, 'utf-8');
const alignStr = fs.readFileSync(alignFilename, 'utf-8');

// Build tree
const { cigarTree } = makeCigarTree(treeStr, alignStr, { omitSeqs });

// Output result
console.log(JSON.stringify(cigarTree));
