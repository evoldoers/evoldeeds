import fs from 'fs/promises';
import Getopt from 'node-getopt';
import { validateCigarTree } from '../validator.js';

// Simple FASTA parser
function parseFasta(text) {
  const lines = text.split(/\r?\n/);
  const seqs = {};
  let currentId = null;
  let currentSeq = [];

  for (const line of lines) {
    if (line.startsWith(">")) {
      if (currentId) seqs[currentId] = currentSeq.join("");
      currentId = line.slice(1).trim().split(/\s+/)[0];
      currentSeq = [];
    } else if (currentId) {
      currentSeq.push(line.trim());
    }
  }

  if (currentId) seqs[currentId] = currentSeq.join("");
  return seqs;
}

// Define options
const getopt = new Getopt([
  ['f', 'fasta=FILE', 'FASTA file of sequences to validate against leaf nodes'],
  ['h', 'help', 'display this help']
]).bindHelp();

// Parse arguments
const opt = getopt.parse(process.argv.slice(2));
const [filename] = opt.argv;

if (!filename) {
  getopt.showHelp();
  process.exit(1);
}

try {
  const data = await fs.readFile(filename, 'utf-8');
  const tree = JSON.parse(data);

  let seqById = null;
  if (opt.options.fasta) {
    const fastaText = await fs.readFile(opt.options.fasta, 'utf-8');
    seqById = parseFasta(fastaText);
  }

  const result = validateCigarTree(tree, { throwOnError: false, seqById });

  if (result.valid) {
    console.log("✅ CIGAR tree is valid.");
  } else {
    console.error("❌ CIGAR tree validation failed.");

    if (result.schemaErrors?.length) {
      console.error("Schema errors:");
      for (const e of result.schemaErrors) console.error("  -", e);
    }

    if (result.logicErrors?.length) {
      console.error("Logic errors:");
      for (const e of result.logicErrors) console.error("  -", e);
    }

    if (result.consistencyErrors?.length) {
      console.error("Sequence consistency errors:");
      for (const e of result.consistencyErrors) console.error("  -", e);
    }

    process.exit(2);
  }
} catch (err) {
  console.error("❌ Exception:", err.message);
  const line = (err.stack || "").split("\n")[1]?.trim();
  if (line) console.error("   at", line);
  process.exit(3);
}
