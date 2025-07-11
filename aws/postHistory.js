import fs from 'fs';
import { makeCigarTree } from '../cigartree.js';

if (process.argv.length < 5) {
    console.error('Usage: ' + process.argv[1] + ' family_id tree.nh align.fa [player]');
    process.exit(1);
}

const url = 'https://api.evoldeeds.com/histories/';

const [ familyId, treeFilename, alignFilename, player ] = process.argv.slice(2);
const treeStr = fs.readFileSync(treeFilename).toString();
const alignStr = fs.readFileSync(alignFilename).toString();

const { cigarTree } = makeCigarTree (treeStr, alignStr, { forceLowerCase: true, omitSeqs: true });
console.warn(JSON.stringify(cigarTree));

const post = async (id, history) => {
    console.warn ("Posting to " + url + id);
    const response = await fetch(url + id, {
        method: "POST",
        // mode: "cors",
        // credentials: "same-origin", // include, *same-origin, omit
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ history, player }),
      });
      console.log("Status: " + response.status);
      const result = await response.json();
      console.log (JSON.stringify(result));
};

post (familyId, cigarTree);


