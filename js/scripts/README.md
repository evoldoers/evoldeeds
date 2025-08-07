# Likelihood calculations

To test likelihood calculation:
 
    node js/scripts/calcscore.js data/lg08evol.json data/tree.nh data/tiny.fa

## Cigar trees

The `calcscore.js` script can take a Newick tree and FASTA alignment file, or the sequences can be encoded into a CIGAR tree.

The `makecigar.js` script can prepare a CIGAR tree with or without sequences.
Without sequences is the way you'd want to post it to the server (since the server already has the sequences, and you want to keep the payload small).
With sequences is the way you'd want to feed it to `calcscore.js`.

Creating a CIGAR tree with sequences:

    node js/scripts/makecigar.js data/tree.nh data/tiny.fa

Creating a CIGAR tree without sequences:

    node js/scripts/makecigar.js data/tree.nh data/tiny.fa --noseq

The following should both work:

    node js/scripts/validatecigar.js data/align.cigartree.json
    node js/scripts/validatecigar.js --sequence data/align.seqcigartree.json

The following should both throw a validation error:

    node js/scripts/validatecigar.js --sequence data/align.cigartree.json
    node js/scripts/validatecigar.js data/align.seqcigartree.json

Calculating tree+MSA log-likelihood score using a CIGAR tree with sequences:

    node js/scripts/makecigar.js data/tree.nh data/tiny.fa >data/align.seqcigartree.json
    node js/scripts/calcscore.js data/lg08evol.json --cigartree data/align.seqcigartree.json

Calculating tree+MSA log-likelihood score using a tree file & alignment file directly:

    node js/scripts/calcscore.js data/lg08evol.json data/tree.nh data/align.fa

