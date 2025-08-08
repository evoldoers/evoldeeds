# Likelihood calculations

To test likelihood calculation:
 
    node js/scripts/calcscore.js data/lg08evol.json data/tree.nh data/tiny.fa

## Cigar trees

The `calcscore.js` script can take a Newick tree and FASTA alignment file, or the sequences can be encoded into a CIGAR tree.

The `makecigar.js` script can prepare a CIGAR tree several different ways:
1 Without sequences. This is the way you'd want to post it to the server (since the server already has the sequences, and you want to keep the payload small).
1 With sequences, but without parameters. This is ready for `calcscore.js`, except that script also requires that you specify model parameters as well. But you can also combine everything in one file:
1 With sequences, and with parameters. This can be passed directly to `calcscore.js` with no other parameters needed.

Creating a CIGAR tree without sequences:

    node js/scripts/makecigar.js data/tree.nh data/tiny.fa --noseq

Creating a CIGAR tree with sequences, but no parameters:

    node js/scripts/makecigar.js data/tree.nh data/tiny.fa

Creating a CIGAR tree with sequences and parameters:

    node js/scripts/makecigar.js --params data/lg08evol.json data/tree.nh data/tiny.fa

The following should both work:

    node js/scripts/validatecigar.js data/align.cigartree.json
    node js/scripts/validatecigar.js --sequence data/align.seqcigartree.json

The following should both throw a validation error:

    node js/scripts/validatecigar.js --sequence data/align.cigartree.json
    node js/scripts/validatecigar.js data/align.seqcigartree.json

Calculating tree+MSA log-likelihood score using a CIGAR tree with sequences (model parameters specified separately):

    node js/scripts/makecigar.js data/tree.nh data/tiny.fa >data/align.seqcigartree.json
    node js/scripts/calcscore.js data/lg08evol.json --cigartree data/align.seqcigartree.json

Calculating tree+MSA log-likelihood score using a CIGAR tree with sequences (model parameters included):

    node js/scripts/makecigar.js -p data/lg08evol.json data/tree.nh data/tiny.fa >data/paramcigar.json
    node js/scripts/calcscore.js -p data/paramcigar.json

Calculating tree+MSA log-likelihood score using a tree file & alignment file directly:

    node js/scripts/calcscore.js data/lg08evol.json data/tree.nh data/align.fa

