# EvolDeeds

This repository contains JavaScript implementations of
various algorithms for computing the likelihoods of phylogenetic alignments
of proteins, as well as the beginnings of a gamification
framework for crowdsourced phylogenetics.

- Admins post domains: curated sets of amino acid sequences, representing protein domain families.
- Players claim the _deed_ for a family by posting the most likely evolutionary explanation for it.
An evolutionary explanation here means a full phylogenetic tree and multiple sequence alignment, with ancestors included in the alignment as wildcard characters.

Histories are scored using a consistent stochastic model for molecular evolution.
The underlying probabilistic models are
[continuous-time Markov chains](https://en.wikipedia.org/wiki/Continuous-time_Markov_chain) for substitutions and
[hidden Markov models](https://web.stanford.edu/class/cs262/archives/notes/lecture8.pdf)
(specifically the [Thorne, Kishino & Felsenstein (1992) model](https://pubmed.ncbi.nlm.nih.gov/1556741/)) for indels.


## JavaScript

The JavaScript code is in the `js/` subdirectory.
It includes implementations of

- [Felsenstein's algorithm](https://en.wikipedia.org/wiki/Felsenstein%27s_tree-pruning_algorithm) for computing the likelihood of the substitutions in the alignment
- HMM-based algorithms for computing the likelihood of the indels in the alignment

Two different models ([H20](https://academic.oup.com/genetics/article/216/4/1187/6065876) and [KM03](https://pubmed.ncbi.nlm.nih.gov/14529629/)) are implemented for calculating the HMM transition probabilities in terms of the parameters of the underlying indel model.

The JavaScript code also includes a JSON data structure (Cigar Tree) that compactly represents a phylogenetic tree, multiple sequence alignment (MSA), and ancestral sequence reconstruction, using a [CIGAR](https://jef.works/blog/2017/03/28/CIGAR-strings-for-dummies/)-like format.

## Data

The `data/` subdirectory contains a few test alignments and parameters.

## Scripts

There are JavaScript and Python scripts in several places:
- `aws/scripts` ... scripts for interacting with the back-end
- `data/scripts` ... data download and preprocessing scripts for working with TreeFam etc.
- `js/scripts` ... scripts for working with trees, alignments, and the Cigar Tree format

Try e.g. `node js/scripts/calcscore.js data/lg08evol.json data/gp120.nh data/gp120.aligned.fa` to compute the score of a reconstruction of HIV's gp120 envelope domain.

## AWS Lambda code

The `aws/` subdirectory contains code implementing a REST API
(using [serverless](https://www.cloudflare.com/learning/serverless/) Amazon Web Services) whereby an admin can set up a sequence dataset,
and users can then post their solutions to the problem of reconstructing the
most likely evolutionary history explaining that dataset,
using the above probabilistic models as a scoring scheme.

## How to work with the AWS database (site administrators)

- Go to `https://api.evoldeeds.com/families` to get a list of families.
- Go to `https://api.evoldeeds.com/families/{family_id}` to get the info for a particular family (e.g. https://api.evoldeeds.com/families/test). Info includes sequences, current best score, and `created` date for current best-scoring history.
- Go to `https://api.evoldeeds.com/histories/{family_id}/{created}` to get a particular history (e.g. https://api.evoldeeds.com/histories/test/1716233572356)
- Use the node scripts in `aws/scripts` to create, delete, or post histories for a family. You will need to set the `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` environment variables as described [here](https://docs.aws.amazon.com/cli/v1/userguide/cli-configure-envvars.html). Ask Ian for these keys


## Front-end client

The `frontend-client/` subdirectory contains a stub for a React/Vite
application that will eventually allow users to submit their own
evolutionary histories using the REST API defined in `aws/`.