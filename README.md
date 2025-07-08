# EvolDeeds

This repository contains JavaScript implementations of
various algorithms for computing the likelihoods of phylogenetic alignments
of proteins, as well as the beginnings of a gamification
framework for crowdsourced phylogenetics.

The underlying probabilistic models for sequence evolution are
[continuous-time Markov chains](https://en.wikipedia.org/wiki/Continuous-time_Markov_chain) for substitutions,
[hidden Markov models](https://web.stanford.edu/class/cs262/archives/notes/lecture8.pdf) for indels,
[Potts models](https://tianyu-lu.github.io/communication/protein/ml/2021/04/09/Potts-Model-Visualized.html) for interactions between amino acids,
and [continuous-time Bayes networks](https://arxiv.org/abs/1301.0591)
(CTBNs) for covariant substitution processes.


## JavaScript

The JavaScript code is in the `js/` subdirectory.
It includes implementations of

- [Felsenstein's algorithm](https://en.wikipedia.org/wiki/Felsenstein%27s_tree-pruning_algorithm) for computing the likelihood of the substitutions in the alignment
- HMM-based algorithms for computing the likelihood of the indels in the alignment

Two different models ([H20](https://academic.oup.com/genetics/article/216/4/1187/6065876) and [KM03](https://pubmed.ncbi.nlm.nih.gov/14529629/)) are implemented for calculating the HMM transition probabilities in terms of the parameters of the underlying indel model.

The JavaScript code also includes a JSON data structure (Cigar Tree) that compactly represents a phylogenetic tree, multiple sequence alignment (MSA), and ancestral sequence reconstruction, using a [CIGAR](https://jef.works/blog/2017/03/28/CIGAR-strings-for-dummies/)-like format.

## Data

The `data/` subdirectory contains a few test alignments and parameters.

## AWS Lambda code

The `aws/` subdirectory contains code implementing a REST API
(using [serverless](https://www.cloudflare.com/learning/serverless/) Amazon Web Services) whereby an admin can set up a sequence dataset,
and users can then post their solutions to the problem of reconstructing the
most likely evolutionary history explaining that dataset,
using the above probabilistic models as a scoring scheme.

## Front-end client

The `frontend-client/` subdirectory contains a stub for a React/Vite
application that will eventually allow users to submit their own
evolutionary histories using the REST API defined in `aws/`.