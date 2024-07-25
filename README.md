# EvolDeeds

This repository contains Python and JavaScript implementations of
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

## Python

The `python/` subdirectory of the repo contains considerably more in the way of algorithms, though the basics should be compatible with the JavaScript code described above.

The Python code is implemented using [Jax](https://github.com/google/jax),
making it suitable for model-fitting (which should be accelerated if using GPUs).

In addition to the models and data structures described above in the JavaScript section, the Python codebase includes
- an implementation of the [CherryML](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10644697/) approach to fitting substitution rate matrices
- several variations of CherryML and combinations with EM-like algorithms for fitting mixtures of substitution models
- an implementation of a [variational algorithm](https://www.jmlr.org/papers/v11/cohn10a.html) for CTBN Potts models, for computing alignment substitution likelihoods where there are interactions between amino acids (i.e. because they are in physical contact in the folded 3D structure)

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