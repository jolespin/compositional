### compositional
Compositional data analysis in Python.

This package is meant to extend the methods of [scikit-bio](http://scikit-bio.org/docs/latest/generated/skbio.stats.composition.html#module-skbio.stats.composition) and serve as a pythonic alternative to [propr](https://github.com/tpq/propr).  For standard compositional data analysis methods, please use `scikit-bio`. 

#### Dependencies:
Compatible for Python 2 and 3.

	pandas
	numpy

#### Install:
```
# "Stable" release (still developmental)
pip install compositional
# Current release
pip install git+https://github.com/jolespin/compositional
```

#### Adapted from the following source:
[propr: An R package to calculate proportionality between vectors of compositional data
 (Thomm Quinn)](https://github.com/tpq/propr)

 
#### Citations (Code):
   
   * Quinn T, Richardson MF, Lovell D, Crowley T (2017) propr: An
   R-package for Identifying Proportionally Abundant Features Using
   Compositional Data Analysis. Scientific Reports 7(16252):
   doi:10.1038/s41598-017-16520-0

   * Espinoza JL. compositional: Compositional data analysis in Python (2020). 
   https://github.com/jolespin/compositional
   
#### Citations (Theory):
   * Quinn TP, Erb I, Gloor G, Notredame C, Richardson MF, Crowley TM
   (2019) A field guide for the compositional analysis of any-omics
   data. GigaScience 8(9). doi:10.1093/gigascience/giz107
 
   * Quinn T, Erb I, Richardson MF, Crowley T (2018) Understanding
   sequencing data as compositions: an outlook and review.
   Bioinformatics 34(16): doi:10.1093/bioinformatics/bty175
 
   * Erb I, Quinn T, Lovell D, Notredame C (2017) Differential
   Proportionality - A Normalization-Free Approach To Differential
   Gene Expression. Proceedings of CoDaWork 2017, The 7th
   Compositional Data Analysis Workshop; available under bioRxiv
   134536: doi:10.1101/134536
 
 
   * Erb I, Notredame C (2016) How should we measure proportionality
   on relative gene expression data? Theory in Biosciences 135(1):
   doi:10.1007/s12064-015-0220-8
 
   * Lovell D, Pawlowsky-Glahn V, Egozcue JJ, Marguerat S, Bahler J
   (2015) Proportionality: A Valid Alternative to Correlation for
   Relative Data. PLoS Computational Biology 11(3):
   doi:10.1371/journal.pcbi.1004075
   
#### Usage:
```python
import compositional as coda
import pandas as pd

# Load abundances (Gomez and Espinoza et al. 2017)
X = pd.read_csv("https://github.com/jolespin/supragingival_plaque_microbiome/blob/master/16S_amplicons/Data/X.tsv.gz?raw=true", 
                sep="\t",
                index_col=0,
                compression="gzip",
)
# Add pseudocount
delta = 1/X.shape[1]**2 # http://scikit-bio.org/docs/latest/generated/skbio.stats.composition.multiplicative_replacement.html
X = X + delta
# print("X.shape: (n={} samples, m={} OTUs)| delta={}".format(*X.shape, delta))
# X.shape: (n=473 samples, m=481 OTUs) | delta=4.322249644494967e-06

# Pairwise variance log-ratio
vlr = coda.pairwise_vlr(X)
# print(vlr.iloc[:4,:4])
#            Otu000514  Otu000001  Otu000038  Otu000003
# Otu000514   0.000000   2.158950   3.763323   3.453961
# Otu000001   2.158950   0.000000   1.821676   1.233443
# Otu000038   3.763323   1.821676   0.000000   2.870469
# Otu000003   3.453961   1.233443   2.870469   0.000000

# Pairwise rho from Erb et al. 2016
rhos = coda.pairwise_rho(X)
# print(rhos.iloc[:4,:4])
#            Otu000514  Otu000001  Otu000038  Otu000003
# Otu000514   1.000000   0.469205   0.168476   0.207426
# Otu000001   0.469205   1.000000   0.267368   0.468015
# Otu000038   0.168476   0.267368   1.000000  -0.033662
# Otu000003   0.207426   0.468015  -0.033662   1.000000
```