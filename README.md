### compositional
Compositional data analysis in Python.

This package is meant to extend the methods of [scikit-bio](http://scikit-bio.org/docs/latest/generated/skbio.stats.composition.html#module-skbio.stats.composition) and serve as a pythonic alternative (not replacement) to *some* functionalities within [propr](https://github.com/tpq/propr).  Please use `scikit-bio` for standard compositional data analysis or `propr` for robust statistical testing.  

#### Dependencies:
Compatible for Python 2 and 3.

```
# Required:
pandas
numpy

# Optional:
scikit-bio
gneiss
ete[2/3]
```	
	

		

#### Install:
```
# "Stable" release (still developmental)
pip install compositional
# Current release
pip install git+https://github.com/jolespin/compositional
```

#### Proportionality methods adapted from the following source:
* [propr: An R package to calculate proportionality between vectors of compositional data
 (Thomm Quinn)](https://github.com/tpq/propr)
 
#### Isometric log-ratio methods use the following sources:
* [scikit-bio: A package providing data structures, algorithms and educational resources for bioinformatics](https://github.com/biocore/scikit-bio)
* [gneiss: a compositional data analysis toolbox designed for analyzing high dimensional proportions (Jamie Morton)](https://github.com/biocore/gneiss)

 
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
   
   * Morton, J.T., Sanders, J., Quinn, R.A., McDonald, D., Gonzalez, A., Vázquez‐Baeza, Y., et al . (2017) Balance trees reveal microbial niche differentiation. mSystems: e00162‐16. doi: 10.1128/mSystems.00162-16


#### Citations (Debut):
   
   * Espinoza JL., Shah N, Singh S, Nelson KE., Dupont CL. Applications of weighted association networks applied to compositional data in biology. https://doi.org/10.1111/1462-2920.15091
_________________________
### Usage:

#### Loading package and obtaining data
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
```

#### Pairwise operations
```
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
# 				Otu000514  Otu000001  Otu000038  Otu000003
# Otu000514   1.000000   0.470328   0.170234   0.209101
# Otu000001   0.470328   1.000000   0.268917   0.469140
# Otu000038   0.170234   0.268917   1.000000  -0.031477
# Otu000003   0.209101   0.469140  -0.031477   1.000000
```

#### Isometric log-ratio transform *without* tree (requires scikit-bio)
```
# Isometric log-ratio
X_ilr_without_tree = coda.transform_ilr(X)
# print(X_ilr_without_tree.iloc[:4,:4])
# S-1409-45.B_RD1 -2.671007  -0.142743 -1.101510  17.067981
# 1104.2_RD1      -2.122899  13.870926 -8.158016  -0.250970
# S-1409-42.B_RD1 -1.914182  -0.025238 -0.019451  16.660011
# 1073.1_RD1      -1.884611   2.345849 -2.729035   2.448122
```

#### Isometric log-ratio transform *with* tree (requires scikit-bio, gneiss, and [Optional: ete[2/3])
```
import requests
from io import StringIO
from skbio import TreeNode

# Get newick tree
url = "https://github.com/jolespin/supragingival_plaque_microbiome/blob/master/16S_amplicons/Data/otus.alignment.fasttree.nw?raw=true"
newick = requests.get(url).text
tree = TreeNode.read(StringIO(newick), convert_underscores=False)
tree.bifurcate()

# Name internal nodes
intermediate_node_index = 1
for node in tree.traverse():
    if not node.is_tip():
        node.name = "y{}".format(intermediate_node_index)
        intermediate_node_index += 1

# Isometric log-ratio transform
X_ilr_with_tree = coda.transform_ilr(X, tree)
# print(X_ilr_with_tree.iloc[:4,:4])
#                        y1         y2          y480            y3
# S-1409-45.B_RD1 -5.022283   7.999347 -1.526686e-16  3.387754e-15
# 1104.2_RD1      -3.676812   5.856319 -1.415420e-17  2.272686e-15
# S-1409-42.B_RD1 -6.413369  10.215028  9.148268e-17  3.511751e-15
# 1073.1_RD1      -4.608491   7.340271  2.027126e-16  2.519377e-15
```

#### Notes:
* Versions prior to v2020.12.16 used `ddof=0` for all variance except during the `vlr` calculation.  This was because `pandas._libs.algos.nancorr` uses `ddof=1` and not `ddof=0`.  This caused specific `rho` values not to be bound by [-1,1].  To retain the performance of `nancorr`, I've set all `ddof=1` to match `nancorr`. 
