#### Changes:
* [2023.8.28]:

	* Implemented `pairwise_partial_correlation_with_basis_shrinkage` using approaches from [Erb et al. 2020](https://www.sciencedirect.com/science/article/pii/S2590197420300082) and [Jin et al. 2022](https://arxiv.org/pdf/2212.00496.pdf)
	* Added `correlation_to_partial_correlation` and `convariance_to_correlation` functions.
	* Added `checks` to `pairwise_aitchison_distance`
	* Added `is_integer_or_proportional` to `check_compositional`
* [2023.8.5]:
	* Implemented `pairwise_aitchison_distance`
	* Reimplemented `plot_compositional` (i.e., `plot_compositions` now) and `plot_prevalence` from [`soothsayer`](github.com/jolespin/soothsayer)
	
* [2023.7.20] - Added the following functions:
	* `assert_acceptable_arguments`
	* `check_compositional`
	* `sparsity`
	* `number_of_components`
	* `prevalence_of_components`
	* `transform_closure`
	* `filter_data_highpass`

* [2022.8.31] - Added support for Python v3.10

#### Future: 
* **Metrics:**

	* Implement `compositional_maximum_entropy` ([Weistuch et al. 2022](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-022-05007-z))
	
* **Plotting:** (Optional: `matplotlib` & `seaborn`)
	* Simplex plots 

* **Misc:**
	* Weight components (e.g. gene size)

