#### Changes:
* [2023..7.20] - Added the following functions:
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
	* Implement `pairwise_partial_correlation` ([Erb 2020](https://www.sciencedirect.com/science/article/pii/S2590197420300082))
	* Implement `compositional_maximum_entropy` ([Weistuch et al. 2022](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-022-05007-z))
	
* **Plotting:** (Optional: `matplotlib` & `seaborn`)
	* Reimplement `plot_compositional` and `plot_prevalence` from [`soothsayer`](github.com/jolespin/soothsayer)
	* Simplex plots 

* **Misc:**
	* Weight components (e.g. gene size)

