# -*- coding: utf-8 -*-

__version__= "2023.7.20"
__author__ = "Josh L. Espinoza"
__email__ = "jespinoz@jcvi.org, jol.espinoz@gmail.com"
__url__ = "https://github.com/jolespin/compositional"
__license__ = "BSD-3"
__developmental__ = True

# =======
# Direct Exports
# =======
__functions__ = [
    # Transforms
    "transform_xlr", "transform_clr", "transform_iqlr", "transform_ilr","transform_closure",
    # Pairwise
    "pairwise_vlr", "pairwise_rho","pairwise_phi",
    # Utilities
    "check_packages","assert_acceptable_arguments","check_compositional",
    # Filtering
    "filter_data_highpass",
    # Metrics
    "sparsity","number_of_components","prevalence_of_components",
]
__classes__ = []

__all__ = sorted(__functions__ + __classes__)

from .compositional import *
