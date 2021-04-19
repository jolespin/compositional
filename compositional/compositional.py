# -*- coding: utf-8 -*-
from __future__ import print_function, division

# Built-ins
import sys,warnings,functools
from collections import Mapping
from importlib import import_module

# Version specific
if sys.version_info.major == 2:
    from StringIO import StringIO
if sys.version_info.major == 3:
    from io import StringIO

# External
import numpy as np
import pandas as pd
from pandas._libs.algos import nancorr

# =========
# Utilities
# =========
# Check packages
def check_packages(packages, namespace=None, import_into_backend=True, verbose=False):
    """
    Check if packages are available (and import into global namespace)
    If package is a tuple then imports as follows: ("numpy", "np") where "numpy" is full package name and "np" is abbreviation
    To import packages into current namespace: namespace = globals()
    To import packages in backend, e.g. if this is used in a module/script, use `import_into_backend`

    packages: str, non-tuple iterable

    usage:
    @check_packages(["sklearn", "scipy", ("numpy", "np")])
    def f():
        pass

    Adapted from the following source:
    soothsayer_utils (https://github.com/jolespin/soothsayer_utils)
    """
    # Force packages into sorted non-redundant list
    if isinstance(packages,(str, tuple)):
        packages = [packages]
    packages = set(packages)

    # Set up decorator for package imports   
    # Wrapper
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            missing_packages = []
            for pkg in packages:
                if isinstance(pkg, tuple):
                    assert len(pkg) == 2, "If a package is tuple type then it must have 2 elements e.g. ('numpy', 'np')"
                    pkg_name, pkg_variable = pkg
                else:
                    pkg_name = pkg_variable = pkg 
                try:
                    package = import_module(pkg_name)
                    if import_into_backend:
                        globals()[pkg_variable] = package
                    if namespace is not None:
                        namespace[pkg_variable] = package
                    if verbose:
                        print("Importing {} as {}".format(pkg_name, pkg_variable), True, file=sys.stderr)
                except ImportError:
                    missing_packages.append(pkg_name)
                    if verbose:
                        print("Cannot import {}:".format(pkg_name), False, file=sys.stderr)
            assert not missing_packages, "Please install the following packages to use this function:\n{}".format( ", ".join(missing_packages))
            return func(*args, **kwargs)

        return wrapper
    return decorator

# ===========================
# Compositional data analysis
# ===========================
# Extension of CLR to use custom centroids, references, and zeros without pseudocounts
def transform_xlr(X, reference_components=None, centroid="mean", return_zeros_as_neginfinity=False, zeros_ok=True):
    """
    # Description
    Extension of CLR to incorporate custom centroids, reference components (iqlr), and handle missing values.
    This implementation is more versatile than skbio's implementation but that makes it slower if it done iteratively.

    # Documentation on CLR:
    http://scikit-bio.org/docs/latest/generated/skbio.stats.composition.clr.html#skbio.stats.composition.clr
    
    # Parameters
        * X:
            - Compositional data
            (1D): pd.Series or 1D np.array
            (2D): pd.DataFrame or 2D np.array
        * centroid: 
            - Can be precomputed or a function applied to the log-transformed composition(s)
            (1/2)D: 'mean', 'median', callable function
            (1D): numeric
            (2D): pd.Series, dict
        * reference_components:
            - Custom group of components used during the centroid calculation
        * return_zeros_as_neginfinity:
            True: Returns zeros as -np.inf 
            False: Returns zeros as np.nan
        * zeros_ok:
            True: Mask zeros with np.nan with warning
            False: Error
    """
    n_dimensions = len(X.shape)
    assert n_dimensions in {1,2}, "`X` must be 1D or 2D"
    assert np.all(X >= 0), "`X` cannot contain negative values because of log-transformation step."
    assert not isinstance(reference_components, tuple), "`reference_components` cannot be type tuple"
    # 1-Dimensional
    if n_dimensions == 1:
        # Check for labels
        components = None
        if isinstance(X, pd.Series):
            components = X.index
            X = X.values
            if reference_components is None:
                reference_components = components
            reference_components = list(map(lambda component: components.get_loc(component), reference_components))
        X = X.astype(float)
        
        # Check for zeros
        X_contains_zeros = False
        n_zeros = np.any(X == 0).flatten().sum()
        if n_zeros:
            if zeros_ok:
                mask_zeros = X == 0
                X[mask_zeros] = np.nan
                X_contains_zeros = True
                warnings.warn("N={} zeros detected in `X`.  Masking zeros as NaN and will default to nan-robust functions if 'mean' or 'median' were provided for centroid".format(n_zeros))
            else:
                raise Exception("N={} zeros detected in `X`.  Either preprocess, add pseudocounts, or `zeros_ok=True".format(n_zeros))    

        # Log transformation
        X_log = np.log(X)

        # Centroid
        centroid_is_string = isinstance(centroid, str)
        centroid_is_function = hasattr(centroid, "__call__")
        centroid_is_precomputed = np.issubdtype(type(centroid), np.number)

        if not centroid_is_precomputed:
            # Get function associated with string for centroid
            if centroid_is_string:
                centroid = centroid.lower()
                assert centroid in {"mean", "median"}, "Please use 'mean','median', or a precomputed centroid"
                if X_contains_zeros:
                    centroid = {"mean":np.nanmean, "median":np.nanmedian}[centroid]
                else:
                    centroid = {"mean":np.mean, "median":np.median}[centroid]
                centroid_is_function = True

            # Compute centroid using function
            if centroid_is_function:
                func = centroid
                centroid = func(X_log[reference_components])

        # Transform
        X_transformed = X_log - centroid

        # Output
        if all([return_zeros_as_neginfinity, X_contains_zeros]):
            X_transformed[mask_zeros] = -np.inf

        if components is not None:
            X_transformed = pd.Series(X_transformed, index=components)
        return X_transformed
    
    # 2-Dimensional
    if n_dimensions == 2:
        # Check for labels
        index = None
        components = None
        if isinstance(X, pd.DataFrame):
            index = X.index
            components = X.columns
            X = X.values
            if reference_components is None:
                reference_components = components
            reference_components = list(map(lambda component: components.get_loc(component), reference_components))
        X = X.astype(float)

        # Check for zeros
        X_contains_zeros = False
        n_zeros = np.any(X == 0).flatten().sum()
        if n_zeros:
            if zeros_ok:
                mask_zeros = X == 0
                X[mask_zeros] = np.nan
                X_contains_zeros = True
                warnings.warn("N={} zeros detected in `X`.  Masking zeros as NaN and will default to nan-robust functions if 'mean' or 'median' were provided for centroid".format(n_zeros))
            else:
                raise Exception("N={} zeros detected in `X`.  Either preprocess, add pseudocounts, or `zeros_ok=True".format(n_zeros))    

        # Log transformation
        X_log = np.log(X)

        # Centroid
        centroid_is_string = isinstance(centroid, str)
        centroid_is_function = hasattr(centroid, "__call__")
        centroid_is_precomputed = False

        # Preprocess precomputed centroid
        if np.all(np.logical_not([centroid_is_string, centroid_is_function])):
            if index is not None:
                if isinstance(centroid, Mapping):
                    centroid = pd.Series(centroid)
                assert isinstance(centroid, pd.Series), "If `centroid` is dict-like/pd.Series then `X` must be a `pd.DataFrame`."
                assert set(centroid.index) >= set(index), "Not all indicies from `centroid` are available in `X.index`."
                centroid = centroid[index].values
            assert len(centroid) == X_log.shape[0], "Dimensionality is not compatible: centroid.size != X.shape[0]."
            centroid_is_precomputed = True

        if not centroid_is_precomputed:
            # Get function associated with string for centroid
            if centroid_is_string:
                centroid = centroid.lower()
                assert centroid in {"mean", "median"}, "Please use 'mean','median', or a precomputed centroid"
                if X_contains_zeros:
                    centroid = {"mean":np.nanmean, "median":np.nanmedian}[centroid]
                else:
                    centroid = {"mean":np.mean, "median":np.median}[centroid]
                centroid_is_function = True

            # Compute centroid using function
            if centroid_is_function:
                func = centroid
                # If function has "axis" argument
                try:
                    centroid = func(X_log[:,reference_components], axis=-1)
                # If function does not have "axis" argument
                except TypeError: 
                    centroid = list(map(func, X_log[:,reference_components]))

        # Broadcast centroid
        centroid = np.asarray(centroid)
        if len(centroid.shape) == 1:
            centroid = centroid[:,np.newaxis]
            
        # Transform
        X_transformed = X_log - centroid

        # Output
        if all([return_zeros_as_neginfinity, X_contains_zeros]):
            X_transformed[mask_zeros] = -np.inf

        if components is not None:
            X_transformed = pd.DataFrame(X_transformed, index=index, columns=components)
        return X_transformed

# CLR Normalization
def transform_clr(X, return_zeros_as_neginfinity=False, zeros_ok=True):
    """
    Wrapper around `transform_xlr`
    
    # Description
    Extension of CLR to handle missing values.
    This implementation is more versatile than skbio's implementation but that makes it slower if it done iteratively.

    # Documentation on CLR:
    http://scikit-bio.org/docs/latest/generated/skbio.stats.composition.clr.html#skbio.stats.composition.clr
    
    # Parameters
        * X:
            - Compositional data
            (1D): pd.Series or 1D np.array
            (2D): pd.DataFrame or 2D np.array
        * return_zeros_as_neginfinity:
            True: Returns zeros as -np.inf 
            False: Returns zeros as np.nan
        * zeros_ok:
            True: Mask zeros with np.nan with warning
            False: Error
    """
    return transform_xlr(X, reference_components=None, centroid="mean", return_zeros_as_neginfinity=return_zeros_as_neginfinity, zeros_ok=zeros_ok)

# Interquartile range log-ratio transform
def transform_iqlr(X, percentile_range=(25,75), centroid="mean", interval_type="open", return_zeros_as_neginfinity=False, zeros_ok=True, ddof=1):
    """
    Wrapper around `transform_xlr`

    # Description
    Interquartile range log-ratio transform
    
    # Parameters
        * X: pd.DataFrame or 2D np.array
        * percentile_range: A 2-element tuple of percentiles
        * interval: 'open' = (a,b) and 'closed' = [a,b].  'open' is used by `propr` R package:
        * centroid, return_zeros_as_neginfinity, and zeros_ok: See `transform_xlr`

    Adapted from the following source:
        * https://github.com/tpq/propr/blob/2bd7c44bf59eaac6b4d329d38afd40ac83e2089a/R/2-proprCall.R#L31
    """
    # Checks
    n_dimensions = len(X.shape)
    assert n_dimensions in {2}, "`X` must be 2D"
    assert np.all(X >= 0), "`X` cannot contain negative values because of log-transformation step."
    assert interval_type in {"closed", "open"}, "`interval_type` must be in the following: {closed, open}"
    percentile_range = tuple(sorted(percentile_range))
    assert len(percentile_range) == 2, "percentile_range must have 2 elements"
    
    index=None
    components=None
    if isinstance(X, pd.DataFrame):
        index = X.index
        components = X.columns
        X = X.values
    # Compute the variance of the XLR transform
    X_xlr = transform_xlr(X, centroid=centroid, return_zeros_as_neginfinity=False, zeros_ok=zeros_ok)
    xlr_var = np.nanvar(X_xlr, axis=0, ddof=ddof)

    # Calculate upper and lower bounds from percentiles
    lower_bound, upper_bound = np.percentile(xlr_var, percentile_range)

    # Get the reference components
    if interval_type == "open":
        reference_components = np.where((lower_bound < xlr_var) & (xlr_var < upper_bound))[0]

    if interval_type == "closed":
        reference_components = np.where((lower_bound <= xlr_var) & (xlr_var <= upper_bound))[0]
    X_iqlr = transform_xlr(X, reference_components=reference_components, centroid=centroid, return_zeros_as_neginfinity=return_zeros_as_neginfinity, zeros_ok=zeros_ok)

    if components is not None:
        X_iqlr = pd.DataFrame(X_iqlr, index=index, columns=components)
    return X_iqlr

# Pairwise variance log-ratio
def pairwise_vlr(X):
    """
    # Description
    Pairwise variance log-ratio
    
    # Parameters
        * X: pd.DataFrame or 2D np.array
        
    Adapted from the following source:
    * https://github.com/tpq/propr
    ddof=1 for compatibility with propr package in R
    To properly handle missing values and optimize speed, nancorr from pandas must be used which does not take ddof
    """
    # Checks
    n_dimensions = len(X.shape)
    assert n_dimensions in {2}, "`X` must be 2D"
    assert np.all(X >= 0), "`X` cannot contain negative values because of log-transformation step."

    components = None
    if isinstance(X, pd.DataFrame):
        components = X.columns
        X = X.values
    X = X.astype("float64")
    n,m = X.shape
    
    # Check for zeros
    n_zeros = np.any(X == 0).flatten().sum()
    if n_zeros:
        raise Exception("N={} zeros detected in `X`.  Either preprocess or add pseudocounts.".format(n_zeros))    

    X_log = np.log(X)
    covariance = nancorr(X_log, cov=True) # covariance = np.cov(X_log.T, ddof=1)
    diagonal = np.diagonal(covariance)
    vlr = -2*covariance + diagonal[:,np.newaxis] + diagonal
    if components is not None:
        vlr = pd.DataFrame(vlr, index=components, columns=components)
    return vlr

# Pairwise rho proportionality
def pairwise_rho(X=None, reference_components=None, centroid="mean", interval_type="open", xlr=None, vlr=None):
    """
    # Description
    Pairwise proportionality `rho` (Erb et al. 2016)
    
    # Parameters
        * X: pd.DataFrame or 2D np.array of compositional data (rows=samples, columns=components)
        * reference_components: See `transform_xlr`.  Can also be `percentiles` for `transform_iqlr` or 'iqlr' string.
        * interval: 'open' = (a,b) and 'closed' = [a,b].  'open' is used by `propr` R package:
        * centroid: See `transform_xlr`
        * xlr: pd.DataFrame or 2D np.array of transformed compositional data (e.g. clr, iqlr) (must be used with `vlr` and not `X`)
        * vlr: pd.DataFrame or 2D np.array of variance log-ratios (must be used with `xlr` and not `X`)

        
    Adapted from the following source:
    * https://github.com/tpq/propr
    Citation:
    * https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004075
    * https://link.springer.com/article/10.1007/s12064-015-0220-8
    ddof=1 for compatibility with propr package in R
    """
    components = None
    # Compute xlr and vlr from X
    if X is not None:
        assert all(map(lambda x: x is None, [xlr, vlr])), "If `X` is not None then `xlr` and `vlr` cannot be provided."
        if isinstance(X, pd.DataFrame):
            components = X.columns
            X = X.values

        vlr = pairwise_vlr(X)
        if isinstance(reference_components, str):
            if reference_components.lower() == "iqlr":
                reference_components = (25,75)
        # Use percentiles
        if isinstance(reference_components, tuple):
            xlr = transform_iqlr(X,percentile_range=reference_components, centroid=centroid, interval_type=interval_type, zeros_ok=False)
        # Use CLR
        else:
            xlr = transform_xlr(X, reference_components=reference_components, centroid=centroid, zeros_ok=False)
    # Provide xlr and vlr
    else:
        assert all(map(lambda x: x is not None, [xlr,vlr])), "If `X` is None then `xlr` and `vlr` must be provided."
        assert type(xlr) is type(vlr), "`xlr` and `vlr` should be same type (i.e. pd.DataFrame, np.ndarray)"
        if isinstance(xlr, pd.DataFrame):
            assert np.all(xlr.columns == vlr.columns) & np.all(xlr.columns == vlr.index), "`xlr.columns` need to be the same as `vlr.index` and `vlr.columns`"
            components = xlr.columns
            xlr = xlr.values
            vlr = vlr.values
            
    # rho (Erb et al. 2016)
    n, m = xlr.shape
    variances = np.var(xlr, axis=0, ddof=1) # variances = np.var(X_xlr, axis=0, ddof=ddof)
    rhos = 1 - (vlr/np.add.outer(variances,variances))    
    if components is not None:
        rhos = pd.DataFrame(rhos, index=components, columns=components)
    return rhos

# Pairwise phi proportionality
def pairwise_phi(X=None, symmetrize=True, triangle="lower", reference_components=None, centroid="mean", interval_type="open", xlr=None, vlr=None):
    """
    # Description
    Pairwise proportionality `phi` (Lovell et al. 2015)
    
    # Parameters
        * X: pd.DataFrame or 2D np.array of compositional data (rows=samples, columns=components)
        * symmetrize: Force symmetric matrix
        * triangle: Use lower or upper triangle for reference during symmetrization
        * reference_components: See `transform_xlr`.  Can also be `percentiles` for `transform_iqlr` or 'iqlr' string.
        * interval: 'open' = (a,b) and 'closed' = [a,b].  'open' is used by `propr` R package:
        * centroid: See `transform_xlr`
        * xlr: pd.DataFrame or 2D np.array of transformed compositional data (e.g. clr, iqlr) (must be used with `vlr` and not `X`)
        * vlr: pd.DataFrame or 2D np.array of variance log-ratios (must be used with `xlr` and not `X`)
        
    Adapted from the following source:
    * https://github.com/tpq/propr
    Citation:
    * https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004075
    ddof=1 for compatibility with propr package in R
    """
    components = None
    # Compute xlr and vlr from X
    if X is not None:
        assert all(map(lambda x: x is None, [xlr, vlr])), "If `X` is not None then `xlr` and `vlr` cannot be provided."
        if isinstance(X, pd.DataFrame):
            components = X.columns
            X = X.values

        vlr = pairwise_vlr(X)
        if isinstance(reference_components, str):
            if reference_components.lower() == "iqlr":
                reference_components = (25,75)
        # Use percentiles
        if isinstance(reference_components, tuple):
            xlr = transform_iqlr(X,percentile_range=reference_components, centroid=centroid, interval_type=interval_type, zeros_ok=False)
        # Use CLR
        else:
            xlr = transform_xlr(X, reference_components=reference_components, centroid=centroid, zeros_ok=False)
    # Provide xlr and vlr
    else:
        assert all(map(lambda x: x is not None, [xlr,vlr])), "If `X` is None then `xlr` and `vlr` must be provided."
        assert type(xlr) is type(vlr), "`xlr` and `vlr` should be same type (i.e. pd.DataFrame, np.ndarray)"
        if isinstance(xlr, pd.DataFrame):
            assert np.all(xlr.columns == vlr.columns) & np.all(xlr.columns == vlr.index), "`xlr.columns` need to be the same as `vlr.index` and `vlr.columns`"
            components = xlr.columns
            xlr = xlr.values
            vlr = vlr.values
            
    # phi (Lovell et al. 2015)
    n, m = xlr.shape
    variances = np.var(xlr, axis=0, ddof=1)#[:,np.newaxis]
    phis = vlr/variances   
    if symmetrize:
        assert triangle in {"lower","upper"}, "`triangle` must be one of the following: {'lower','upper'}"
        if triangle == "upper":
            idx_triangle = np.tril_indices(m, -1)
        if triangle == "lower":
            idx_triangle = np.triu_indices(m, 1)
        phis[idx_triangle] = phis.T[idx_triangle]
    if components is not None:
        phis = pd.DataFrame(phis, index=components, columns=components)
    return phis


# ILR Transformation
@check_packages(["skbio"])
def transform_ilr(X:pd.DataFrame, tree=None,  check_polytomy=True,  verbose=True):
    """
    if `tree` is None then orthonormal basis for Aitchison simplex defaults to J.J.Egozcue orthonormal basis.
    """
    # Imports
    from skbio import TreeNode
    from skbio.stats.composition import ilr

    assert isinstance(X, pd.DataFrame), "`X` must be a pd.DataFrame"
    assert not np.any(X == 0), "`X` cannot contain zeros because of log-transforms.  Preprocess or use a pseudocount e.g. (X+1) or (X/(1/X.shape[1]**2))"

    # Determine tree module
    def _infer_tree_type(tree):
        tree_type = None
        query_type = str(tree.__class__).split("'")[1].split(".")[0]
        if query_type in {"skbio"}:
            tree_type = "skbio"
        if query_type in {"ete2","ete3"}:
            tree_type = "ete"
        assert tree_type is not None, "Please use either skbio or ete[2/3] tree.  Tree type deterined as {}".format(query_type)
        return tree_type

    # Get leaves from tree
    def _get_leaves(tree, tree_type):
        if tree_type == "skbio":
            leaves_in_tree =  set(map(lambda leaf:leaf.name, tree.tips()))
        if tree_type == "ete":
            leaves_in_tree =  set(tree.get_leaf_names())
        return leaves_in_tree

    def _check_polytomy(tree, tree_type):
        if tree_type == "ete":
            # Check bifurcation
            n_internal_nodes = len(list(filter(lambda node:node.is_leaf() == False, tree.traverse())))
            n_leaves = len(list(filter(lambda node:node.is_leaf(), tree.traverse())))
            if n_internal_nodes < (n_leaves - 1):
                raise Exception("Please resolve tree polytomy and force bifurcation: Use `tree.resolve_polytomy()` before naming nodes for `ete`")

        if tree_type == "skbio":
            # Check bifurcation
            n_internal_nodes = len(list(filter(lambda node:node.is_tip() == False, tree.traverse())))
            n_leaves = len(list(filter(lambda node:node.is_tip(), tree.traverse())))
            if n_internal_nodes < (n_leaves - 1):
                raise Exception("Please resolve tree polytomy and force bifurcation: Use `tree.bifurcate()` before naming nodes for `skbio`")

    # ETE Tree
    if sys.version_info.major == 2:
        ete_info = ("ete2","ete")
    if sys.version_info.major == 3:
        ete_info = ("ete3","ete")
    @check_packages([ete_info])
    def _ete_to_skbio( tree):
        # Convert ete to skbio
        tree = TreeNode.read(StringIO(tree.write(format=1, format_root_node=True)), convert_underscores=False)
        return tree

    def _prune_tree(tree, tree_type, leaves):
        if tree_type == "ete":
            tree.prune(leaves)
        if tree_type == "skbio":
            tree = tree.shear(leaves)
            tree.prune()
        return tree
    
    # ILR with tree
    @check_packages(["gneiss"], import_into_backend=False)
    def _ilr_with_tree(X, tree):
        # Import ilr_transform from gneiss
        from gneiss.composition import ilr_transform

        # Check tree type
        tree_type = _infer_tree_type(tree)

        # Check leaves 
        components = set(X.columns)
        leaves_in_tree =  _get_leaves(tree=tree, tree_type=tree_type)
        assert components <= leaves_in_tree, "Not all components (X.columns) are represented in tree"

        # Prune tree
        if components < leaves_in_tree:
            tree = tree.copy()
            n_leaves_before_pruning = len(leaves_in_tree)
            tree = _prune_tree(tree=tree, tree_type=tree_type, leaves=components)
            n_leaves_after_pruning = len(_get_leaves(tree=tree, tree_type=tree_type))
            n_pruned = n_leaves_before_pruning - n_leaves_after_pruning
            if verbose:
                print("Pruned {} attributes to match components (X.columns)".format(n_pruned), file=sys.stderr)

        # Polytomy
        if check_polytomy:
            _check_polytomy(tree=tree, tree_type=tree_type)

        # ETE
        if tree_type == "ete":
            tree = _ete_to_skbio(tree=tree)
        return ilr_transform(table=X, tree=tree)
        
    # ILR without tree
    def _ilr_without_tree(X):
        return pd.DataFrame(ilr(X), index=X.index)

    # Without tree
    if tree is None:
        return _ilr_without_tree(X=X)
    # With tree
    else:
        return _ilr_with_tree(X=X, tree=tree)

