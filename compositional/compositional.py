# -*- coding: utf-8 -*-
from __future__ import print_function, division

# Built-ins
import sys,warnings,functools, operator
from collections.abc import Mapping 
from importlib import import_module
from itertools import combinations

# Version specific
if sys.version_info.major == 2:
    from StringIO import StringIO
if sys.version_info.major == 3:
    from io import StringIO

# External
import numpy as np
import pandas as pd
from pandas._libs.algos import nancorr
from scipy.spatial.distance import pdist, squareform

# =========
# Utilities
# =========
def assert_acceptable_arguments(query, target, operation="le", message="Invalid option provided.  Please refer to the following for acceptable arguments:"):
    """
    le: operator.le(a, b) : <=
    eq: operator.eq(a, b) : ==
    ge: operator.ge(a, b) : >=
    """
    def is_nonstring_iterable(obj):
        condition_1 = hasattr(obj, "__iter__")
        condition_2 =  not type(obj) == str
        return all([condition_1,condition_2])
    
    # If query is not a nonstring iterable or a tuple
    if any([
            not is_nonstring_iterable(query),
            isinstance(query,tuple),
            ]):
        query = [query]
    query = set(query)
    target = set(target)
    func_operation = getattr(operator, operation)
    assert func_operation(query,target), "{}\n{}".format(message, target)

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

def check_compositional(X, n_dimensions:int=None, acceptable_dimensions:set={1,2}, is_integer=False, is_proportional=False, is_integer_or_proportional=False):
    """
    # Description
    Check that 1D and 2D NumPy/Pandas objects are the correct shape and >= 0

    # Parameters
        * X:
            - Compositional data
            (1D): pd.Series or 1D np.array
            (2D): pd.DataFrame or 2D np.array
        * n_dimensions: int   
    """
    if n_dimensions is None:
        n_dimensions = len(X.shape)
    if not hasattr(acceptable_dimensions, "__iter__"):
        acceptable_dimensions = {acceptable_dimensions}
    assert n_dimensions in acceptable_dimensions, "`X` must be {}".format(" or ".join(map(lambda d: f"{d}D", acceptable_dimensions)))
    assert np.all(X >= 0), "`X` cannot contain negative values."
    if is_integer:
        assert np.all(X == X.astype(int)), "`X` must be integer counts"
    if is_proportional:
        assert np.allclose(X.sum(axis=1), np.ones(X.shape[0])), "`X` must be proportional and each composition must sum up to 1"
    if is_integer_or_proportional:
        assert any([
            np.all(X == X.astype(int)),
            np.allclose(X.sum(axis=1), np.ones(X.shape[0])),
        ]), "`X` must be integer counts or proportional so each composition sums up to 1"

# ===========================
# Summary metrics
# ===========================
def sparsity(X, checks=True):
    """
    # Description
    Calculates the sparsity (i.e., ratio of zeros) in a NumPy or Pandas object

    # Parameters
        * X:
            - Compositional data
            (1D): pd.Series or 1D np.array
            (2D): pd.DataFrame or 2D np.array
        * checks:
            Check whether or not dimmensions are correct and data is >= 0
    * Output
        Ratio zeros
    """
    n_dimensions = len(X.shape)
    if checks:
        check_compositional(X, n_dimensions)

    if n_dimensions == 2:
        if isinstance(X, pd.DataFrame):
            X = X.values
        X = X.ravel()
    number_of_zeros = np.sum(X == 0)
    number_of_values = X.size
    return number_of_zeros/number_of_values

def number_of_components(X, checks=True):
    """
    # Description
    Calculates the number of detected components (i.e., richness) in a NumPy or Pandas object

    # Parameters
        * X:
            - Compositional data
            (1D): pd.Series or 1D np.array of a composition (i.e., sample)
            (2D): pd.DataFrame or 2D np.array (rows=samples/compositions, columns=features/components)
        * checks:
            Check whether or not dimmensions are correct and data is >= 0
    * Output
        Number of components per composition (i.e., sample)
    """
    n_dimensions = len(X.shape)
    
    if checks:
        check_compositional(X, n_dimensions)
    
    if n_dimensions == 2:
        return (X > 0).sum(axis=1)
        
    else:
        return (X > 0).sum()

def prevalence_of_components(X, minimum_count=1, checks=True):
    """
    # Description
    Calculates the prevalence of detected components in a NumPy or Pandas object

    # Parameters
        * X:
            - Compositional data
            (1D): pd.Series or 1D np.array of a component vector
            (2D): pd.DataFrame or 2D np.array (rows=samples/compositions, columns=features/components)
        * checks:
            Check whether or not dimmensions are correct and data is >= 0
    * Output
        Number of compositions where a component was detected
    """
    
    n_dimensions = len(X.shape)
    
    if checks:
        check_compositional(X, n_dimensions)
    
    if n_dimensions == 2:
        return (X >= minimum_count).sum(axis=0)
        
    else:
        return (X >= minimum_count).sum()
    
# ==========
# Conversion
# ==========
def covariance_to_correlation(V):
    """
    Convert Covariance matrix to Correlation matrix efficiently.
    
    Arguments:
    V: a covariance matrix (i.e. symmetric and positive definite)

    Implementation help from @joão-areias:
    https://stackoverflow.com/a/76978819/678572
    """
    p, d = V.shape
    Is = np.sqrt(1 / np.diag(V))  # diag( 1/sigma_i )

    r = V.copy()  # keep dimnames
    r *= Is.reshape(-1, 1) * Is.reshape(1, -1)
    np.fill_diagonal(r, 1)  # exact in diagonal
    return r


def correlation_to_partial_correlation(m, tol=1e-15):
    """
    Convert a correlation matrix to a partial correlation matrix efficiently.

    Arguments:
    m: a correlation matrix
    tol: tolerance for calculating the pseudo-inverse

    Implementation help from @joão-areias:
    https://stackoverflow.com/a/76978819/678572
    """
    # Invert, then negate off-diagonal entries
    m = -np.linalg.pinv(m, rcond=tol)
    np.fill_diagonal(m, -np.diag(m))

    # Standardize and return
    return covariance_to_correlation(m)

# ===========================
# Transforms
# ===========================
def transform_closure(X, checks=True):
    """
    # Description
    Closure (e.g., total sum scaling, relative abundance) that can handle 1D and 2D NumPy  and Pandas objects

    # Parameters
        * X:
            - Compositional data
            (1D): pd.Series or 1D np.array
            (2D): pd.DataFrame or 2D np.array
        * checks:
            Check whether or not dimmensions are correct and data is >= 0
    * Output
        Closure transformed matching input object class
    """
    
    n_dimensions = len(X.shape)

    if checks:
        check_compositional(X, n_dimensions)
    
    if n_dimensions == 2:
        
        index = None
        components = None
        
        if isinstance(X, pd.DataFrame):
            index = X.index
            components = X.columns
            X = X.values
            
        X_closure = X/X.sum(axis=1).reshape(-1,1)
        
        if index is not None:
            X_closure = pd.DataFrame(X_closure, index=index, columns=components)
        
    else:
        components = None
        if isinstance(X, pd.Series):
            components = X.index
            X = X.values
            
        X_closure = X/X.sum()
        if components is not None:
            X_closure = pd.Series(X_closure, index=components)

    return X_closure

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
    check_compositional(X, n_dimensions)

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
    check_compositional(X, n_dimensions)
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
def pairwise_vlr(X, redundant_form:bool=True):
    """
    # Description
    Pairwise variance log-ratio
    
    # Parameters
        * X: pd.DataFrame or 2D np.array
        * redundant_form:
            - True: Return output in squareform
            - False: Return the dereplicated VLR
    # Output: 
        - Returns Pairwise VLR
            * X -> pd.DataFrame
                - redundant_form: True
                    pd.DataFrame with index and columns equal to X.columns
                - redundant_form: False
                    pd.Series with index as a frozenset of combinations (i.e., list(map(frozenset, combinations(components, 2))))
            * X -> np.array
                - redundant_form: True
                    2D np.array
                - redundant_form: False
                    1D np.array
                
    Adapted from the following source:
    * https://github.com/tpq/propr
    ddof=1 for compatibility with propr package in R
    To properly handle missing values and optimize speed, nancorr from pandas must be used which does not take ddof
    """
    # Checks
    n_dimensions = len(X.shape)
    check_compositional(X, n_dimensions, acceptable_dimensions={2})

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

    # Format and add labels if Pandas
    if redundant_form:
        if components is not None:
            vlr = pd.DataFrame(vlr, index=components, columns=components)
    else:
        vlr = squareform(vlr, checks=False)

        if components is not None:
            components = pd.Index(list(map(frozenset, combinations(components, 2))), name=components.name)
            if components.name is None:
                components.name = "vlr"
            vlr = pd.Series(vlr, index=components)

    return vlr

# Pairwise rho proportionality
def pairwise_rho(X=None, redundant_form:bool=True, reference_components=None, centroid="mean", interval_type="open", xlr=None, vlr=None):
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
        * redundant_form:
            - True: Return output in squareform
            - False: Return the dereplicated rho
    # Output: 
        - Returns pairwise rho
            * X -> pd.DataFrame
                - redundant_form: True
                    pd.DataFrame with index and columns equal to X.columns
                - redundant_form: False
                    pd.Series with index as a frozenset of combinations (i.e., list(map(frozenset, combinations(components, 2))))
            * X -> np.array
                - redundant_form: True
                    2D np.array
                - redundant_form: False
                    1D np.array

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

    # Format and add labels if Pandas
    if redundant_form:
        if components is not None:
            rhos = pd.DataFrame(rhos, index=components, columns=components)
    else:
        rhos = squareform(rhos, checks=False)

        if components is not None:
            components = pd.Index(list(map(frozenset, combinations(components, 2))), name=components.name)
            if components.name is None:
                components.name = "rho"
            rhos = pd.Series(rhos, index=components)
    return rhos

# Pairwise phi proportionality
def pairwise_phi(X=None, redundant_form:bool=True, symmetrize=True, triangle="lower", reference_components=None, centroid="mean", interval_type="open", xlr=None, vlr=None):
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
        * redundant_form:
            - True: Return output in squareform
            - False: Return the dereplicated phi
    # Output: 
        - Returns pairwise phi
            * X -> pd.DataFrame
                - redundant_form: True
                    pd.DataFrame with index and columns equal to X.columns
                - redundant_form: False
                    pd.Series with index as a frozenset of combinations (i.e., list(map(frozenset, combinations(components, 2))))
            * X -> np.array
                - redundant_form: True
                    2D np.array
                - redundant_form: False
                    1D np.array

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

    # Format and add labels if Pandas
    if redundant_form:
        if components is not None:
            phis = pd.DataFrame(phis, index=components, columns=components)
    else:
        phis = squareform(phis, checks=False)

        if components is not None:
            components = pd.Index(list(map(frozenset, combinations(components, 2))), name=components.name)
            if components.name is None:
                components.name = "phi"
            phis = pd.Series(phis, index=components)
    return phis




@check_packages(["sklearn"])
def pairwise_partial_correlation_with_basis_shrinkage(X: pd.DataFrame, redundant_form=True, checks=True):
    """
    # Description
    Computes pairwise partial correlation with basis shrinkage using the Ledoit-Wolf shrinkage method

    Please cite the following references:
        * Erb et al. 2020 (https://www.sciencedirect.com/science/article/pii/S2590197420300082)
        * Jin et al. 2022 (https://arxiv.org/pdf/2212.00496.pdf)
    
    # Parameters
        * X:
            - Compositional data
            (2D): pd.DataFrame or 2D np.array
        * redundant_form:
            - True: Return output in squareform
            - False: Return the dereplicated distances
        * checks:
            - Check if data is non-negative and 2D

    # Output: 
        - Returns pairwise partial correlation with basis shrinkage
            * X -> pd.DataFrame
                - redundant_form: True
                    pd.DataFrame with index and columns equal to X.index
                - redundant_form: False
                    pd.Series with index as a frozenset of combinations (i.e., list(map(frozenset, combinations(index, 2))))
            * X -> np.array
                - redundant_form: True
                    2D np.array
                - redundant_form: False
                    1D np.array


    # Notes:
        As implemented in bShrink (https://github.com/tpq/propr/blob/12553b3bcd159649f25d9a0e480250c1eee1d965/R/1-propr.R#L326) 
        I've started a GitHub issue to get this resolved: https://github.com/scikit-learn/scikit-learn/issues/27192
        
        However, R implementation uses an updated method not yet available in Scikit-learn (or Python AFAIK):
            * Opgen-Rhein, R., and K. Strimmer. 2007. Accurate ranking of differentially expressed genes by a distribution-free shrinkage approach. 
                Statist. Appl. Genet. Mol. Biol. 6:9. <DOI:10.2202/1544-6115.1252>
            * Schafer, J., and K. Strimmer. 2005. A shrinkage approach to large-scale covariance estimation and implications for functional genomics. 
                Statist. Appl. Genet. Mol. Biol. 4:32. <DOI:10.2202/1544-6115.1175>
    """
    from sklearn.covariance import ledoit_wolf

    n_dimensions = len(X.shape)
    if checks:
        check_compositional(X, n_dimensions=n_dimensions, acceptable_dimensions={2})
    
    # Convert input data to a NumPy array
    components = None
    if isinstance(X, pd.DataFrame):
        components = X.columns
        X = X.values

    # Get dimensions
    D = X.shape[1]

    # Transform counts to log proportions
    P = X / X.sum(axis=1).reshape(-1, 1)
    B = np.log(P)

    # Covariance shrinkage
    Cb, shrinkage = ledoit_wolf(B)
    G = np.eye(D) - np.ones((D, D)) / D
    cov = G @ Cb @ G # cov = np.dot(np.dot(G, Cb), G)

    # Partial correlation
    pcorr = correlation_to_partial_correlation(cov)
    

    # Format and add labels if Pandas
    if redundant_form:
        if components is not None:
            pcorr = pd.DataFrame(pcorr, index=components, columns=components)
    else:
        pcorr = squareform(pcorr, checks=False)

        if components is not None:
            components = pd.Index(list(map(frozenset, combinations(components, 2))), name=components.name)
            if components.name is None:
                components.name = "pairwise_partial_correlation"
            pcorr = pd.Series(pcorr, index=components)

    return pcorr

def pairwise_aitchison_distance(X, redundant_form:bool=True, checks=True):
    """
    # Description
    Computes pairwise Aitchison distance on a matrix (i.e., CLR transform -> Euclidean distance)
    
    # Parameters
        * X:
            - Compositional data
            (2D): pd.DataFrame or 2D np.array
        * redundant_form:
            - True: Return output in squareform
            - False: Return the dereplicated distances
        * checks:
            - Check if data is non-negative and 2D

    # Output: 
        - Returns pairwise Aitchison distances
            * X -> pd.DataFrame
                - redundant_form: True
                    pd.DataFrame with index and columns equal to X.index
                - redundant_form: False
                    pd.Series with index as a frozenset of combinations (i.e., list(map(frozenset, combinations(index, 2))))
            * X -> np.array
                - redundant_form: True
                    2D np.array
                - redundant_form: False
                    1D np.array
            
    """
    n_dimensions = len(X.shape)
    if checks:
        check_compositional(X, n_dimensions=n_dimensions, acceptable_dimensions={2})

    # Convert input data to a NumPy array
    index=None
    if isinstance(X, pd.DataFrame):
        index = X.index
        X = X.values

    # CLR Transform
    X_clr = transform_clr(X)

    # Aitchison Distance
    aitchison_distance = pdist(X_clr)

    # Format and add labels if Pandas
    if redundant_form:
        aitchison_distance = squareform(aitchison_distance, checks=False)
        if index is not None:
            aitchison_distance = pd.DataFrame(aitchison_distance, index=index, columns=index)
    else:
        if index is not None:
            index = pd.Index(list(map(frozenset, combinations(index, 2))), name=index.name)
            if index.name is None:
                index.name = "aitchison_distance"
            aitchison_distance = pd.Series(aitchison_distance, index=index)

    return aitchison_distance

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
    
# =========    
# Filtering
# =========
def _filter_data(
    X:pd.DataFrame,
    total_counts,
    prevalence,
    components,
    mode,
    order_of_operations:list=["total_counts", "prevalence", "components"],
    interval_type="closed",
    ):

    check_compositional(X, acceptable_dimensions=2)
    assert_acceptable_arguments(query=order_of_operations,target=["total_counts", "prevalence", "components"], operation="le")
    assert_acceptable_arguments(query=[mode],target=["highpass", "lowpass"], operation="le")
    assert_acceptable_arguments(query=[interval_type],target=["closed", "open"], operation="le")


    def _get_elements(data,tol,operation):
        return data[lambda x: operation(x,tol)].index

    def _filter_total_counts(X, tol, operation):
        data = X.sum(axis=1)
        return X.loc[_get_elements(data, tol, operation),:]

    def _filter_prevalence(X, tol, operation):
        conditions = [
            isinstance(tol, float),
            0.0 < tol <= 1.0,
        ]

        if all(conditions):
            tol = round(X.shape[0]*tol)
        data = (X > 0).sum(axis=0)
        assert tol <= X.shape[0], "If prevalence is an integer ({}), it cannot be larger than the number of samples ({}) in the index".format(tol, X.shape[0])
        return X.loc[:,_get_elements(data, tol, operation)]

    def _filter_components(X, tol, operation):
        data = (X > 0).sum(axis=1)
        return X.loc[_get_elements(data, tol, operation),:]

    if interval_type == "closed":
        operations = {"highpass":operator.ge, "lowpass":operator.le}
    if interval_type == "open":
        operations = {"highpass":operator.gt, "lowpass":operator.lt}

    # Defaults
    if mode == "highpass":
        if components is None:
            components = 0
        if total_counts is None:
            total_counts = 0
        if prevalence is None:
            prevalence = 0
    if mode == "lowpass":
        if components in {None, np.inf}:
            components = X.shape[1]
        if total_counts in {None, np.inf}:
            total_counts = np.inf
        if prevalence in {None, np.inf}:
            prevalence = X.shape[0]
            
    functions = dict(zip(["total_counts", "prevalence", "components"], [_filter_total_counts, _filter_prevalence, _filter_components]))
    thresholds = dict(zip(["total_counts", "prevalence", "components"], [total_counts, prevalence, components]))

    for strategy in order_of_operations:
        tol = thresholds[strategy]
        if tol is not None:
            X = functions[strategy](X=X,tol=tol, operation=operations[mode])

    return X
        
def filter_data_highpass(
    X:pd.DataFrame,
    minimum_total_counts=1,
    minimum_prevalence=1,
    minimum_components=1,
    order_of_operations:list=["minimum_total_counts", "minimum_prevalence", "minimum_components"],
    interval_type="closed",
    ):

    """
    # Description
    Highpass filter compositional table to include data higher than a minimum
    
    # Parameters
        * X: pd.DataFrame or 2D np.array of compositional data (rows=compositions/samples, columns=components/features)
        * minimum_total_counts:  The minimum total counts in a composition (sum per row) (axis=0)
        * minimum_prevalence: The minimum number of compositions that must contain the components (axis=1)
        * minimum_components: The minimum number of detected components (axis=0)
        * order_of_operations: Order of filtering scheme.  Choose between: ["minimum_total_counts", "minimum_prevalence", "minimum_components"]

    Adapted from the following source:
    * https://github.com/jolespin/soothsayer
 
    """
    assert_acceptable_arguments(query=order_of_operations,target=["minimum_total_counts", "minimum_prevalence", "minimum_components"], operation="le")
        
    order_of_operations = list(map(lambda x: "_".join(x.split("_")[1:]), order_of_operations))

    return _filter_data(
        X=X,
        total_counts=minimum_total_counts,
        prevalence=minimum_prevalence,
        components=minimum_components,
        mode="highpass",
        order_of_operations=order_of_operations,
        interval_type=interval_type,
        )

# def filter_data_lowpass(
#     X:pd.DataFrame,
#     maximum_total_counts=np.inf,
#     maximum_prevalence=np.inf,
#     maximum_components=np.inf,
#     order_of_operations:list=["maximum_total_counts", "maximum_prevalence", "maximum_components"],
#     interval_type="closed",
#     ):

#     """
#     # Description
#     Lowpass filter compositional table to include data lower than a maximum
    
#     # Parameters
#         * X: pd.DataFrame or 2D np.array of compositional data (rows=compositions/samples, columns=components/features)
#         * maximum_total_counts:  The maximum total counts in a composition (sum per row) (axis=0)
#         * maximum_prevalence: The maximum number of compositions that must contain the components (axis=1)
#         * maximum_components: The maximum number of detected components (axis=0)
#         * order_of_operations: Order of filtering scheme.  Choose between: ["maximum_total_counts", "maximum_prevalence", "maximum_components"]

#     Adapted from the following source:
#     * https://github.com/jolespin/soothsayer
 
#     """

# ========
# Plotting
# ========
# Plot compositional data
@check_packages(["matplotlib", "seaborn"])
def plot_compositions(
    X:pd.DataFrame,
    colors:pd.Series=None, #"evenness"
    classes:pd.Series=None,
    class_colors:pd.Series=None,
    sizes:pd.Series=28,
    markers:pd.Series=None,
    sample_labels:dict=None,

    horizontal_lines:list=[],
    vertical_lines:list=[],
    
    # cbar=True,
    continuous_palette="gist_heat_r",
    color_kde_1d="black",
    color_line="black",
    marker_border_color="white",

    figsize=(8,5),
    title=None,
    style="seaborn-white",

    show_xgrid=True,
    show_ygrid=True,
    show_kde_1d=True,
    show_kde_2d=True,
    show_legend=True,

    xlabel=None,
    ylabel=None,
    legend_kws=dict(),
    legend_title=None,
    
    title_kws=dict(),
    legend_title_kws=dict(),
    axis_label_kws=dict(),
    annot_kws=dict(),
    line_kws=dict(),
    kde_1d_kws=dict(),
    kde_2d_kws=dict(),
    rug_kws=dict(),
    # cbar_kws=dict(),
    
    log_scale=False,
    pad_title=0.01, 
    xmin=0, 
    ymin=0, 
    vmin=None,
    vmax=None,

    checks=True,

    **scatter_kws,

    ):
    """
    Plot compositions of total counts (x-axis) vs. number of detected components (y-axis)
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_hex, Normalize 
    from matplotlib.scale import LogScale
    import seaborn as sns
        
    # Defaults
    _title_kws = {"fontsize":14, "fontweight":"bold", "y":1 + pad_title}
    _title_kws.update(title_kws)
    _legend_kws = {'fontsize': 10}#, 'loc': 'center left', 'bbox_to_anchor': (1, 0.5)}
    _legend_kws.update(legend_kws)
    _legend_title_kws = {"size":12, "weight":"bold"}
    _legend_title_kws.update(legend_title_kws)
    _axis_label_kws = {"fontsize":14}
    _axis_label_kws.update(axis_label_kws)

    
    _kde_1d_kws = {"alpha":0.618} # Rug takes too long when theres a lot points
    _kde_1d_kws.update(kde_1d_kws)
    _rug_kws = {"height":0.15, "clip_on":False, "alpha":0.618}
    _rug_kws.update(rug_kws)
        
    _kde_2d_kws = { "alpha":0.618, "fill":True, "levels":6}
    _kde_2d_kws.update(kde_2d_kws)
    _line_kws = {"linewidth":1.1618, "linestyle":"--", "alpha":1.0, "color":color_line}
    _line_kws.update(line_kws)
    _annot_kws = {}
    _annot_kws.update(annot_kws)
    _scatter_kws={"edgecolor":marker_border_color, "linewidths":0.618, "alpha":0.618}
    _scatter_kws.update(scatter_kws)
        
    # Data
    X = X.fillna(0)
    if checks:
        check_compositional(X, acceptable_dimensions={2})
        assert np.all(X == X.astype(int)), "X must be integer data and should not be closure transformed"
        
    # Total number of counts
    sample_to_totalcounts = X.sum(axis=1)
        
    remove_samples = sample_to_totalcounts == 0
    if np.any(remove_samples):
        warnings.warn("Removing the following observations because depth = 0: {}".format(remove_samples.index[remove_samples]))
        sample_to_totalcounts = sample_to_totalcounts[~remove_samples]

    samples = sample_to_totalcounts.index
        
    # Number of detected components
    sample_to_ncomponents = number_of_components(X.loc[samples], checks=False)

    # Number of samples and classes
    number_of_samples = sample_to_ncomponents.size
    number_of_classes = 0
        
    # Marker size
    if not isinstance(sizes, pd.Series):
        sizes = pd.Series([sizes]*number_of_samples, index=samples)
    assert np.all(sizes.notnull())

    # Plotting Data
    df_data = pd.DataFrame([
        sample_to_totalcounts,
        sample_to_ncomponents,
        sizes,
        ], index=["x","y","sizes"],
     ).T
    for field in ["x","sizes"]:
        df_data[field] = df_data[field].astype(float)
    for field in ["y"]:
        df_data[field] = df_data[field].astype(int)

    # Default color
    if all([
        colors is None,
        classes is None,
        class_colors is None,
        ]):
        colors = color_kde_1d
        
    # Colors from pd.Series
    if colors is not None:
        assert classes is None, "If `colors` are provided then `classes` and `class_colors` cannot be provided"
        assert class_colors is None, "If `colors` are provided then `classes` and `class_colors` cannot be provided"
        if not isinstance(colors, pd.Series):
            colors = pd.Series([colors]*number_of_samples, index=samples)
        assert np.all(colors.notnull())

        # Custom colors
        try:        
            colors = colors.map(to_hex)
            df_data["custom_colors"] = colors
        # Continuous colors
        except ValueError:
            if vmin is None:
                vmin = colors.min()
            if vmax is None:
                vmax = colors.max()
            df_data["continuous_colors"] = colors
            
        if legend_title is None:
            legend_title = colors.name
        
    # Colors from classes
    if classes is not None:
        assert class_colors is not None, "`class_colors` is required for using `classes`"
        classes = pd.Series(classes)
        class_colors = pd.Series(class_colors)
        assert np.all(classes.index == samples), "`classes` must be a pd.Series with the same index ordering as `X.index`"
        assert np.all(classes.map(lambda x: x in class_colors)), "Classes in `class` must have a color in `class_colors`"
            
        if colors is not None:
            warnings.warn("`colors` will be ignored and superceded by class_colors and classes")
            
        if legend_title is None:
            legend_title = classes.name

        # Add to plotting data
        df_data["classes"] = classes

        # Number of classes
        number_of_classes = df_data["classes"].nunique()

        # Markers
        if markers is not None:
            markers = pd.Series(markers)
            assert set(markers.index) == set(classes.unique()), "`markers` must be a pd.Series with all the classes from `classes`"

    # Plotting
    with plt.style.context(style):
        # Create new figure and don't overwrite existing figure
        plt.figure()

        # Logscale
        logscale10 = LogScale(axis=0,base=10)

        # Simple scatter plot
        conditions = [
            not show_kde_1d,
            not show_kde_2d,
        ]
        if all(conditions):
            if "classes" in df_data.columns:
                ax_scatter = sns.scatterplot(data=df_data, x="x", y="y", sizes="sizes", hue="classes", hue_order=class_colors.index.tolist(), palette=class_colors.values.tolist(), marker=markers,  **_scatter_kws)
            if "continuous_colors" in df_data.columns:
                ax_scatter = sns.scatterplot(data=df_data, x="x", y="y", sizes="sizes", hue="continuous_colors", hue_norm=(vmin,vmax), palette=continuous_palette, **_scatter_kws)
            if "custom_colors" in df_data.columns:
                ax_scatter = sns.scatterplot(data=df_data, x="x", y="y", sizes="sizes", c=df_data["custom_colors"],  **_scatter_kws)
                
            if log_scale:
                ax_scatter.set_xscale(logscale10)
                
            fig = plt.gcf()
            output = (fig, ax_scatter)

        # Distributions
        else:
            # 1D Only
            conditions = [
                show_kde_1d,
                not show_kde_2d,
            ]
            if all(conditions):
                if "classes" in df_data.columns:
                    g = sns.jointplot(data=df_data, kind="scatter", x="x", y="y", sizes="sizes", hue="classes", hue_order=class_colors.index.tolist(), palette=class_colors.values.tolist(), marker=markers, **_scatter_kws)
                    g.plot_marginals(sns.rugplot, data=df_data, hue="classes", hue_order=class_colors.index.tolist(), palette=class_colors.values.tolist(), **_rug_kws)

                if "continuous_colors" in df_data.columns:
                    g = sns.jointplot(data=df_data, kind="kde", x="x", y="y",  color=color_kde_1d, alpha=0, **{k:v  for k,v in _kde_2d_kws.items() if k != "alpha"})
                    g.plot_joint(sns.scatterplot, data=df_data, hue="continuous_colors", hue_norm=(vmin,vmax),palette=continuous_palette, **_scatter_kws)
                    g.plot_marginals(sns.rugplot, data=df_data, hue="continuous_colors", hue_norm=(vmin,vmax), palette=continuous_palette, **_rug_kws)

                if "custom_colors" in df_data.columns:
                    g = sns.jointplot(data=df_data, kind="kde", x="x", y="y",  color=color_kde_1d, alpha=0, **{k:v  for k,v in _kde_2d_kws.items() if k != "alpha"})
                    g.plot_joint(sns.scatterplot, data=df_data, c=df_data["custom_colors"], **_scatter_kws)
                    g.plot_marginals(sns.rugplot, data=df_data, c=color_kde_1d, **_rug_kws)
                    
                ax_scatter = g.ax_joint
                
                if log_scale:
                    ax_scatter.set_xscale(logscale10)
                    g.ax_marg_x.set_xscale(logscale10)
                    
                fig = plt.gcf()
                output = (fig, g, df_data)

            # 2D Only
            conditions = [
                not show_kde_1d,
                show_kde_2d,
            ]
            
            if all(conditions):
                if "classes" in df_data.columns:
                    ax_scatter = sns.scatterplot(data=df_data, x="x", y="y", sizes="sizes", hue="classes", hue_order=class_colors.index.tolist(), palette=class_colors.values.tolist(), marker=markers, **_scatter_kws)
                    sns.kdeplot(data=df_data,  x="x", y="y", sizes="sizes", hue="classes", hue_order=class_colors.index.tolist(), palette=class_colors.values.tolist(), marker=markers, ax=ax_scatter,zorder=0,  **_kde_2d_kws)
                if "continuous_colors" in df_data.columns:
                    ax_scatter = sns.scatterplot(data=df_data, x="x", y="y", sizes="sizes", hue="continuous_colors",  hue_norm=(vmin,vmax),palette=continuous_palette, **_scatter_kws)
                    sns.kdeplot(data=df_data,  x="x", y="y", color=color_kde_1d, ax=ax_scatter, zorder=0, **_kde_2d_kws)
                if "custom_colors" in df_data.columns:
                    ax_scatter = sns.scatterplot(data=df_data, x="x", y="y", sizes="sizes", c=df_data["custom_colors"], **_scatter_kws)
                    sns.kdeplot(data=df_data,  x="x", y="y", color=color_kde_1d, ax=ax_scatter, zorder=0, **_kde_2d_kws)
                    
                if log_scale:
                    ax_scatter.set_xscale(logscale10)
                    
                fig = plt.gcf()
                output = (fig, ax_scatter, df_data)

            # 1D & 2D
            conditions = [
                show_kde_1d,
                show_kde_2d,
            ]
            if all(conditions):
                if "classes" in df_data.columns:
                    g = sns.jointplot(data=df_data, kind="scatter", x="x", y="y", sizes="sizes", hue="classes", hue_order=class_colors.index.tolist(), palette=class_colors.values.tolist(), marker=markers,  **_scatter_kws)
                    g.plot_joint(sns.kdeplot, hue="classes", hue_order=class_colors.index.tolist(), palette=class_colors.values.tolist(), zorder=0, **_kde_2d_kws)
                    g.plot_marginals(sns.rugplot, data=df_data, hue="classes", hue_order=class_colors.index.tolist(), palette=class_colors.values.tolist(), **_rug_kws)

                if "continuous_colors" in df_data.columns:
                    g = sns.jointplot(data=df_data, kind="kde", x="x", y="y",  color=color_kde_1d, **_kde_2d_kws)#, sizes=1e-3, alpha=0, **{k:v  for k,v in _scatter_kws.items() if k != "alpha"})
                    g.plot_joint(sns.scatterplot, data=df_data, hue="continuous_colors", hue_norm=(vmin,vmax), palette=continuous_palette,  **_scatter_kws)
                    g.plot_marginals(sns.rugplot, data=df_data, hue="continuous_colors", hue_norm=(vmin,vmax), palette=continuous_palette, **_rug_kws)

                if "custom_colors" in df_data.columns:
                    g = sns.jointplot(data=df_data, kind="kde", x="x", y="y",  color=color_kde_1d, **_kde_2d_kws)#, sizes=1e-3, alpha=0, **{k:v  for k,v in _scatter_kws.items() if k != "alpha"})
                    g.plot_joint(sns.scatterplot, data=df_data, c=df_data["custom_colors"], **_scatter_kws)
                    g.plot_marginals(sns.rugplot, data=df_data, c=color_kde_1d, **_rug_kws)


                ax_scatter = g.ax_joint
                if log_scale:
                    ax_scatter.set_xscale(logscale10)
                    g.ax_marg_x.set_xscale(logscale10)
                    
                fig = plt.gcf()
                output = (fig, g, df_data)

        # Horizontal lines
        if horizontal_lines:
            if not hasattr(horizontal_lines, "__iter__"):
                horizontal_lines = [horizontal_lines]
            for line in horizontal_lines:
                ax_scatter.axhline(line, **_line_kws)
        if vertical_lines:
            if not hasattr(vertical_lines, "__iter__"):
                vertical_lines = [vertical_lines]
            for line in vertical_lines:
                ax_scatter.axvline(line, **_line_kws)

        # Labels
        if xlabel is None:
            xlabel = "Total Counts"
            if log_scale:
                xlabel = "%s [log$_{10}$]"%(xlabel)
        if ylabel is None:
            ylabel = "Number of Components"

        # Legend
        if show_legend:
            conditions = [
                number_of_classes > 1,
                "continuous_colors" in df_data.columns,
            ]
            if any(conditions):
                if legend_title is None:
                    if colors is not None:
                        legend_title = colors.name
                        
                if legend_title is not None:
                    legend = ax_scatter.get_legend()
                    if legend is not None:
                        legend.set_title(legend_title, prop=_legend_title_kws)
                        
        # Annotations
        if sample_labels is not None:
            assert hasattr(sample_labels, "__iter__"), "sample_labels must be an iterable or a mapping between sample and label"
            
            if isinstance(sample_labels, (Mapping, pd.Series)):
                sample_labels = dict(sample_labels)
            else:
                sample_labels = dict(zip(sample_labels, sample_labels))
                
            for k,v in sample_labels.items():
                if k not in df_data.index:
                    assert k in X.index, ("{} is not in X.index".format(k))
                    warnings.warn("{} is not in X.index after removing empty compositions".format(k))
                else:
                    x, y = df_data.loc[k,["x","y"]]
                    ax_scatter.text(x=x, y=y, s=v, **_annot_kws)

        if xmin is not None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ax_scatter.set_xlim(xmin, max(ax_scatter.get_xlim()))
       
        if ymin is not None:
            ax_scatter.set_ylim(xmin, max(ax_scatter.get_ylim()))

        ax_scatter.set_xlabel(xlabel, **_axis_label_kws)
        ax_scatter.set_ylabel(ylabel, **_axis_label_kws)
        ax_scatter.xaxis.grid(show_xgrid)
        ax_scatter.yaxis.grid(show_ygrid)

        
        # Figure
        fig.set_size_inches(figsize)
        if title is not None:
            fig.suptitle(title, **_title_kws)

        return output
    
# Plot prevalence of components
@check_packages(["matplotlib"])
def plot_prevalence(
    X:pd.DataFrame, 
    minimum_count=1, 
    component_type="Components",
    color="black",
    classes:pd.Series=None,
    class_colors:pd.Series=None,
    marker_border_color="white",

    figsize=(13,5),
    title=None,
    style="seaborn-white",

    show_prevalence=[1,2,0.5,1.0],
    show_xgrid=True,
    show_ygrid=True,
    show_legend=True,
    ylabel=None,
    xlabel="Prevalence",
    legend_kws=dict(),
    legend_title=None,
    
    title_kws=dict(),
    legend_title_kws=dict(),
    axis_label_kws=dict(),
    line_kws=dict(),
    fig_kws=dict(),
    scatter_kws=dict(),
    fill_kws=dict(),

    number_of_component_ticks=10,
    number_of_prevalence_ticks=25,
    
    fill=True,

    ax=None,
    checks=True,
    ):

    import matplotlib.pyplot as plt

    _fig_kws = {"figsize":figsize}
    _fig_kws.update(fig_kws)
    _title_kws = {"fontsize":14, "fontweight":"bold"}
    _title_kws.update(title_kws)
    _legend_kws = {'fontsize': 10}#, 'loc': 'center left', 'bbox_to_anchor': (1, 0.5)}
    _legend_kws.update(legend_kws)
    _legend_title_kws = {"size":12, "weight":"bold"}
    _legend_title_kws.update(legend_title_kws)
    _axis_label_kws = {"fontsize":14}
    _axis_label_kws.update(axis_label_kws)
    _line_kws = {"linewidth":1, "linestyle":"-", "alpha":1.0}
    _line_kws.update(line_kws)
    _scatter_kws = {"linewidth":1, "alpha":0.95}
    _scatter_kws.update(scatter_kws)
    _fill_kws = {"alpha":0.1618}
    _fill_kws.update(fill_kws)


    def _format_prevalence_distribution(prevalence, n):
        # Number of attributes for tolerance
        prevalence_to_ncomponents = dict()
        for tol in range(1,n+1):
            prevalence_to_ncomponents[tol] = prevalence.map(lambda x: x >= tol).sum()
    
        # Prevalence
        prevalence_to_ncomponents = pd.Series(prevalence_to_ncomponents).sort_values(ascending=True)
        return prevalence_to_ncomponents

    def _format_prevalence_query(show_prevalence, n):
        if not hasattr(show_prevalence, "__iter__"):
            show_prevalence = [show_prevalence]
        tmp = list()
        for x in show_prevalence:
            if isinstance(x, float):
                assert x >= 0.0
                assert x <= 1.0
                x = round(x * n)
                if x not in tmp:
                    tmp.append(x)
            else:
                if x not in tmp:
                    tmp.append(x)
        return tmp

    with plt.style.context(style):

        if ax is  None:
            fig, ax = plt.subplots(**_fig_kws)
        else:
            fig = plt.gcf()
            
        # Dimensions
        if checks:
            check_compositional(X, acceptable_dimensions={2})
        n,m = X.shape
        samples = X.index

        prevalence_values = list()
        # Class-specific prevalence
        if classes is not None:
            assert class_colors is not None, "`class_colors` is required for using `classes`"
            classes = pd.Series(classes)
            class_colors = pd.Series(class_colors)
            assert np.all(classes.index == samples), "`classes` must be a pd.Series with the same index ordering as `X.index`"
            assert np.all(classes.map(lambda x: x in class_colors)), "Classes in `class` must have a color in `class_colors`"
                
            # Number of classes
            number_of_classes = classes.nunique()

            if legend_title is None:
                legend_title = classes.name

            class_to_prevalence = dict()
            for id_class, X_class in X.groupby(classes, axis=0):
                prevalence = prevalence_of_components(X_class, minimum_count=minimum_count, checks=False)
                prevalence_to_ncomponents = _format_prevalence_distribution(prevalence, n=X_class.shape[0])
                ax.plot(prevalence_to_ncomponents.index, prevalence_to_ncomponents, color=class_colors[id_class])
                ax.scatter(prevalence_to_ncomponents.index, prevalence_to_ncomponents, color=class_colors[id_class], label=id_class, edgecolor=marker_border_color, **_scatter_kws)
                class_to_prevalence[id_class] = prevalence_to_ncomponents
                prevalence_values += prevalence_to_ncomponents.tolist()

                if show_prevalence is not None:
                    prevalence_values_to_show = _format_prevalence_query(show_prevalence, n=X_class.shape[0])

                    for x_pos in sorted(prevalence_values_to_show):
                        y_pos = prevalence_to_ncomponents.loc[x_pos]
                        label = "Prevalence({}) = {} {}".format(x_pos, y_pos, component_type)
                        ax.plot([x_pos,x_pos], [0,y_pos], color=class_colors[id_class],  label=label, **_line_kws)
                        ax.plot([0 ,x_pos], [y_pos,y_pos], color=class_colors[id_class],**_line_kws)
                        if fill:
                            ax.fill_between([0,x_pos], [y_pos,y_pos], color=class_colors[id_class], **_fill_kws)
                
            output = (fig, ax, pd.DataFrame(class_to_prevalence).T)

        # Global prevalence
        else:
            prevalence = prevalence_of_components(X, minimum_count=minimum_count, checks=False)
            prevalence_to_ncomponents = _format_prevalence_distribution(prevalence, n=n)
            ax.plot( prevalence_to_ncomponents.index, prevalence_to_ncomponents, color=color)
            ax.scatter(prevalence_to_ncomponents.index, prevalence_to_ncomponents, color=color, edgecolor=marker_border_color, **_scatter_kws) 
            prevalence_values += prevalence_to_ncomponents.tolist()

            # Specific prevalence values
            if show_prevalence is not None:
                prevalence_values_to_show = _format_prevalence_query(show_prevalence, n=n)

                for x_pos in sorted(prevalence_values_to_show):
                    y_pos = prevalence_to_ncomponents.loc[x_pos]
                    label = "Prevalence({}) = {} {}".format(x_pos, y_pos, component_type)
                    ax.plot([x_pos,x_pos], [0,y_pos], color=color,  label=label, **_line_kws)
                    ax.plot([0 ,x_pos], [y_pos,y_pos], color=color,**_line_kws)
                    if fill:
                        ax.fill_between([0,x_pos], [y_pos,y_pos], color=color, **_fill_kws)

            output = (fig, ax, prevalence_to_ncomponents)
            
        prevalence_values = sorted(set(prevalence_values))

        if show_legend:
            ax.legend(**_legend_kws)
            if legend_title is not None:
                ax.legend_.set_title(legend_title, prop=_legend_title_kws)

        # Set limits
        ax.set_xlim((0,n))
        ax.set_ylim((1,m))
        
        prevalence_ticks = np.linspace(1,n,min(number_of_prevalence_ticks,n)).astype(int)
        ax.set_xticks(prevalence_ticks)
        ax.set_xticklabels(ax.get_xticks())

        component_ticks = np.linspace(1,m,min(number_of_component_ticks,m)).astype(int)
        ax.set_yticks(component_ticks)
        ax.set_yticklabels(ax.get_yticks())

        # Axis Labels

        ax.set_xlabel(xlabel, **_axis_label_kws)
        if ylabel is None:
            ylabel = "Number of {}".format(component_type)
            
        ax.set_ylabel(ylabel,  **_axis_label_kws)

        # Show grids
        ax.xaxis.grid(show_xgrid)
        ax.yaxis.grid(show_ygrid)

        # Title
        if title is not None:
            ax.set_title(title, **_title_kws)
            
        return output