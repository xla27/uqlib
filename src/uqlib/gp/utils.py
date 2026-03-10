"""
Gaussian Process Utilities Module

This module contains shared utility functions for all Gaussian Process variants
(gpr.py, heterogpr.py, sparsegpr.py) to eliminate code duplication.

Functions are organized by category:
1. Data Preparation Utilities
2. Hyperparameter Bounds Utilities
3. Latin Hypercube Sampling Utilities
4. Prediction Transformation Utilities
5. Optimization Utilities
6. Cholesky Operations Utilities
"""

import numpy as np
from scipy.stats import qmc
from scipy.optimize import minimize
from scipy.linalg import cho_solve, cholesky
import multiprocessing
from itertools import repeat


# =====================================================================
# 1. DATA PREPARATION UTILITIES
# =====================================================================

def standardize_targets(y):
    """
    Compute mean and standard deviation of target values.
    
    This standardization is applied to the output before training to:
    - Improve numerical stability during optimization
    - Make hyperparameter bounds more meaningful
    - Normalize learning process
    
    Parameters
    ----------
    y : array-like, shape (n_samples,) or (n_samples, 1)
        Target values
    
    Returns
    -------
    mean : float
        Mean of target values
    std : float
        Standard deviation of target values
    
    Examples
    --------
    >>> y = np.array([1., 2., 3., 4., 5.])
    >>> mean, std = standardize_targets(y)
    >>> print(mean, std)
    3.0 1.414...
    """
    y = np.asarray(y).flatten()
    return np.mean(y), np.std(y)


def normalize_targets(y, mean, std):
    """
    Normalize (z-score) target values using precomputed mean and standard deviation.
    
    Normalization formula: y_norm = (y - mean) / std
    
    This is applied after standardization to ensure:
    - Zero mean
    - Unit variance
    - Better conditioning for numerical algorithms
    
    Parameters
    ----------
    y : array-like, shape (n_samples,) or (n_samples, 1)
        Target values to normalize
    mean : float
        Mean computed from training data (via standardize_targets)
    std : float
        Standard deviation computed from training data (via standardize_targets)
    
    Returns
    -------
    y_norm : ndarray, shape (n_samples, 1)
        Normalized target values with zero mean and unit variance
    
    Notes
    -----
    This function is used consistently across all GP implementations to ensure
    numerical stability and reproducibility.
    
    Examples
    --------
    >>> y = np.array([1., 2., 3., 4., 5.])
    >>> mean, std = standardize_targets(y)
    >>> y_norm = normalize_targets(y, mean, std)
    >>> np.allclose(np.mean(y_norm), 0.0)
    True
    >>> np.allclose(np.std(y_norm), 1.0)
    True
    """
    y = np.asarray(y)
    ndata = y.shape[0]
    y_norm = (y.reshape(ndata, 1) - mean * np.ones((ndata, 1))) / std
    return y_norm


def initialize_gp_metadata(gp, x, y):
    """
    Initialize common metadata attributes used across all GP variants.
    
    This function centralizes the initialization of fundamental GP attributes
    to ensure consistency across different GP implementations. It sets:
    - Input data (x) and output data (y)
    - Problem dimensions (dim, ndata)
    - Normalized targets (y_norm)
    
    Parameters
    ----------
    gp : object
        GP object to initialize. Must have attributes:
        - y_train_mean, y_train_std (set via standardize_targets)
    x : ndarray, shape (n_samples, n_features)
        Training input data (should be in [0,1]^d)
    y : ndarray, shape (n_samples,) or (n_samples, 1)
        Training output data
    
    Returns
    -------
    None
        Modifies gp object in-place, setting:
        - gp.x : training inputs
        - gp.y : training outputs
        - gp.dim : number of input dimensions
        - gp.ndata : number of training samples
        - gp.y_norm : normalized targets
    
    Notes
    -----
    This function must be called after standardize_targets() and
    before normalize_targets() can be used with gp object.
    
    Examples
    --------
    >>> gp = SomeGPClass(kernel)
    >>> gp.y_train_mean, gp.y_train_std = standardize_targets(y)
    >>> initialize_gp_metadata(gp, x, y)
    >>> assert gp.ndata == x.shape[0]
    >>> assert gp.dim == x.shape[1]
    """
    gp.x = x
    gp.y = y
    gp.dim = x.shape[1]
    gp.ndata = x.shape[0]
    gp.y_norm = normalize_targets(y, gp.y_train_mean, gp.y_train_std)


# =====================================================================
# 2. HYPERPARAMETER BOUNDS UTILITIES
# =====================================================================

def build_hyperparameter_bounds(kernel_bounds_list, extra_bounds_dict=None):
    """
    Construct hyperparameter bounds from kernels and optional extra parameters.
    
    This utility function provides a flexible way to build bounds for different
    GP variants:
    - Standard GP: only kernel bounds
    - Heteroscedastic GP: kernel_f + kernel_g + mu_0 + LAMBDA bounds
    - Sparse GP: kernel bounds + noise_level bounds
    
    Parameters
    ----------
    kernel_bounds_list : list of ndarray
        List of kernel bounds arrays. Each element should be shape (n_hyp, 2)
        containing [lower_bound, upper_bound] for each hyperparameter.
        Example: [kernel_f.bounds, kernel_g.bounds]
    
    extra_bounds_dict : dict, optional
        Dictionary mapping parameter names to (lower, upper) bound tuples.
        Keys are used only for ordering (alphabetically sorted).
        Example: {'mu_0': (-10, 0), 'noise': (1e-6, 1e-2)}
    
    Returns
    -------
    hyp_lw : ndarray
        Array of lower bounds for all hyperparameters
    hyp_up : ndarray
        Array of upper bounds for all hyperparameters
    
    Notes
    -----
    The function concatenates bounds in the order:
    1. All kernel bounds (in order of kernel_bounds_list)
    2. Extra parameter bounds (alphabetically sorted by key)
    
    This ensures consistent ordering across multiple runs.
    
    Examples
    --------
    # Standard GP: only kernel bounds
    >>> kernel = SomeKernel()
    >>> hyp_lw, hyp_up = build_hyperparameter_bounds([kernel.bounds])
    
    # Heteroscedastic GP: multiple kernels + extra parameters
    >>> hyp_lw, hyp_up = build_hyperparameter_bounds(
    ...     [kernel_f.bounds, kernel_g.bounds],
    ...     {'mu_0': (-10, 0), 'lambda_i': (0, 10)}
    ... )
    
    # Sparse GP: kernel + noise bounds
    >>> hyp_lw, hyp_up = build_hyperparameter_bounds(
    ...     [kernel.bounds],
    ...     {'noise': (1e-6, 1e-2)}
    ... )
    """
    hyp_lw = np.array([])
    hyp_up = np.array([])
    
    # Accumulate kernel bounds in order
    for bounds in kernel_bounds_list:
        bounds = np.asarray(bounds)
        hyp_lw = np.append(hyp_lw, bounds[:, 0])
        hyp_up = np.append(hyp_up, bounds[:, 1])
    
    # Add extra bounds in sorted order
    if extra_bounds_dict is not None:
        for param_name in extra_bounds_dict.keys():
            lb, ub = extra_bounds_dict[param_name]
            hyp_lw = np.append(hyp_lw, lb)
            hyp_up = np.append(hyp_up, ub)
    
    return hyp_lw, hyp_up


# =====================================================================
# 3. LATIN HYPERCUBE SAMPLING UTILITIES
# =====================================================================

def generate_lhs_points(bounds, multistart, nproc):
    """
    Generate Latin Hypercube sampled initial points for multistart optimization.
    
    This function creates a set of well-distributed initial points for
    multistart optimization of hyperparameters. It uses Latin Hypercube Sampling
    (LHS) to ensure good coverage of the hyperparameter space, then distributes
    these points across multiple processors.
    
    The LHS method is superior to random sampling because it:
    - Ensures no clustering of initial points
    - Provides better coverage of the parameter space
    - Improves chances of finding the global optimum
    
    Points are distributed as evenly as possible:
    - Each processor gets approximately multistart / nproc points
    - Remaining points are distributed to the first processors
    
    Parameters
    ----------
    bounds : list of tuples or list of lists
        Hyperparameter bounds, list of (lower, upper) tuples.
        Example: [(0.1, 10), (0.01, 1), (1e-3, 1)]
    
    multistart : int
        Total number of starting points to generate
    
    nproc : int
        Number of processors for parallel optimization.
        Initial points will be distributed across these processors.
    
    Returns
    -------
    init_point_list : list of ndarray
        List of length nproc, where each element is an array of shape
        (n_starts_for_processor_i, n_hyperparameters) containing the
        initial points assigned to processor i.
    
    multistart_vec : list of int
        List of length nproc, where each element is the number of starting
        points assigned to processor i.
    
    Notes
    -----
    The function uses scipy.stats.qmc for Latin Hypercube Sampling.
    Random seed is NOT fixed, so different runs may produce different
    (but equally good) distributions.
    
    Examples
    --------
    >>> bounds = [(0.1, 10), (1e-3, 1), (-5, 5)]
    >>> init_pts, starts_vec = generate_lhs_points(bounds, multistart=100, nproc=4)
    >>> len(init_pts)
    4
    >>> sum(starts_vec)
    100
    >>> init_pts[0].shape  # Each processor gets different numbers
    (25, 3)
    
    Notes
    -----
    Typical usage in model_fitting functions:
    
    >>> bounds = list(zip(hyp_lw, hyp_up))
    >>> init_point_list, multistart_vec = generate_lhs_points(
    ...     bounds, multistart, gp.nproc
    ... )
    >>> # Then distribute to workers via multiprocessing
    """
    # Extract lower and upper bounds
    hyp_lw = np.array([b[0] for b in bounds])
    hyp_up = np.array([b[1] for b in bounds])
    
    # Calculate number of starts per processor
    multistart_vec = [int(multistart / nproc) for _ in range(nproc)]
    
    # Generate LHS samples in unit hypercube [0,1]
    sampler = qmc.LatinHypercube(d=len(bounds))
    initial_points = sampler.random(n=multistart)
    
    # Scale to [hyp_lw, hyp_up]
    initial_points = qmc.scale(initial_points, hyp_lw, hyp_up)
    
    # Distribute initial points across processors
    init_point_list = [
        initial_points[i * (int(multistart / nproc)) : (i + 1) * (int(multistart / nproc))]
        for i in range(nproc)
    ]
    
    # Redistribute any remaining initial points to first processors
    assigned = nproc * int(multistart / nproc)
    if assigned < multistart:
        for i in range(multistart - assigned):
            init_point_list[i] = np.vstack((init_point_list[i], 
                                           initial_points[assigned + i, :]))
    
    return init_point_list, multistart_vec


# =====================================================================
# 4. PREDICTION TRANSFORMATION UTILITIES
# =====================================================================

def destandardize_predictions(y_mean, y_train_mean, y_train_std, 
                              y_var=None, standardized=False):
    """
    Apply de-standardization transformation to GP predictions.
    
    When a GP is trained on standardized targets (zero mean, unit variance),
    predictions are returned in the same standardized space. This function
    transforms predictions back to the original data scale.
    
    De-standardization formulas:
    - Mean: y_pred = y_mean_std * y_train_std + y_train_mean
    - Variance: y_var_pred = y_var_std * y_train_std^2
    
    Parameters
    ----------
    y_mean : ndarray
        Predicted mean in standardized space
    
    y_train_mean : float
        Mean of training targets (from standardize_targets)
    
    y_train_std : float
        Standard deviation of training targets (from standardize_targets)
    
    y_var : ndarray, optional
        Predicted variance/covariance in standardized space.
        If None, only y_mean is transformed.
    
    standardized : bool, default=False
        If True, return predictions in standardized space (no transformation).
        If False, apply de-standardization.
    
    Returns
    -------
    y_mean_out : ndarray
        Predicted mean (standardized or de-standardized based on 'standardized')
    
    y_var_out : ndarray or None
        Predicted variance (if y_var was provided)
    
    Notes
    -----
    This function is used consistently across all GP predict() methods to ensure
    consistent output handling.
    
    Examples
    --------
    # During prediction in standardized space
    >>> y_mean_std, y_var_std = gp_predict_standardized(xtest)
    
    # Transform back to original scale
    >>> y_mean, y_var = destandardize_predictions(
    ...     y_mean_std, gp.y_train_mean, gp.y_train_std, y_var_std
    ... )
    
    # Or keep in standardized space
    >>> y_mean_std, y_var_std = destandardize_predictions(
    ...     y_mean_std, gp.y_train_mean, gp.y_train_std, y_var_std,
    ...     standardized=True
    ... )
    """
    if standardized:
        # Return standardized predictions
        if y_var is not None:
            return y_mean, y_var
        else:
            return y_mean
    else:
        # De-standardize predictions to original scale
        y_mean_out = y_mean * y_train_std + y_train_mean
        
        if y_var is not None:
            y_var_out = y_var * (y_train_std ** 2)
            return y_mean_out, y_var_out
        else:
            return y_mean_out


# =====================================================================
# 5. OPTIMIZATION UTILITIES
# =====================================================================

def run_multiprocessing_optimization(optimizer_func, gp, 
                                     init_points_list, multistart_vec,
                                     extra_args=None):
    """
    Run multistart L-BFGS-B optimization in parallel across multiple processors.
    
    This function encapsulates the multiprocessing pattern used in all GP
    variants. It:
    1. Creates a process pool
    2. Distributes optimization tasks
    3. Gathers results from all workers
    4. Closes the pool
    
    The multiprocessing approach is necessary because:
    - L-BFGS-B hyperparameter optimization can get stuck in local minima
    - Multiple starting points improve chances of finding global optimum
    - Parallel execution significantly reduces wall-clock time
    
    Parameters
    ----------
    optimizer_func : callable
        The optimization function that will be called by each worker.
        Should have signature: optimizer_func(init_point, *extra_args)
    
    gp : object
        GP object passed to optimizer_func
    
    init_points_list : list of ndarray
        List of initial points for each processor (from generate_lhs_points)
    
    multistart_vec : list of int
        For GP and HeteroGP: Number of starts per processor (from generate_lhs_points)
        For SparseGP: Number of inducing points 
    
    extra_args : tuple, optional
        Additional arguments to pass to optimizer_func after gp
    
    Returns
    -------
    all_results : list
        Aggregated results from all workers, in the order produced by
        pool.starmap
    
    Notes
    -----
    This function assumes that each worker returns results in a consistent
    format. The unpacking of results depends on the specific optimizer_func.
    
    Examples
    --------
    # This is typically used inside a *_model_fitting function:
    
    >>> init_pts, starts_vec = generate_lhs_points(bounds, multistart, gp.nproc)
    >>> results = run_multiprocessing_optimization(
    ...     multistart_opt, gp, multistart, init_pts, starts_vec
    ... )
    >>> opt_funcs, opt_thetas = zip(*results)
    """
    nproc = len(multistart_vec)
    
    pool = multiprocessing.Pool(processes=nproc)
    
    # Prepare arguments for starmap
    func_args = zip(
        repeat(optimizer_func),
        multistart_vec,
        repeat(gp),
        init_points_list
    )
    
    if extra_args is not None:
        # If there are extra arguments, append them
        func_args = [(*args, *extra_args) for args in func_args]

    # Flag for greedy algorithm
    try:

        greedy = extra_args[0]

    except:

        greedy = False
    
    # Run optimization in parallel
    results = pool.starmap(multistart_opt_wrapper, func_args)
    
    pool.close()
    pool.join()

    # unpacking results
    opt_func = np.array([])
    opt_theta = np.empty((0, gp.hyp_lw.shape[0]))
    if greedy: opt_xm = []

    for n in range(gp.nproc):
        opt_func = np.append(opt_func, results[n][0])
        opt_theta = np.vstack((opt_theta, results[n][1]))
        if greedy: opt_xm.append(results[n][2])
    
    if greedy:

        return opt_func, opt_theta, opt_xm 
    
    else:
        
        return opt_func, opt_theta


def multistart_opt_wrapper(optimizer_func, multistart, gp, init_set, greedy=False):
    """
    Wrapper function for multiprocessing pool workers.
    
    This wrapper is called by each process to handle a subset of the
    multistart optimization tasks. It iterates through the assigned
    initial points and runs L-BFGS-B optimization for each.
    
    Parameters
    ----------
    optimizer_func : callable
        Objective function for scipy.optimize.minimize
    
    multistart : int
        Number of optimization starts for this process
    
    gp : object
        GP object (shared via multiprocessing)
    
    init_set : ndarray
        Initial points for this process

    greedy : bool
        Greedy algorithm for SparseGP
    
    Returns
    -------
    opt_results : tuple
        Aggregated results from all starts for this process.
        Format depends on optimizer_func return value.
    
    Notes
    -----
    This is an internal function typically called through
    run_multiprocessing_optimization. It should not be called directly.
    """

    opt_func = np.zeros(multistart)
    opt_theta = np.zeros((multistart, init_set.shape[1]))

    if greedy:
        M_INDEX = []
        init_point = np.squeeze(init_set)
    
    for i in range(multistart):

        if greedy:
            while True:
                t = np.random.randint(0, gp.ndata-1)
                if t not in M_INDEX:
                    M_INDEX.append(t)
                    break

            xm = gp.x[M_INDEX,:]
            args = (gp, xm, False)
            jac = False

        else:
            init_point = init_set[i, :]
            args = (gp, True)
            jac = True
        
        results = minimize(
            optimizer_func,
            init_point,
            args=args,
            method="L-BFGS-B",
            jac=jac,
            bounds=list(zip(gp.hyp_lw, gp.hyp_up)),
            tol=1e-7,
            options={'disp': False, 'maxfun': 10000}
        )
        
        if results.success:
            opt_func[i] = results.fun
            opt_theta[i, :] = results.x.reshape(1, -1)
        else:
            # If optimization fails, use large value to penalize
            opt_func[i] = np.nan
            if greedy:
                opt_theta[i, :] = opt_theta[i-1, :]
            else:
                opt_theta[i, :] = np.nan * np.ones(init_point.shape[0])

        if greedy:
            init_point = opt_theta[i, :]

    if greedy:

        return opt_func[-1], opt_theta[-1,:], xm
    
    else:

        return opt_func, opt_theta


# =====================================================================
# 6. CHOLESKY OPERATIONS UTILITIES
# =====================================================================

def solve_cholesky_system(K, y, tych=0.0):
    """
    Solve a system using Cholesky decomposition with optional regularization.
    
    This utility function encapsulates the common pattern of:
    1. Adding Tikhonov regularization to the matrix
    2. Computing Cholesky decomposition
    3. Solving the linear system
    
    This pattern appears throughout all GP implementations.
    
    Parameters
    ----------
    K : ndarray, shape (n, n)
        Covariance matrix to decompose
    
    y : ndarray, shape (n,) or (n, 1)
        Right-hand side vector
    
    tych : float, optional
        Tikhonov regularization parameter. Added to diagonal as:
        K + tych * I
        Default: 0.0 (no regularization)
    
    Returns
    -------
    L : ndarray, lower triangular
        Cholesky factor such that K + tych*I = L @ L.T
    
    alpha : ndarray
        Solution to (K + tych*I) @ alpha = y
    
    Notes
    -----
    Tikhonov regularization improves numerical stability when:
    - K is ill-conditioned
    - Very small eigenvalues exist
    - Floating-point errors accumulate
    
    Examples
    --------
    >>> K = some_covariance_matrix  # shape (n, n)
    >>> y = targets  # shape (n,)
    >>> L, alpha = solve_cholesky_system(K, y, tych=1e-6)
    >>> # Now: (K + 1e-6*I) @ alpha = y (approximately)
    """
    
    n = K.shape[0]
    K_reg = K + tych * np.eye(n)
    
    L = cholesky(K_reg, lower=True, overwrite_a=True, check_finite=False)
    alpha = cho_solve((L, True), y, check_finite=False)
    
    return L, alpha


def compute_cholesky_inverse(L):
    """
    Compute matrix inverse from its Cholesky factor.
    
    Given the Cholesky decomposition K = L @ L.T, this function
    efficiently computes K^{-1} using the triangular structure.
    
    Parameters
    ----------
    L : ndarray, lower triangular
        Cholesky factor from cholesky(K, lower=True)
    
    Returns
    -------
    K_inv : ndarray
        Inverse of the matrix K (where K = L @ L.T)
    
    Notes
    -----
    This is more numerically stable and efficient than np.linalg.inv(K)
    when the Cholesky factor is already available.
    
    Examples
    --------
    >>> K = some_covariance_matrix
    >>> L = cholesky(K, lower=True)
    >>> K_inv = compute_covariance_inverse(L)
    """
    
    n = L.shape[0]
    K_inv = cho_solve((L, True), np.eye(n), check_finite=False)
    
    return K_inv


def kl_div_normals(m1, K1, m2, K2):
    """
    Compute exact Kullback-Leibler divergence of two multivariate Gaussians,
    N(x|m1, K1) and N(x|m2, K2) 
        
    Parameters
    ----------
    m1 : ndarray, shape(n,)
        mean vector of Gaussian multivariate 1

    K1 : ndarray, shape(n,n)
        covariance matrix of Gaussian multivariate 1

    m2 : ndarray, shape(n,)
        mean vector of Gaussian multivariate 2

    K2 : ndarray, shape(n,n)
        covariance matrix of Gaussian multivariate 2
    
    Returns
    -------
    kl_div : float
        Kullback-Leibler divergence
    
    Notes
    -----
    This is more numerically stable and efficient than naive implementation
    when the Cholesky factor is already available.
    """

    L1 = cholesky(K1, lower=True, overwrite_a=False, check_finite=False)

    L2, alpha = solve_cholesky_system(K2, m2 - m1)
    
    invK2 = compute_cholesky_inverse(L2)

    kl_div = 2 * np.sum(np.log(np.diag(L2))) - 2 * np.sum(np.log(np.diag(L1))) - m2.size
    kl_div += np.trace(invK2 @ K1)
    kl_div += (m2 - m1).T @ alpha
    kl_div *= 0.5

    return kl_div
