import numpy          as np
from scipy.linalg   import cholesky, solve_triangular

from .utils import (
    standardize_targets, initialize_gp_metadata,
    build_hyperparameter_bounds, generate_lhs_points, 
    destandardize_predictions, run_multiprocessing_optimization,
    solve_cholesky_system, compute_cholesky_inverse
)

# -------------------------------------------------------------------
#  Sparse Gaussian Process Class
# -------------------------------------------------------------------

class SparseGaussianProcess():
    """
    Class for Gaussian Process Sparse regression.
    
    Note:
    - fitting is performed according to a variational-greedy approach, Titsias (2009)
    - inputs (xinput) always belong to the unit hypercube 
    - outputs (mean and variance) will always be statistically standardized

    Methods:
    - fit
    - predict
    - set_noise

    Note: the kernel must be a GaussianProcessKernel from scikit-learn,
    but it should not have the noise component, which is handled separately by the SparseGP
    """

    def __init__(self, kernel, nproc=1, reg_tych=0.0):
        self.kernel  = kernel
        self.nproc   = nproc
        self.tych    = reg_tych

    def fit(self, x, y, M=50, multistart=10):
        """
        SparseGP fitting through greedy algorithm and constructing matrices for prediction.

        M -> int for the number of inducing points.
        """

        # Standardization
        self.y_train_mean, self.y_train_std = standardize_targets(y)

        if not hasattr(self, 'noise_bounds'):
            self.set_noise_bounds()

        initialize_gp_metadata(self, x, y)

        self.hyp_lw, self.hyp_up = build_hyperparameter_bounds(
            [self.kernel.bounds],
            {'noise': (self.noise_bounds[0], self.noise_bounds[1])}
        )

        # fitting method
        fitoptfunc = neg_elbo

        # Global optimization of the model loss function (hyperparameters are log-transformed for the optimization and then transformed back)
        bounds = list(zip(self.hyp_lw, self.hyp_up))
        init_poin_list, _ = generate_lhs_points(bounds, self.nproc, self.nproc)
        multistart_vec = [M] * self.nproc

        # multiprocessing for multipoint optimization
        opt_func, opt_theta, opt_xm = run_multiprocessing_optimization(fitoptfunc, 
                                                                       self, 
                                                                       init_poin_list, 
                                                                       multistart_vec, 
                                                                       extra_args=(True,))

        self.kernel.theta = opt_theta[np.nanargmin(opt_func),:-1]
        self.noise_level  = opt_theta[np.nanargmin(opt_func),-1]
        self.xm           = opt_xm[np.nanargmin(opt_func)]

        # All the matrices used for GP predicition are built

        # Kernel computation
        KMM = self.kernel(self.xm)
        KMN = self.kernel(self.xm, self.x)

        # Cholesky decomposition of the noisy covariance matrix - Faster computation without matrix inversion
        SIGMA = np.linalg.inv(KMM + 1 / self.noise_level * KMN @ KMN.T + self.tych * np.eye(self.xm.shape[0]))

        self.mu_m = 1 / self.noise_level * KMM @ SIGMA @ KMN @ self.y_norm
        self.A_m  = KMM @ SIGMA @ KMM

        self.LMM, self.alpha = solve_cholesky_system(KMM, self.mu_m, self.tych)

        # W = cho_solve((self.LMM, True), self.A_m, check_finite=False)
        # self.Z = np.transpose(cho_solve((self.LMM, True), W.T, check_finite=False))
        KMMinv = compute_cholesky_inverse(self.LMM)
        self.Z = KMMinv @ self.A_m @ KMMinv

    def predict(self, xtest, return_cov=True, standardized=False):
        """
        Method to compute the prediction of the GP at single or multiple test points.

        Inputs:
        - xtest is a ndarray of shape (ntest, dim) which is the set of points where the surrogate will predict.
            Even if xtest is a point, it must have two dimensions i.e., (1, dim)
            IT MUST BE NORMALIZED WITHIN THE UNIT HYPERCUBE!
        - return_cov is a boolean to check is the covariance matrix is returned
        - standardized is a boolean to check whether to return standardized statistics (i.e., predicted with zero mean data and unit std)
            or to return de-standardized statistics

        Note: xtest must always have two dimensions!!! 
        """      

        KSTARM = self.kernel(xtest, self.xm)
        KMSTAR = KSTARM.T

        y_mean = KSTARM @ self.alpha

        if return_cov:
            V = solve_triangular(self.LMM, KMSTAR, lower=True, check_finite=False)
            
            y_cov = self.kernel(xtest, xtest) - V.T @ V + KSTARM @ self.Z @ KMSTAR

            return destandardize_predictions(y_mean, self.y_train_mean, 
                                 self.y_train_std, y_cov, standardized)
        
        else:

            return destandardize_predictions(y_mean, self.y_train_mean, 
                                 self.y_train_std, None, standardized)
            
    def set_noise_bounds(self, lb=1e-6, ub=1e-2):
        self.noise_bounds = np.array([lb, ub])


# -------------------------------------------------------------------
#  Negative Variational Log Marginal Likelihood Lower Bound
# -------------------------------------------------------------------

def neg_elbo(theta, gp, xm, eval_gradient=False):
    """
    Function to compute the Negative Lower Bound on the Log Marginal Likelihood for SparseGP model selection.

    Inputs:
    - theta is the vector of hyperparameters, the last is always the noise level (sigma^2)
    - xm is the (M, dim) array of inducing points
    - eval_gradient is a boolean that tells whether to compute the gradient or not

    Outputs:
    - neg_lml is the negative LML for the optimization method, which is a minimization algorithm
    - neg_grad is the negative LML gradient for the optimization method

    Note: the exact formula is in Titsias (2009), but the implementation follows the numerically stable 
    formula of Krasser: https://krasserm.github.io/2020/12/12/gaussian-processes-sparse/
    """
    gp.kernel.theta = theta[:-1]
    noise_level     = theta[-1]

    # Kernel computation
    KNN = gp.kernel(gp.x)
    KMM = gp.kernel(xm)
    KMN = gp.kernel(xm, gp.x)

    # Cholesky decomposition
    LMM = cholesky(KMM + gp.tych * np.eye(xm.shape[0]),
                   lower=True, overwrite_a=True, check_finite=False)          # KMM = LMM @ LMM.TT
    
    D = solve_triangular(LMM, 1/np.sqrt(noise_level) * KMN, 
                           lower=True, overwrite_b=True, check_finite=False)    
    
    LB = cholesky(np.eye(xm.shape[0]) + D @ D.T,   
                  lower=True, overwrite_a=True, check_finite=False)           # LB @ LB.T = sigma^2 I + QNN
    
    c = solve_triangular(LB, 1/np.sqrt(noise_level) * D @ gp.y_norm,
                         lower=True, overwrite_b=True, check_finite=False)

    # Computation of the marginal likelihood
    log_marg_like = - gp.ndata/2 * np.log(2*np.pi*noise_level)           \
                    - np.sum(np.diag(LB))                           \
                    - 1/2 / noise_level * np.dot(gp.y_norm.T,gp.y_norm)  \
                    + 1/2 * np.dot(c.T,c)                           \
                    - 1/2 / noise_level * np.linalg.trace(KNN)           \
                    + 1/2 * np.linalg.trace(D @ D.T)

    if eval_gradient == False:
        return - log_marg_like

    if eval_gradient == True:
        raise ValueError('Gradient not yet implemented!')

    






