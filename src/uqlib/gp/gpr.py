import numpy          as np
from scipy.linalg   import solve_triangular

from .utils import (
    standardize_targets, initialize_gp_metadata,
    build_hyperparameter_bounds, generate_lhs_points, 
    destandardize_predictions, run_multiprocessing_optimization,
    solve_cholesky_system, compute_cholesky_inverse
)

# -------------------------------------------------------------------
#  Gaussian Process Class
# -------------------------------------------------------------------

class GaussianProcessRegressor():

    """
    Class for Gaussian Process regression from Rasmussen and Williams.
    
    Note:
    - fitting is performed according to Maximum Likelihood Estimate, with Tychonov regularization
    - inputs (xinput) always belong to the unit hypercube 
    - outputs (mean and variance) will always be statistically standardized

    Methods:
    - fit: fitting of the GP according to a MLE
    - predict: statistics of the GP at a set of test points
    """

    def __init__(self, kernel, nproc=1, reg_tych=0.0):
        self.kernel  = kernel
        self.nproc   = nproc
        self.tych    = reg_tych

    def fit(self, x, y, multistart=10):
        """
        GP fitting via MLE/MAP approach. 
        DIRECT or L-BFGS-B are used for LML/APE optimization, in the latter a multistart approach is employed.
        """

        # Standardization
        self.y_train_mean, self.y_train_std = standardize_targets(y)

        # Preprocessing
        initialize_gp_metadata(self, x, y)

        self.hyp_lw, self.hyp_up = build_hyperparameter_bounds([self.kernel.bounds])

        # fitting method
        fitoptfunc = logmarglike   
            
        # Global optimization of the model loss function (hyperparameters are log-transformed for the optimization and then transformed back)
        bounds = list(zip(self.hyp_lw, self.hyp_up))
        init_poin_list, multistart_vec = generate_lhs_points(bounds, multistart, self.nproc)
                
        # Multiprocessing for multipoint optimization
        opt_func, opt_theta = run_multiprocessing_optimization(fitoptfunc, self, init_poin_list, multistart_vec)

        self.kernel.theta = opt_theta[np.nanargmin(opt_func),:]

        # All the matrices used for GP predicition are built
        K = self.kernel(self.x, eval_gradient=False)

        self.L, self.alpha = solve_cholesky_system(K, self.y_norm, self.tych)
        
    def predict(self, xtest, return_cov=True, standardized=False):
        """
        Method to compute the prediction of the GP at single or multiple test points.

        Inputs:
        - xtest is a ndarray of shape (ntest, dim) which is the set of points where the surrogate will predict.
            Even if xtest is a point, it must have two dimensions i.e., (1, dim)
            IT MUST BE NORMALIZED WITHIN THE UNIT HYPERCUBE!
        - return_cov is a boolean to check is the covariance matrix is returned
        - standardized is a boolean to check whether to return standardized statistics (i.e., predicted with zer mean data and unit std)
            or to return de-standardized statistics

        Note: xtest must always have two dimensions!!! 
        """      
        k_mat = self.kernel(self.x, xtest)

        y_mean = k_mat.T @ self.alpha

        if return_cov:
            V = solve_triangular(self.L, k_mat, lower=True, check_finite=False)
            
            y_cov = self.kernel(xtest, xtest) - V.T @ V

            return destandardize_predictions(y_mean, self.y_train_mean, 
                                 self.y_train_std, y_cov, standardized)
        
        else:

            return destandardize_predictions(y_mean, self.y_train_mean, 
                                 self.y_train_std, None, standardized)


# -------------------------------------------------------------------
#  Log Marginal Likelihood
# -------------------------------------------------------------------

def logmarglike(theta, gp, eval_gradient=False):
    """
    Function to compute the Log Marginal Likelihood (LML) of the GP and its gradient w.r.t. the hyperparameters, which are log-transformed

    Inputs:
    - theta is the vector of hyperparameters, typically dim + 2 for the SE-ARD kernel
    - eval_gradient is a boolean that tells whether to compute the gradient or not

    Outputs:
    - neg_lml is the negative LML for the optimization method, which is a minimization algorithm
    - neg_grad is the negative LML gradient for the optimization method
    """
    gp.kernel.theta = theta

    # Kernel computation
    K, K_gradient = gp.kernel(gp.x, eval_gradient=True)

    # Cholesky decomposition of the noisy covariance matrix
    L, alpha = solve_cholesky_system(K, gp.y_norm, gp.tych)

    # Computation of the marginal likelihood
    log_marg_like = - 0.5 * gp.y_norm.T @ alpha - np.sum(np.log(np.diag(L))) - gp.ndata / 2 * np.log(2*np.pi)
    
    if eval_gradient == False:
        return - log_marg_like

    if eval_gradient == True:

        # Computation of the marginal likelihood gradient
        # from Rasmussen & Williams, each gradient component theta_j is equal to
        #  0.5 * trace((alpha . alpha^T - K^-1) . dK_dtheta_j)            
        K_inv = compute_cholesky_inverse(L)
        inner_term = (alpha @ alpha.T - K_inv)
        inner_term = inner_term[..., np.newaxis]

        marg_like_grad_dims = 0.5 * np.einsum("ijl,jik->kl", inner_term, K_gradient)
        marg_like_grad = np.squeeze(marg_like_grad_dims)

        return - log_marg_like, - marg_like_grad
    
    






