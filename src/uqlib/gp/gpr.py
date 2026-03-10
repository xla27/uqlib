import numpy          as np
from scipy.optimize import minimize
from scipy.linalg   import cholesky, cho_solve, solve_triangular
from scipy.stats    import qmc

import multiprocessing
from itertools import repeat

# -------------------------------------------------------------------
#  Gaussian Process Class
# -------------------------------------------------------------------

class GaussianProcess():

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
        self.y_train_mean = np.mean(y)
        self.y_train_std  = np.std(y)

        model_fitting(self, x, y, multistart=multistart)

        # All the matrices used for GP predicition are built

        # Standardization
        self.y_norm = (self.y.reshape(self.ndata,1) - self.y_train_mean * np.ones((self.ndata,1))) / self.y_train_std 

        # Kernel computation
        K = self.kernel(self.x, eval_gradient=False)

        '''
        # Cholesky decomposition of the noisy covariance matrix - Standard computation
        L    = np.linalg.cholesky(K + self.tych**2 * np.eye(self.ndata))
        Linv = np.linalg.inv(L)
        alpha = np.linalg.inv(L.T) @ (Linv @ y_norm)
        '''

        # Cholesky decomposition of the noisy covariance matrix - Faster computation without matrix inversion
        self.L     = np.linalg.cholesky(K + self.tych * np.eye(self.ndata))
        self.alpha = cho_solve((self.L, True), 
                               self.y_norm, 
                               check_finite=False)
        
    def predict(self, xtest, return_cov=True, normalized=True, standardized=False):
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

            if standardized:
                return y_mean, y_cov
            
            if not standardized:
                y_mean = y_mean * self.y_train_std + self.y_train_mean
                y_cov  = y_cov * self.y_train_std**2
                return y_mean, y_cov
        
        else:

            if standardized:
                return y_mean
            
            if not standardized:
                y_mean = y_mean * self.y_train_std + self.y_train_mean
                return y_mean


# -------------------------------------------------------------------
#  Model Fitting
# -------------------------------------------------------------------
 
def model_fitting(gp, x, y, multistart=100):
    """
    GP fitting via MLE/MAP approach. 
    DIRECT or L-BFGS-B are used for LML/APE optimization, in the latter a multistart approach is employed.
    """
    gp.x       = x
    gp.y       = y
    gp.dim     = x.shape[1]
    gp.ndata   = x.shape[0]

    # fitting method
    fitoptfunc = logmarglike

    # Hyperparameters bounds
    hyp_lw = gp.kernel.bounds[:,0]
    hyp_up = gp.kernel.bounds[:,1] 

    gp.hyp_lw = hyp_lw
    gp.hyp_up = hyp_up

    # Normalization
    y_norm = (gp.y.reshape(gp.ndata,1) - gp.y_train_mean * np.ones((gp.ndata,1))) / gp.y_train_std

    gp.y_norm = y_norm
        
    # Global optimization of the model loss function (hyperparameters are log-transformed for the optimization and then transformed back)
    bounds = list(zip(hyp_lw, hyp_up))

    multistart_vec = [int(multistart / gp.nproc) for i in range(gp.nproc)]

    # sampling the whole set of initial points via LHS and then partion it for the number of processors
    sampler = qmc.LatinHypercube(d=len(bounds))
    initial_points = sampler.random(n=multistart)
    initial_points = np.repeat(hyp_lw[np.newaxis, :], repeats=multistart, axis=0) + initial_points*(
        np.repeat(hyp_up[np.newaxis, :], repeats=multistart, axis=0) - np.repeat(hyp_lw[np.newaxis, :], repeats=multistart, axis=0)
    )
    init_poin_list = [ initial_points[i*(int(multistart / gp.nproc)) : (i+1)*(int(multistart / gp.nproc))] for i in range(gp.nproc)]
    
    # redistributing the remaing part of initial points
    assigned = (gp.nproc)*int(multistart / gp.nproc)
    if assigned < multistart:
        for i in range(multistart - assigned):
            init_poin_list[i] = np.vstack((init_poin_list[i], initial_points[assigned + i,:]))
            
    # multiprocessing for multipoint optimization
    pool = multiprocessing.Pool(processes=gp.nproc) 
    opt_lml_tuple, opt_theta_tuple = zip(*pool.starmap(multistart_opt, zip(repeat(fitoptfunc), multistart_vec, repeat(gp), init_poin_list)))        
    pool.close()
    pool.join()

    opt_lml = np.array([])
    opt_theta = np.empty((0,len(bounds)))
    for n in range(gp.nproc):
        opt_lml = np.append(opt_lml, opt_lml_tuple[n])
        opt_theta = np.vstack((opt_theta, opt_theta_tuple[n]))

    gp.kernel.theta = opt_theta[np.argmin(opt_lml),:]


def multistart_opt(fitoptfunc, multistart, gp, init_set):
    """
    Function to allow multiprocessing for LML maximization.
    Basically, it parallelizes the for-cycle of L-BFGS-B multistart

    Input:
    - multistart is an integer that represent the number of for-cycle per each process
    - gp is the GaussianProcess class
    - init_set is the set of initial points LHS-sampled

    Output:
    - opt_lml is the vector of the optimized (neg)LML of each multistart
    - opt_theta is the array of the optimized hyperparameters of each multistart

    """

    opt_lml = np.zeros(multistart)
    opt_theta = np.zeros((multistart, len(gp.hyp_lw)))
    for i in range(multistart):
        log_initial = init_set[i, :]

        results = minimize(fitoptfunc, 
                           log_initial, 
                           args=(gp, True), 
                           method="L-BFGS-B", 
                           jac=True, 
                           bounds=list(zip(gp.hyp_lw, gp.hyp_up)),
                           tol=1e-7, 
                           options={'disp': False, 'maxfun':10000})
        
        if results.success == False:
            opt_lml[i] = 1e10
        else:
            opt_lml[i] = results.fun
        opt_theta[i,:] = (results.x).reshape(1, len(gp.hyp_lw))

    return opt_lml, opt_theta


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
    K     = K + gp.tych * np.eye(gp.ndata)                 # Tychonov regularization
    L     = cholesky(K, lower=True, overwrite_a=True, check_finite=False)
    # from Rasmussen & Williams:
    # alpha = L^T \ (L \ y)
    alpha = cho_solve((L, True), gp.y_norm, check_finite=False)

    # Computation of the marginal likelihood
    log_marg_like = - 0.5 * gp.y_norm.T @ alpha - np.sum(np.log(np.diag(L))) - gp.ndata / 2 * np.log(2*np.pi)
    
    if eval_gradient == False:
        return - log_marg_like

    if eval_gradient == True:

        # Computation of the marginal likelihood gradient
        # from Rasmussen & Williams, each gradient component theta_j is equal to
        #  0.5 * trace((alpha . alpha^T - K^-1) . dK_dtheta_j)            
        K_inv = cho_solve((L, True), np.eye(gp.ndata), check_finite=False)
        inner_term = (alpha @ alpha.T - K_inv)
        inner_term = inner_term[..., np.newaxis]

        marg_like_grad_dims = 0.5 * np.einsum("ijl,jik->kl", inner_term, K_gradient)
        marg_like_grad = np.squeeze(marg_like_grad_dims)

        return - log_marg_like, - marg_like_grad
    
    






