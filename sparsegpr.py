"""
Module for Gaussian Process regression, from Rasmussen and Williams.
"""
import os
import time
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 
os.environ['OPENBLAS_NUM_THREADS'] = "1"

import numpy          as np
from scipy.optimize import minimize, direct
from scipy.linalg   import cholesky, cho_solve, solve_triangular
from scipy.stats    import qmc

import multiprocessing
from itertools import repeat

# -------------------------------------------------------------------
#  Sparse Gaussian Process Class
# -------------------------------------------------------------------

class SparseGaussianProcess():

    """
    Class for Gaussian Process regression from Rasmussen and Williams.
    
    Note:
    - fitting is performed according to Maximum Likelihood Estimate, with Tychonov regularization
    - inputs (xinput) always belong to the unit hypercube 
    - outputs (mean and variance) will always be statistically standardized

    Methods:
    - fit
    - predict
    - set_noise

    Note: the kernel must be a GaussianProcessKernel from scikit-learn,
    but it should not have the noise component, which is handled separately by the SparseGP
    """

    def __init__(self, kernel, nproc, reg_tych=0.0):
        self.kernel  = kernel
        self.nproc   = nproc
        self.tych    = reg_tych

    def fit(self, doe, M=50, multistart=10):
        """
        SparseGP fitting through greedy algorithm and constructing matrices for prediction.

        M -> int for the number of inducing points.
        """

        x, y = doe.data()

        # Standardization
        self.y_train_mean = np.mean(y)
        self.y_train_std  = np.std(y)

        sparse_model_fitting(self, x, y, M, multistart=self.nproc if self.nproc <= 10 else 10)

        # All the matrices used for GP predicition are built

        # Kernel computation
        KMM = self.kernel(self.xm)
        KMN = self.kernel(self.xm, self.x)

        # Cholesky decomposition of the noisy covariance matrix - Faster computation without matrix inversion
        self.LMM = np.linalg.cholesky(KMM + self.tych * np.eye(self.xm.shape[0]))
        SIGMA = np.linalg.inv(KMM + 1 / self.noise_level * KMN @ KMN.T + self.tych * np.eye(self.xm.shape[0]))

        self.mu_m = 1 / self.noise_level * KMM @ SIGMA @ KMN @ self.y_norm
        self.A_m  = KMM @ SIGMA @ KMM

        self.alpha = cho_solve((self.LMM, True), self.mu_m, check_finite=False)

        # W = cho_solve((self.LMM, True), self.A_m, check_finite=False)
        # self.Z = np.transpose(cho_solve((self.LMM, True), W.T, check_finite=False))
        KMMinv = cho_solve((self.LMM,True), np.eye(self.xm.shape[0]), check_finite=False)
        self.Z = KMMinv @ self.A_m @ KMMinv

    def predict(self, xtest, return_cov=True, normalized=True, standardized=False):
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
            
    def set_noise(self, bndlw=1e-6, bndup=1e-2):
        self.noise_bounds = np.array([bndlw, bndup])


# -------------------------------------------------------------------
#  Model Fitting
# -------------------------------------------------------------------
 
def sparse_model_fitting(gp, x, y, M, multistart=10):
    """
    Sparse GP model fitting through greedy algorithm
    """
    gp.x       = x
    gp.y       = y
    gp.dim     = x.shape[1]
    gp.ndata   = x.shape[0]

    # Normalization
    gp.y_norm = (gp.y.reshape(gp.ndata,1) - gp.y_train_mean * np.ones((gp.ndata,1))) / gp.y_train_std

    # fitting method
    fitoptfunc = neg_elbo

    # Kernel hyperparameters bounds
    if not hasattr(gp, 'noise_bounds'):
        gp.set_noise()

    hyp_lw = np.append(gp.kernel.bounds[:,0], gp.noise_bounds[0])
    hyp_up = np.append(gp.kernel.bounds[:,1], gp.noise_bounds[1])

    gp.hyp_lw = hyp_lw
    gp.hyp_up = hyp_up

    bounds = list(zip(hyp_lw, hyp_up))
    multistart_vec = [int(multistart / gp.nproc) for i in range(gp.nproc)]

    # sampling the whole set of initial points via LHS and then partion it for the number of processors
    sampler = qmc.LatinHypercube(d=len(bounds))
    initial_points = sampler.random(n=multistart)
    initial_points = qmc.scale(initial_points, hyp_lw, hyp_up)
    init_poin_list = [ initial_points[i*(int(multistart / gp.nproc)) : (i+1)*(int(multistart / gp.nproc))] for i in range(gp.nproc)]
    
    # redistributing the remaing part of initial points
    assigned = (gp.nproc)*int(multistart / gp.nproc)
    if assigned < multistart:
        for i in range(multistart - assigned):
            init_poin_list[i] = np.vstack((init_poin_list[i], initial_points[assigned + i,:]))
            
    # multiprocessing for multipoint optimization
    pool = multiprocessing.Pool(processes=gp.nproc) 
    opt_func_tuple, opt_theta_tuple, xm_tuple = zip(*pool.starmap(multistart_opt, 
                                                       zip(repeat(fitoptfunc),
                                                       multistart_vec, 
                                                       repeat(gp), 
                                                       repeat(M),
                                                       init_poin_list)))        
    pool.close()
    pool.join()

    opt_func = np.array([])
    for n in range(gp.nproc):
        opt_func = np.append(opt_func, opt_func_tuple[n])
        print(f'WORK {n} - ELBO {-opt_func_tuple[n]:.6e}')


    gp.kernel.theta = opt_theta_tuple[np.nanargmin(opt_func)][:-1]
    gp.noise_level  = opt_theta_tuple[np.nanargmin(opt_func)][-1]
    gp.xm           = xm_tuple[np.nanargmin(opt_func)]

def multistart_opt(fitoptfunc, multistart, gp, M, theta0):
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
    # GREEDY ALGORITHM
    opt_func = np.zeros(M)
    opt_theta = np.zeros((M, theta0.size))
    M_INDEX = []
    i_opt = 0
    theta0 = np.squeeze(theta0)
    for j in range(M):

        while True:
            t = np.random.randint(0, gp.ndata-1)
            if t not in M_INDEX:
                M_INDEX.append(t)
                break

        xm = gp.x[M_INDEX,:]

        results = minimize(fitoptfunc, 
                           theta0, 
                           args=(gp, xm, False), 
                           method="L-BFGS-B", 
                           jac=False, 
                           bounds=list(zip(gp.hyp_lw, gp.hyp_up)),
                           tol=1e-7, 
                           options={'disp': False, 'maxfun':10000})
        
        if results.success == False:
            opt_func[i_opt] = np.nan
            opt_theta[i_opt,:] = opt_theta[i_opt-1,:]
        else:
            opt_func[i_opt] = results.fun
            opt_theta[i_opt,:] = np.squeeze(results.x)

        # starting from the new optimum
        theta0 = opt_theta[i_opt,:]

        i_opt+=1


    return opt_func[-1], opt_theta[-1,:], xm

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
    noise_level          = theta[-1]

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

    






