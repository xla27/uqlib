import numpy          as np
from scipy.linalg   import cholesky, solve_triangular, cho_solve, solve
from scipy.stats    import norm
from scipy.special  import expit, log_expit, erf

from .utils import (
    standardize_targets, initialize_gp_metadata,
    build_hyperparameter_bounds, generate_lhs_points, 
    destandardize_predictions, run_multiprocessing_optimization,
    solve_cholesky_system, compute_cholesky_inverse
)

# -------------------------------------------------------------------
#  Gaussian Process Classifier Class
# -------------------------------------------------------------------

class GaussianProcessClassifier():

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

    def __init__(self, kernel, likelihood, nproc=1, reg_tych=0.0):
        self.kernel  = kernel

        if likelihood == 'logistic':
            self.likelihood = Logistic()

        elif likelihood == 'probit':
            self.likelihood = Probit()

        self.nproc   = nproc
        self.tych    = reg_tych

    def fit(self, x, y, multistart=10):
        """
        GP fitting via MLE/MAP approach. 
        DIRECT or L-BFGS-B are used for LML/APE optimization, in the latter a multistart approach is employed.
        """

        # Preprocessing
        initialize_gp_metadata(self, x, y, normalize=False)
        self.y = np.where(self.y < 1, -1, 1)

        self.hyp_lw, self.hyp_up = build_hyperparameter_bounds([self.kernel.bounds])

        # fitting method
        fitoptfunc = laplace_loglike   
            
        # Global optimization of the model loss function (hyperparameters are log-transformed for the optimization and then transformed back)
        bounds = list(zip(self.hyp_lw, self.hyp_up))
        init_poin_list, multistart_vec = generate_lhs_points(bounds, multistart, self.nproc)
                
        # Multiprocessing for multipoint optimization
        opt_func, opt_theta = run_multiprocessing_optimization(fitoptfunc, self, init_poin_list, multistart_vec)

        # IMPLEMENT HYPERPARAMETERS OPTIMIZATION
        self.kernel.theta = opt_theta[np.nanargmin(opt_func),:]

        # Matrix precomputations
        K = self.kernel(self.x, eval_gradient=False)

        _, (self.fhat, _, self.L, self.W_sr) = self.mode(K, return_utils=True)

        self.w = np.squeeze(self.likelihood.gradlog(self.y, self.fhat))

        return
    
    def predict(self, xtest):

        k_mat = self.kernel(self.x, xtest)

        f_map = k_mat.T @ self.w

        return np.where(f_map > 0, +1, -1)
    
    def predict_proba(self, xtest):
        """
        Method to compute the prediction of the GP at single or multiple test points.
        The y = +1 prediction probability is returned.

        Inputs:
        - xtest is a ndarray of shape (ntest, dim) which is the set of points where the surrogate will predict.
            Even if xtest is a point, it must have two dimensions i.e., (1, dim)
            IT MUST BE NORMALIZED WITHIN THE UNIT HYPERCUBE!

        Note: xtest must always have two dimensions!!! 
        """    

        f_mean, f_var = self.predict_latent(xtest)

        if isinstance(self.likelihood, Logistic):

            alpha = 1 / (2 * f_var)
            gamma = LAMBDAS * f_mean

            integrals = (np.sqrt(np.pi / alpha)
                         * erf(gamma * np.sqrt(alpha / (alpha + LAMBDAS**2)))
                         / (2 * np.sqrt(f_var * 2 * np.pi)))
            
            pi_star = np.sum(COEFS * integrals, axis=0) + 0.5 * np.sum(COEFS)

            # k = (1 + np.pi/8 * f_var)**(-1/2)

            # pi_star = expit(k * f_mean)

        elif isinstance(self.likelihood, Probit):

            pi_star = norm.cdf(f_mean / np.sqrt(f_var + 1))

        return np.vstack((1 - pi_star, pi_star)).T
        
    def predict_latent(self, xtest):
        """
        Method to compute the mean and variance prediction of the latent variable

        Inputs:
        - xtest is a ndarray of shape (ntest, dim) which is the set of points where the surrogate will predict.
            Even if xtest is a point, it must have two dimensions i.e., (1, dim)
            IT MUST BE NORMALIZED WITHIN THE UNIT HYPERCUBE!

        Note: xtest must always have two dimensions!!! 
        """      
        
        k_mat = self.kernel(self.x, xtest)

        f_mean = k_mat.T @ self.w

        V = solve(self.L, self.W_sr[:, np.newaxis] * k_mat, check_finite=False)

        f_var = self.kernel.diag(xtest) - np.einsum("ij,ij->j", V, V)

        return f_mean, f_var

    def mode(self, K, return_utils=False):
        """
        Method to compute the mode of the approxiamte posterior
        """

        f = np.zeros_like(self.y)

        obj = - np.inf

        for _ in range(100):

            W = - self.likelihood.hesslog(self.y, f)
            W_sr = np.sqrt(W)
            W_sr_K = W_sr[:,np.newaxis] * K

            b = W * f + self.likelihood.gradlog(self.y, f)
            B = np.eye(self.ndata) + W_sr_K * W_sr
            L = cholesky(B, lower=True)
            a = b - W_sr * cho_solve((L, True), np.dot(W_sr_K, b), check_finite=False)
            f_new = np.dot(K, a)

            obj_new = - 0.5 * np.dot(a,f_new) + self.likelihood.log(self.y, f_new) - np.sum(np.log(np.diag(L)))   

            if obj_new - obj < 1e-10:
                break

            elif obj_new < obj:
                raise Exception('Diverging Newton Method for mode finding.')
            
            obj = obj_new
            f = f_new
            
        log_approx_marg = obj_new #- np.sum(np.log(np.diag(L)))  

        if return_utils:

            return log_approx_marg, (f_new, a, L, W_sr)  
        
        else: 

            return log_approx_marg

# Values required for approximating the logistic sigmoid by
# error functions. coefs are obtained via:
# x = np.array([0, 0.6, 2, 3.5, 4.5, np.inf])
# b = logistic(x)
# A = (erf(np.dot(x, LAMBDAS)) + 1) / 2
# coefs = lstsq(A, b)[0]
LAMBDAS = np.array([0.41, 0.4, 0.37, 0.44, 0.39])[:, np.newaxis]
COEFS = np.array([-1854.8214151, 
                  3516.89893646,
                  221.29346712, 
                  128.12323805, 
                  -2010.49422654])[:, np.newaxis]

# -------------------------------------------------------------------
#  Likelihoods
# -------------------------------------------------------------------

class Logistic():

    def __init__(self):
        return
    
    def __call__(self, y, f):
        return np.prod(expit(y * f))
    
    def log(self, y, f):
        return np.sum(log_expit(y * f))
    
    def gradlog(self, y, f):
        
        t = 0.5 * (y + np.ones_like(y))

        pi = expit(f)

        return t - pi
    
    def hesslog(self, y, f):

        pi = expit(f)

        return - pi * (1 - pi)
    
    def thirdlog(self, y, f):

        pi = expit(f)

        return pi * (1 - pi) * (1 - 2 * pi)

        

class Probit():

    def __init__(self):
        self.dist = norm
        return
    
    def __call__(self, y, f):
        return np.prod(self.dist.cdf(y * f))
    
    def log(self, y, f):
        return np.sum(self.dist.logcdf(y * f))
    
    def gradlog(self, y, f):
        
        num = y * self.dist.pdf(f)
        den = self.dist.cdf(y * f)

        return num / den
    
    def hesslog(self, y, f):

        num1 = self.dist.pdf(f) ** 2
        den1 = self.dist.cdf(y * f) ** 2
        a1 = - num1 / den1

        num2 = y * f * self.dist.pdf(f) 
        den2 = self.dist.cdf(y * f)
        a2 = - num2 / den2

        return a1 + a2
    
    def thirdlog(self, y, f):

        raise NotImplementedError('Not yet implemented.')

# -------------------------------------------------------------------
#  Log Marginal Likelihood
# -------------------------------------------------------------------

def laplace_loglike(theta, gp, eval_gradient=False):
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

    log_approx_marg, (f, a, L, W_sr)  = gp.mode(K, return_utils=True)
    
    if eval_gradient == False:

        return - log_approx_marg

    if eval_gradient == True:
        
        R = W_sr[:, np.newaxis] * cho_solve((L, True), np.diag(W_sr), check_finite=False, overwrite_b=True)

        C = solve(L, W_sr[:, np.newaxis] * K, check_finite=False, overwrite_a=True, overwrite_b=True)

        b = np.einsum("ijk,j->ik", K_gradient, gp.likelihood.gradlog(gp.y, f))

        s1 = 0.5 * np.einsum("ij,jik->k", (np.outer(a,a) - R), K_gradient)

        s2 = - 0.5 * ( np.diag(K) - np.einsum("ij, ij -> j", C, C) ) * gp.likelihood.thirdlog(gp.y, f)

        s3 = np.einsum("ij,jk->ik",(np.eye(b.shape[0]) - K @ R), b)

        approx_marg_grad = s1 + np.einsum("i,ij->j", s2, s3)

        return - log_approx_marg, - approx_marg_grad
    
    
