import numpy          as np
from scipy.optimize import minimize
from scipy.linalg   import cholesky, cho_solve, solve_triangular
from scipy.stats    import qmc

import multiprocessing
from itertools import repeat

# -------------------------------------------------------------------
#  Heteroscedastic Gaussian Process Class
# -------------------------------------------------------------------

class HeteroscedasticGaussianProcess():
    """
    Class for Heteroscedastic Gaussian Process regression.
    
    Note:
    - fitting is performed according to a variational approach, Lazaro-Gredilla & Titsias (2011)
    - inputs (xinput) always belong to the unit hypercube 
    - outputs (mean and variance) will always be statistically standardized

    Methods:
    - fit
    - predict
    - set_noise_scale_bounds i.e., bounds for mu_0

    Note: the kernels must be GaussianProcessKernel from scikit-learn.
    """

    def __init__(self, kernel_f, kernel_g, nproc, reg_tych=0.0):
        '''
        - kernel_f is the Kernel for the GP of the latent function f ~ GP(f|0, K_g)
        - kernel_g is the Kernel for the GP of the log of the variance 
            eps ~ N(0, r) with r = exp(g) and g ~ GP(g|mu_0, K_g)
        '''
        self.kernel_f = kernel_f
        self.kernel_g = kernel_g
        self.nproc    = nproc
        self.tych     = reg_tych

        self.n_hyp_kerf = len(kernel_f.hyperparameters)
        self.n_hyp_kerg = len(kernel_g.hyperparameters)


    def fit(self, x, y, multistart=20):

        # Standardization
        self.y_train_mean = np.mean(y)
        self.y_train_std  = np.std(y)

        hetero_model_fitting(self, x, y, multistart=multistart)

        # All the matrices used for GP predicition are built
        KF = self.kernel_f(self.x) + self.tych * np.eye(self.ndata) 
        KG = self.kernel_g(self.x) + self.tych * np.eye(self.ndata) 

        # computing mu, Sigma of N(g|mu, Sigma)
        mu = self.mu_0 * np.ones(self.ndata) + KG @ ( self.LAMBDA - 0.5 * np.eye(self.ndata) ) @ np.ones(self.ndata)
        L_KG  = cholesky(KG, lower=True, overwrite_a=False, check_finite=False)
        KGINV = cho_solve((L_KG, True), np.eye(self.ndata), overwrite_b=True)
        L_KGINVLAM = cholesky(KGINV + self.LAMBDA + self.tych * np.eye(self.ndata), lower=True, overwrite_a=True, check_finite=True)
        Sigma = cho_solve((L_KGINVLAM, True), np.eye(self.ndata), overwrite_b=True)

        R = np.diag( np.exp( mu - 0.5 * np.diag( Sigma ) ) )
        # L_KFR is the cholesky decomposition of (K_f + R)
        self.L_KFR = cholesky(KF + R, lower=True, overwrite_a=True, check_finite=False)
        self.alpha = cho_solve((self.L_KFR, True), self.y_norm)

        # L_KGLAMINV is the cholesky decomposition of (K_g + LAMBDA^-1)
        temp = np.diag(self.LAMBDA)
        self.L_KGLAMINV = cholesky(KG + np.diag(1/temp), lower=True, overwrite_a=True, check_finite=False)

        return
    
    def predict(self, xtest, return_var=True, normalized=True, standardized=False):
        """
        Method to compute the prediction of the GP at single or multiple test points.

        Inputs:
        - xtest is a ndarray of shape (ntest, dim) which is the set of points where the surrogate will predict.
            Even if xtest is a point, it must have two dimensions i.e., (1, dim)
            IT MUST BE NORMALIZED WITHIN THE UNIT HYPERCUBE!
        - return_var is a boolean to check is the variance is returned
        - standardized is a boolean to check whether to return standardized statistics (i.e., predicted with zer mean data and unit std)
            or to return de-standardized statistics

        Note: xtest must always have two dimensions!!! 
        """   
        KFSTAR = self.kernel_f(self.x, xtest)

        y_mean = KFSTAR.T @ self.alpha   

        if return_var:

            # variance contribution from f
            V = solve_triangular(self.L_KFR, KFSTAR, lower=True, check_finite=False)

            cstar2 = self.kernel_f(xtest, xtest)
            cstar2 -= V.T @ V
            cstar2 = np.diag(cstar2)

            # variance contribution from g
            KGSTAR = self.kernel_g(self.x, xtest)
            mustar = KGSTAR.T @ (self.LAMBDA - 0.5 * np.eye(self.ndata)) @ np.ones(self.ndata)
            mustar += self.mu_0 * np.ones(xtest.shape[0])

            W = solve_triangular(self.L_KGLAMINV, KGSTAR, lower=True, check_finite=False)
            sigmastar = self.kernel_g(xtest, xtest)
            sigmastar -= W.T @ W
            sigmastar = np.diag(sigmastar)

            y_var = cstar2 + np.exp(mustar + 0.5 * sigmastar)

            if standardized:
                return y_mean, y_var
            
            if not standardized:
                y_mean = y_mean * self.y_train_std + self.y_train_mean
                y_var  = y_var * self.y_train_std**2
                return y_mean, y_var
            
        else:

            if standardized:
                return y_mean
            
            if not standardized:
                y_mean = y_mean * self.y_train_std + self.y_train_mean
                return y_mean           

        return
    
    def set_noise_scale_bounds(self, lb=1e-6, ub=1e-2):
        self.noise_scale_bounds = np.array([lb, ub])
    

# -------------------------------------------------------------------
#  Model Fitting
# -------------------------------------------------------------------
 
def hetero_model_fitting(gp, x, y, multistart=10):
    """
    Heteroscedastic GP model fitting 
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
    if not hasattr(gp, 'noise_scale_bounds'):
        gp.set_noise_scale_bounds()

    # hyperparameters bounds [kernel_f, kernel_g, mu_0, LAMBDA]
    hyp_lw = gp.kernel_f.bounds[:,0];                     hyp_up = gp.kernel_f.bounds[:,1] 
    hyp_lw = np.append(hyp_lw, gp.kernel_g.bounds[:,0]);  hyp_up = np.append(hyp_up, gp.kernel_g.bounds[:,1])
    hyp_lw = np.append(hyp_lw, gp.noise_scale_bounds[0]); hyp_up = np.append(hyp_up, gp.noise_scale_bounds[1])
    hyp_lw = np.append(hyp_lw, np.zeros(gp.ndata));       hyp_up = np.append(hyp_up, 1 * np.ones(gp.ndata)); 

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
    opt_func_tuple, opt_theta_tuple = zip(*pool.starmap(multistart_opt, 
                                                        zip(repeat(fitoptfunc),
                                                        multistart_vec, 
                                                        repeat(gp), 
                                                        init_poin_list)))        
    pool.close()
    pool.join()

    opt_func = np.array([])
    for n in range(gp.nproc):
        opt_func = np.append(opt_func, opt_func_tuple[n])

    gp.kernel_f.theta = opt_theta_tuple[np.nanargmin(opt_func)][ : gp.n_hyp_kerf]
    gp.kernel_g.theta = opt_theta_tuple[np.nanargmin(opt_func)][gp.n_hyp_kerf : (gp.n_hyp_kerf + gp.n_hyp_kerg)]
    gp.mu_0           = opt_theta_tuple[np.nanargmin(opt_func)][(gp.n_hyp_kerf + gp.n_hyp_kerg)]
    gp.LAMBDA         = np.diag( opt_theta_tuple[np.nanargmin(opt_func)][(gp.n_hyp_kerf + gp.n_hyp_kerg+1) : ])


def multistart_opt(fitoptfunc, multistart, gp, theta0):
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

    opt_func = np.zeros(multistart)
    opt_theta = np.zeros_like(theta0)

    for j in range(multistart):

        theta_init = theta0[j,:]

        results = minimize(fitoptfunc, 
                           theta_init, 
                           args=(gp, False), 
                           method="L-BFGS-B", 
                           jac=False, 
                           bounds=list(zip(gp.hyp_lw, gp.hyp_up)),
                           tol=1e-7, 
                           options={'disp': False, 'maxfun':10000})
        
        if results.success == False:
            opt_func[j] = np.nan
            opt_theta[j,:] = opt_theta[j-1,:]
        else:
            opt_func[j] = results.fun
            opt_theta[j,:] = np.squeeze(results.x)

    return np.nanmin(opt_func), opt_theta[np.nanargmin(opt_func),:]

# -------------------------------------------------------------------
#  Negative Variational Log Marginal Likelihood Lower Bound
# -------------------------------------------------------------------

def neg_elbo(theta, gp, eval_gradient=False):
    """
    Function to compute the Negative Lower Bound on the Log Marginal Likelihood for heteroscedastic GP model selection.

    Inputs:
    - theta is the vector of hyperparameters, the first are related to kernel_f, the second to kernel_g, 
    then mu_0, and, lastly, the LAMBDA diagonal 
    - gp is the heteroscedastic GP object
    - eval_gradient is a boolean that tells whether to compute the gradient or not

    Outputs:
    - neg_elbo is the negative Evidence Lower Bound
    - neg_grad is the negative ELBO gradient for the optimization method

    Note: the exact formula is in Lazaro-Gredilla & Titsias (2011)
    """

    gp.kernel_f.theta = theta[ : gp.n_hyp_kerf]
    gp.kernel_g.theta = theta[gp.n_hyp_kerf : (gp.n_hyp_kerf + gp.n_hyp_kerg)]
    mu_0              = theta[(gp.n_hyp_kerf + gp.n_hyp_kerg)]
    LAMBDA            = np.diag(theta[(gp.n_hyp_kerf + gp.n_hyp_kerg + 1) : ])

    # Kernel computations
    if eval_gradient:
        KF, dKF = gp.kernel_f(gp.x, eval_gradient=True)  
        KG, dKG = gp.kernel_g(gp.x, eval_gradient=True) 
    else:
        KF = gp.kernel_f(gp.x) 
        KG = gp.kernel_g(gp.x) 
    KF += gp.tych * np.eye(gp.ndata) 
    KG += gp.tych * np.eye(gp.ndata) 

    # computing mu, Sigma of N(g|mu, Sigma)
    mu = mu_0 * np.ones(gp.ndata) + KG @ ( LAMBDA - 0.5 * np.eye(gp.ndata) ) @ np.ones(gp.ndata)
    L_KG  = cholesky(KG, lower=True, overwrite_a=False, check_finite=False)
    KGINV = cho_solve((L_KG, True), np.eye(gp.ndata), overwrite_b=True)
    L_KGINVLAM = cholesky(KGINV + LAMBDA + gp.tych * np.eye(gp.ndata), lower=True, overwrite_a=True, check_finite=True)
    Sigma = cho_solve((L_KGINVLAM, True), np.eye(gp.ndata), overwrite_b=True)

    # 1st contribution log N(y|0, KF + R)
    R = np.diag( np.exp( mu - 0.5 * np.diag( Sigma ) ) )
    L_KFR = cholesky( KF + R, lower=True, overwrite_a=True, check_finite=True)
    alpha = cho_solve((L_KFR, True), gp.y_norm)

    elbo = - gp.ndata/2 * np.log(2*np.pi) - np.sum(np.log(np.diag(L_KFR))) - 0.5 * gp.y_norm.T @ alpha

    # 2nd contribution -1/4 tr(Sigma)
    elbo -= 1/4 * np.linalg.trace(Sigma)

    # 3rd contribution KL( N(g|mu, Sigma) || N(g|mu_0 * 1, KG) )
    elbo -= kl_div_normals(mu, Sigma, mu_0 * np.ones(gp.ndata), KG)

    if not eval_gradient:
        return - elbo

    if eval_gradient:

        elbo_grad = np.zeros_like(theta)

        # gradient w.r.t. hyperpar of theta_f
        KFR_inv = cho_solve((L_KFR, True), np.eye(gp.ndata), check_finite=False)
        inner_term1 = (alpha @ alpha.T - KFR_inv)
        elbo_grad[ : gp.n_hyp_kerf] = 1/2 * np.einsum("ij,jik->k", inner_term1, dKF)

        # gradients of mu and sigma 
        # order (theta_g, mu0, lambda)
        inner_term2 = ( LAMBDA - 0.5 * np.eye(gp.ndata) ) @ np.ones(gp.ndata)
        dmu_dthetag = np.einsum("ijk,j->ik", dKG, inner_term2)
        dmu_dlambda = KG

        inner_term3 = KGINV @ Sigma
        temp1 = np.einsum("ijl,jk->ikl", dKG, inner_term3)
        dSigma_dthetag = np.einsum("ji,jkl->ikl", inner_term3, temp1)
        dSigma_dlambda = - np.einsum('ji,ik->jki', Sigma, Sigma)

        # gradient of R
        idx = np.arange(gp.ndata)
        inner_term4 = np.exp( mu - 0.5 * np.diag( Sigma ) )
        dRfull_dthetag = np.einsum('i,jk->ijk', inner_term4, dmu_dthetag - 0.5 * np.diagonal(dSigma_dthetag).T)
        dR_dthetag = np.zeros_like(dRfull_dthetag)
        dR_dthetag[idx, idx, :] = dRfull_dthetag[idx, idx, :]
        
        dR_dmu0 = R[...,np.newaxis]

        dRfull_dlambda = np.einsum('i,jk->ijk', inner_term4, dmu_dlambda - 0.5 * np.diagonal(dSigma_dlambda).T)
        dR_dlambda = np.zeros_like(dRfull_dlambda)
        dR_dlambda[idx, idx, :] = dRfull_dlambda[idx, idx, :]

        # gradient of -1/4 * tr(Sigma)
        dtr_dthetag = - 1/4 * np.einsum('iij', dSigma_dthetag)
        dtr_dlambda = - 1/4 * np.einsum('iij', dSigma_dlambda)

        # gradient of the KL divergence
        Sigma_inv = np.linalg.inv(Sigma)
        dkl_dthetag = - 1/2 * np.einsum('ij,jik->k', KGINV, dKG)
        dkl_dthetag += 1/2 * np.einsum('ij,jik->k', Sigma_inv, dSigma_dthetag)
        dkl_dthetag -= 1/2 * np.einsum('ij,jik->k', KGINV, dSigma_dthetag - np.einsum('ijl,jk->ikl', dKG, KGINV @ Sigma))
        dkl_dthetag -= 1/2 * np.einsum('i,ij->j', (( LAMBDA - 1/2 * np.eye(gp.ndata) ) @ np.ones(gp.ndata)).T, 
                                      np.einsum('ijk,i', dKG, ( LAMBDA - 1/2 * np.eye(gp.ndata) ) @ np.ones(gp.ndata)) )
        
        dkl_dlambda = + 1/2 * np.einsum('ij,jik->k', Sigma_inv, dSigma_dlambda)
        dkl_dlambda -= 1/2 * np.einsum('ij,jik->k', KGINV, dSigma_dlambda)
        dkl_dlambda -= 1/2 * np.ones(gp.ndata) * (KG @ ( LAMBDA - 0.5 * np.eye(gp.ndata) ) @ np.ones(gp.ndata))
        dkl_dlambda -= 1/2 * np.einsum('j,ij', np.ones(gp.ndata), ( LAMBDA - 0.5 * np.eye(gp.ndata) ) @ KG)

        # gradient w.r.t. theta_g
        elbo_grad[gp.n_hyp_kerf : (gp.n_hyp_kerf + gp.n_hyp_kerg)] = 0.5 * np.einsum("ij,jik->k", inner_term1, dR_dthetag)
        elbo_grad[gp.n_hyp_kerf : (gp.n_hyp_kerf + gp.n_hyp_kerg)] += dtr_dthetag + dkl_dthetag

        # gradient w.r.t. mu_0
        elbo_grad[(gp.n_hyp_kerf + gp.n_hyp_kerg)] = 0.5 * np.einsum("ij,jik->k", inner_term1, dR_dmu0)

        # gradient w.r.t. lambda        
        elbo_grad[(gp.n_hyp_kerf + gp.n_hyp_kerg + 1):] = 0.5 * np.einsum("ij,jik->k", inner_term1, dR_dlambda)
        elbo_grad[(gp.n_hyp_kerf + gp.n_hyp_kerg + 1):] += dtr_dlambda + dkl_dlambda

        return - elbo, - elbo_grad
    

def kl_div_normals(mu_1, Sigma_1, mu_2, Sigma_2):
    '''
    Exact Kullback-Leibler divergence of two multivariate Gaussians,
    N(x|mu_1, Sigma_1) and N(x|mu_2, Sigma_2) 
    '''

    chol_Sigma1 = cholesky(Sigma_1, lower=True, overwrite_a=False, check_finite=False)
    chol_Sigma2 = cholesky(Sigma_2, lower=True, overwrite_a=False, check_finite=False)
    
    inv_Sigma2 = cho_solve((chol_Sigma2, True), np.eye(mu_2.size), check_finite=False)

    alpha = cho_solve((chol_Sigma2, True), mu_2 - mu_1, check_finite=False)

    kl_div = 2 * np.sum(np.log(np.diag(chol_Sigma2))) - 2 * np.sum(np.log(np.diag(chol_Sigma1))) - mu_2.size
    kl_div += np.trace(inv_Sigma2 @ Sigma_1)
    kl_div += (mu_2 - mu_1).T @ alpha
    kl_div *= 0.5

    return kl_div
