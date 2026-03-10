import numpy          as np
from scipy.linalg   import cholesky, solve_triangular

from .utils import (
    standardize_targets, initialize_gp_metadata,
    build_hyperparameter_bounds, generate_lhs_points, 
    destandardize_predictions, run_multiprocessing_optimization,
    solve_cholesky_system, compute_cholesky_inverse,
    kl_div_normals
)

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

    def __init__(self, kernel_f, kernel_g, nproc=1, reg_tych=0.0):
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
        self.y_train_mean, self.y_train_std = standardize_targets(y)

        if not hasattr(self, 'noise_scale_bounds'):
            self.set_noise_scale_bounds()

        # Preprocessing
        initialize_gp_metadata(self, x, y)

        self.hyp_lw, self.hyp_up = build_hyperparameter_bounds(
            [self.kernel_f.bounds, self.kernel_g.bounds],
            {
                'mu_0': (self.noise_scale_bounds[0], self.noise_scale_bounds[1]),
                'lambda': (0 * np.ones(self.ndata), 10 * np.ones(self.ndata))  # for LAMBDA parameters
            }
        )

        # fitting method
        fitoptfunc = neg_elbo

        # Global optimization of the model loss function (hyperparameters are log-transformed for the optimization and then transformed back)
        bounds = list(zip(self.hyp_lw, self.hyp_up))
        init_poin_list, multistart_vec = generate_lhs_points(bounds, multistart, self.nproc)
        
        # Multiprocessing for multipoint optimization
        opt_func, opt_theta = run_multiprocessing_optimization(fitoptfunc, self, init_poin_list, multistart_vec)

        self.kernel_f.theta = opt_theta[np.nanargmin(opt_func),: self.n_hyp_kerf]
        self.kernel_g.theta = opt_theta[np.nanargmin(opt_func),self.n_hyp_kerf : (self.n_hyp_kerf + self.n_hyp_kerg)]
        self.mu_0           = 10**(opt_theta[np.nanargmin(opt_func),(self.n_hyp_kerf + self.n_hyp_kerg)])
        self.LAMBDA         = np.diag( opt_theta[np.nanargmin(opt_func),(self.n_hyp_kerf + self.n_hyp_kerg+1) : ])

        # All the matrices used for GP predicition are built
        KF = self.kernel_f(self.x) + self.tych * np.eye(self.ndata) 
        KG = self.kernel_g(self.x) + self.tych * np.eye(self.ndata) 

        # computing mu, Sigma of N(g|mu, Sigma)
        mu = self.mu_0 * np.ones(self.ndata) + KG @ ( self.LAMBDA - 0.5 * np.eye(self.ndata) ) @ np.ones(self.ndata)
        L_KG  = cholesky(KG, lower=True, overwrite_a=False, check_finite=False)
        KGINV = compute_cholesky_inverse(L_KG)

        _, Sigma = solve_cholesky_system(KGINV + self.LAMBDA, np.eye(self.ndata), self.tych)

        R = np.diag( np.exp( mu - 0.5 * np.diag( Sigma ) ) )
        # L_KFR is the cholesky decomposition of (K_f + R)
        self.L_KFR, self.alpha = solve_cholesky_system(KF + R, self.y_norm)

        # L_KGLAMINV is the cholesky decomposition of (K_g + LAMBDA^-1)
        temp = np.diag(self.LAMBDA)
        self.L_KGLAMINV = cholesky(KG + np.diag(1/temp), lower=True, overwrite_a=True, check_finite=False)
    
    def predict(self, xtest, return_var=True, standardized=False):
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

            return destandardize_predictions(y_mean, self.y_train_mean, 
                                 self.y_train_std, y_var, standardized)
            
        else:

            return destandardize_predictions(y_mean, self.y_train_mean, 
                                 self.y_train_std, None, standardized)          

    def set_noise_scale_bounds(self, lb=1e-6, ub=1e-2):
        self.noise_scale_bounds = np.array([np.log10(lb), np.log10(ub)])
    

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
    mu_0              = 10**theta[(gp.n_hyp_kerf + gp.n_hyp_kerg)]
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
    KGINV = compute_cholesky_inverse(L_KG)

    _, Sigma = solve_cholesky_system(KGINV + LAMBDA, np.eye(gp.ndata), gp.tych)

    # 1st contribution log N(y|0, KF + R)
    R = np.diag( np.exp( mu - 0.5 * np.diag( Sigma ) ) )
    L_KFR, alpha = solve_cholesky_system(KF + R, gp.y_norm)

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
        KFR_inv = compute_cholesky_inverse(L_KFR)
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
        elbo_grad[(gp.n_hyp_kerf + gp.n_hyp_kerg)] = 0.5 * np.einsum("ij,jik->k", inner_term1, dR_dmu0) * mu_0 * np.log(10)

        # gradient w.r.t. lambda        
        elbo_grad[(gp.n_hyp_kerf + gp.n_hyp_kerg + 1):] = 0.5 * np.einsum("ij,jik->k", inner_term1, dR_dlambda)
        elbo_grad[(gp.n_hyp_kerf + gp.n_hyp_kerg + 1):] += dtr_dlambda + dkl_dlambda

        return - elbo, - elbo_grad
    
