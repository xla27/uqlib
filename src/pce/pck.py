import os, sys, shutil, copy
import multiprocessing

import numpy as np
from scipy.linalg import lstsq, inv, cholesky, cho_solve, cho_factor, solve, solve_triangular
from scipy.stats import qmc, uniform, norm
from scipy.optimize import minimize

from SALib.sample import saltelli
from SALib.analyze import sobol

from itertools import repeat, product, combinations

from .pce import PCE

# PCKriging class
class PCKriging(PCE):

    pdf_types = ['U', 'N', 'G', 'B']

    def __init__(self, dim, degree, pdf_var, truncation, kernel, nproc=1, nugget=1e-8):

        super().__init__(dim, degree, pdf_var, truncation)

        self.kernel = kernel
        self.nugget = nugget
        self.nproc = nproc

    def training(self, X, y, method='ML'):

        self.ndata, _ = X.shape
            
        if y.ndim == 1:
            y = y[:, np.newaxis]
            self.noutputs = 1
        else:
            self.noutputs = y.shape[1]
        
        self.X_train = X
        self.y_train = y

        # information matrix
        self.F = np.zeros((self.ndata, len(self.multindices)))

        for j, poly in enumerate(self.polynomials):

            prod = 1
            for k, p in enumerate(poly):
                prod *= p(X[:,k])
            
            self.F[:,j] = prod

        # computing the hyperparameters mle
        self._hyperparameters_opt(method)

        # precomputing the fitted kernel and other matrices to perform prediction
        R = self.kernel_(self.X_train)
        R[np.diag_indices_from(R)] += self.nugget

        try:
            self.L_ = cholesky(R, lower=True, check_finite=False)
        except np.linalg.LinAlgError as exc:
            exc.args = (
                (
                    f"The kernel, {self.kernel_}, is not returning a positive "
                    "definite matrix. Try gradually increasing the 'nugget' "
                    "parameter of your RecursiveMultiFidelitySurrogate estimator."
                ),
            ) + exc.args
            raise

        # computing the parameters (beta, sigma2) mle 
        temp1 = self.F.T @ cho_solve((self.L_, True), self.y_train, check_finite=False)
        temp2 = self.F.T @ cho_solve((self.L_, True), self.F, check_finite=False)
        self.beta_hat = solve(temp2, temp1, overwrite_a=True, check_finite=False)

        self.y_disc_ = self.y_train - self.F @ self.beta_hat
        self.alpha_ = cho_solve((self.L_, True),  self.y_disc_, check_finite=False)
        self.sigma2_hat = 1/self.ndata * np.einsum("ik,ik->k", self.y_disc_, self.alpha_)

        return
    
    def predict(self, X, return_cov=False):

        f = self._predict_basis(X)

        r_trans = self.kernel_(X, self.X_train)

        y_mean = f.dot(self.beta_hat) + r_trans @ self.alpha_
    
        if return_cov:
            V = solve_triangular(self.L_, r_trans.T, lower=True, check_finite=False)
            y_cov = self.kernel_(X) - V.T @ V
            y_cov = self.sigma2_hat.reshape(1,1,self.noutputs) * np.repeat(y_cov[...,np.newaxis], self.noutputs, axis=-1) 
            
            return y_mean, y_cov

        else:
            return y_mean
    
    def moments(self, n_mc=10000):

        X = self._sample_x(n_samples=n_mc)

        y = self.predict(X, return_cov=False)

        mean = np.mean(y, axis=0)
        var  = np.var(y, axis=0)

        return np.squeeze(mean), np.squeeze(var)
    
    def sobol(self, calc_second=False, return_total=False, n_mc=1024):

        # salib problem definition
        self.sobol_problem = {}
        self.sobol_problem['num_vars'] = self.dim
        self.sobol_problem['names'] = [f'x{i+1}' for i in range(self.dim)]
        self.sobol_problem['bounds'] = []
        self.sobol_problem['dists'] = []

        for _, var in enumerate(self.pdf_var):

            if var == 'U':
                self.sobol_problem['bounds'].append([-1.0, 1.0])
                self.sobol_problem['dists'].append('unif')

            elif var == 'N':
                self.sobol_problem['bounds'].append([0.0, 1.0])
                self.sobol_problem['dists'].append('norm') 

        # saltelli sampling of inputs
        X_sobol = saltelli.sample(self.sobol_problem, 
                                  calc_second_order=calc_second, 
                                  N=n_mc)

        # evaluating the model
        y_sobol = self.predict(X_sobol, return_cov=False)

        # computing the sobol indices
        s1 = np.zeros((self.dim, self.noutputs))
        st = np.zeros((self.dim, self.noutputs))
        if calc_second:
            s2 = np.zeros((int(self.dim * (self.dim - 1) / 2), self.noutputs)) 

        for i_out in range(self.noutputs):

            s = sobol.analyze(self.sobol_problem, 
                              y_sobol[:,i_out], 
                              calc_second_order=calc_second,
                              n_processors=self.nproc)
            
            s1[:,i_out] = s['S1']
            st[:,i_out] = s['ST']
            if calc_second:
                s2[:,i_out] = s['S2']

        if calc_second:

            if return_total:
                return s1, s2, st
            else:
                return s1, s2
            
        else:

            if return_total:
                return s1, st
            else:
                return s1

    def compute_err_loo(self):

        R_inv = cho_solve((self.L_, True), np.eye(self.ndata), check_finite=False)

        err_loo = np.zeros((self.ndata, self.noutputs))

        for i_data in range(self.ndata):
            
            # removing datum from matrices
            y_loo = np.delete(self.y_train, i_data, axis=0)
            F_loo = np.delete(self.F, i_data, axis=0)
            R_inv_loo = np.delete(R_inv, i_data, axis=0)
            R_inv_loo = np.delete(R_inv_loo, i_data, axis=1)

            # regressor estimated without datum
            beta_loo = inv(F_loo.T @ R_inv_loo @ F_loo) @ F_loo.T @ R_inv_loo @ y_loo

            err_loo[i_data,:] = 1 / R_inv[i_data, i_data] * (R_inv @ (self.y_train - self.F @ beta_loo))[i_data,:]
    
        self.err_loo = err_loo

        return err_loo
    
    def compute_err_mse(self):
        '''
        Mean square error computed through Leave-one-out.
        '''

        if not hasattr(self, 'err_loo'):
            self.compute_err_loo()

        self.err_mse = np.sum(self.err_loo**2, axis=0) / self.ndata

        return self.err_mse

    def _predict_basis(self, X):

        nsamples, _ = X.shape

        f = np.zeros((nsamples, len(self.polynomials)))

        for j, poly in enumerate(self.polynomials):

            prod = 1
            for k, p in enumerate(poly):
                prod *= p(X[:,k])
            
            f[:,j] = prod

        return f

    def _sample_x(self, n_samples):
        '''
        Generating samples of inputs from standard distributions
        '''
        X = np.zeros((n_samples, self.dim))

        for i_var, var in enumerate(self.pdf_var):

            sampler = qmc.LatinHypercube(d = 1)
            samples = np.squeeze(sampler.random(n_samples))

            if var == 'U':
                #X[:,i_var] = qmc.scale(samples, np.array([-1]), np.array([1]))
                X[:,i_var] = uniform.ppf(samples, loc=-1, scale=2)
            elif var == 'N':
                X[:,i_var] = norm.ppf(samples)

        return X

    def _hyperparameters_opt(self, method):
        """
        Hyperparameters fitting via MLE or CV approach. 
        L-BFGS-B are used for optimization, with a multistart approach.
        The function, beside fitting, populates the precomputed matrices needed for mono-fidelity GPs to
        perform prediction.

        Input:
        - mfgp is the MultiFidelityGaussianProcess object that is fitted
        - doe is the DOE object
        """
        nproc = self.nproc
        multistart = max(10, nproc)

        # fitting method
        if method == 'ML':
            loss_func = log_likelihood
        elif method == 'CV':
            loss_func = loss_loo

        hyp_lw = self.kernel.bounds[:,0]
        hyp_up = self.kernel.bounds[:,1]   

        # sampling the whole set of initial points via LHS and then partion it for the number of processors
        sampler = qmc.LatinHypercube(d=hyp_lw.shape[0])
        samples = sampler.random(multistart)
        initial_points = np.repeat(hyp_lw[np.newaxis,:], multistart, axis=0) \
                        + samples * (np.repeat(hyp_up[np.newaxis,:], multistart, axis=0) -  np.repeat(hyp_lw[np.newaxis,:], multistart, axis=0))
        init_poin_list = [ initial_points[i*(int(multistart / nproc)) : (i+1)*(int(multistart / nproc))] for i in range(nproc)]
        
        # redistributing the remaing part of initial points
        assigned = (nproc)*int(multistart / nproc)
        if assigned < multistart:
            for i in range(multistart - assigned):
                init_poin_list[i] = np.vstack((init_poin_list[i], initial_points[assigned + i,:]))

        # multiprocessing for multipoint optimization
        pool = multiprocessing.Pool(processes=nproc) 
        opt_loss_tuple, opt_theta_tuple = zip(*pool.starmap(multistart_opt, 
                                                            zip(repeat(loss_func), 
                                                            repeat(copy.deepcopy(self)), 
                                                            init_poin_list)))        
        pool.close()
        pool.join()

        opt_loss_batch = np.empty(0)
        opt_theta_batch = np.empty((0, hyp_lw.shape[0]))
        for i in range(nproc):
            opt_loss_batch = np.append(opt_loss_batch, opt_loss_tuple[i], axis=0)
            opt_theta_batch = np.append(opt_theta_batch, opt_theta_tuple[i], axis=0)

        # optimal theta and loss
        opt_theta = opt_theta_batch[np.argmin(opt_loss_batch),:]
        opt_loss  = np.amin(opt_loss_batch)

        # finalizing the data structure for the GPR
        self.kernel.theta = opt_theta[:self.dim]
        self.kernel._check_bounds_params()

        # setting kernel hyperparameters
        self.kernel_ = self.kernel                                   # kernel_ is the one and only trained kernel, reference for prediction
        self.log_likelihood_value_ = - opt_loss

        return

# -------------------------------------------------------------------
#  Log Likelihood
# -------------------------------------------------------------------

def log_likelihood(theta, pck, eval_gradient=False):
    """
    Function to compute the Log Likelihood of the GP and its gradient w.r.t. the hyperparameters.
    The hyperparameters of the kernel are log-transformed, the correlation parameter is estimated via least squares.

    Inputs:
    - theta is the vector of log_transformed hyperparameters.
    - gp is the mono-fidelity GaussianProcessRegressor object
    - eval_gradient is a boolean that tells whether to compute the gradient or not

    Outputs:
    - -log_like is the negative logL 
    - -log_like_grad is the negative logL gradient
    """
    kernel = pck.kernel
    kernel.theta = theta
    
    # Kernel computation
    if eval_gradient:
        R, R_gradient = kernel(pck.X_train, eval_gradient=True)
    else:
        R = kernel(pck.X_train, eval_gradient=False)

    R[np.diag_indices_from(R)] += pck.nugget

    # Cholesky decomposition of the noisy covariance matrix
    try:
        L = cholesky(R, lower=True, check_finite=False)
    except np.linalg.LinAlgError:
        return (np.inf, np.zeros_like(theta)) if eval_gradient else np.inf

    # computing the mle of the regression coefficients
    temp1 = pck.F.T @ cho_solve((L, True), pck.y_train, check_finite=False)
    temp2 = pck.F.T @ cho_solve((L, True), pck.F, check_finite=False)
    beta_hat = solve(temp2, temp1, overwrite_a=True, check_finite=False)

    y_disc = pck.y_train - pck.F @ beta_hat
    if y_disc.ndim == 1:
        y_disc = y_disc[:, np.newaxis]

    alpha = cho_solve((L, True), y_disc, check_finite=False)
    Q_   = np.einsum("ik,ik->k", y_disc, alpha)

    # Computation of the marginal likelihood
    log_like_dims = - pck.ndata / 2 * np.log(Q_) 
    log_like_dims -= np.sum(np.log(np.diag(L))) 
    log_like_dims -= pck.ndata / 2 * (np.log(2 * np.pi / pck.ndata) + 1)
    # the log likelihood is sum-up across the outputs
    log_like = log_like_dims.sum(axis=-1)

    if eval_gradient :

        # Computation of the likelihood gradient
        # from Rasmussen & Williams, each gradient component theta_j is equal to
        #  0.5 * trace((alpha . alpha^T - K^-1) . dK_dtheta_j)       

        inner_term = pck.ndata * np.einsum("ik,jk->ijk", alpha, alpha) / Q_.reshape(1, 1, pck.noutputs)
        R_inv = cho_solve((L, True), np.eye(pck.ndata), check_finite=False)

        inner_term -= R_inv[..., np.newaxis]
        log_like_grad_dims = 0.5 * np.einsum(
            "ijl,jik->kl", inner_term, R_gradient
        )

        log_like_grad = log_like_grad_dims.sum(axis=-1)

        return -log_like, -log_like_grad
    
    else:
        return -log_like
    

# -------------------------------------------------------------------
#  Loss LOOCV
# -------------------------------------------------------------------

def loss_loo(theta, pck, eval_gradient=False):
    """
    Function to compute the Log Likelihood of the GP and its gradient w.r.t. the hyperparameters.
    The hyperparameters of the kernel are log-transformed.

    Inputs:
    - theta is the vector of log_transformed hyperparameters.
    - gp is the mono-fidelity GaussianProcessRegressor object
    - eval_gradient is a boolean that tells whether to compute the gradient or not

    Outputs:
    - -loss_loo is the negative logL 
    - -loss_loo_grad is the negative logL gradient
    """
    kernel = pck.kernel
    kernel.theta = theta
    
    # Kernel computation
    if eval_gradient:
        R, R_gradient = kernel(pck.X_train, eval_gradient=True)
    else:
        R = kernel(pck.X_train, eval_gradient=False)

    R[np.diag_indices_from(R)] += pck.nugget

    # Cholesky decomposition of the noisy covariance matrix
    try:
        L = cholesky(R, lower=True, check_finite=False)
    except np.linalg.LinAlgError:
        return (np.inf, np.zeros_like(theta)) if eval_gradient else np.inf

    y_disc = pck.y_train 

    alpha = cho_solve((L, True), y_disc, check_finite=False)
    R_inv = cho_solve((L, True), np.eye(pck.ndata), check_finite=False)
    A = R_inv**(-2) * np.eye(pck.ndata)

    # Computation of the marginal likelihood
    loss_loo_dims = np.einsum("ik,kj->ij", A, alpha)
    loss_loo_dims = np.einsum("ik,ik->k" , alpha, loss_loo_dims)
    # the log likelihood is sum-up across the outputs
    loss_loo = loss_loo_dims.sum(axis=-1)

    if eval_gradient :

        # Computation of the loss gradient  

        temp1 = np.einsum("ijk,ji->ijk", R_gradient, R_inv)     # R_gradient @ R_inv
        temp2 = np.einsum("ij,jik->ijk", R_inv, temp1)          # R_inv @ R_gradient @ R_inv
        temp3 = temp2 * np.repeat(np.eye(pck.ndata)[...,np.newaxis], theta.shape[0], axis=-1) # diag(R_inv @ R_gradient @ R_inv)
        temp4 = np.repeat(2 * A[...,np.newaxis]**(3/2), theta.shape[0], axis=-1)
         
        A_gradient = temp4 * temp3

        inner_term1 = np.einsum("ik,jk->ijk", alpha, alpha)

        inner_term2 = A_gradient 
        inner_term2 -= np.einsum("ijk,ji->ijk", R_gradient, R_inv @ A)
        inner_term2 -= np.einsum("ij,jik->ijk", R_inv @ A, R_gradient)

        loss_loo_grad_dims = np.einsum(
            "ijl,jik->kl", inner_term1, inner_term2
        )

        loss_loo_grad = loss_loo_grad_dims.sum(axis=-1)

        return -loss_loo, -loss_loo_grad
    
    else:
        return -loss_loo

# -------------------------------------------------------------------
#  Utilities
# -------------------------------------------------------------------

def multistart_opt(loss_func, pck, init_set):
    """
    Function to allow multiprocessing for sutiable loss function maximization.
    Basically, it parallelizes the for-cycle of L-BFGS-B multistart.

    Input:
    - loss_func is the function that has to be optimized
    - gp is the mono-fidelity GaussianProcessRegressor object
    - lev is the level of the MF model
    - init_set is the set of initial points LHS-sampled

    Output:
    - opt_loss is the vector of the optimized (neg)loss of each multistart
    - opt_theta is the array of the optimized hyperparameters of each multistart

    """
    hyp_lw = pck.kernel.bounds[:,0]
    hyp_up = pck.kernel.bounds[:,1]

    multistart, dim = init_set.shape

    opt_loss = np.zeros(multistart)
    opt_theta = np.zeros((multistart, dim))

    for i in range(multistart):
        log_initial = init_set[i, :]

        results = minimize(loss_func, 
                           log_initial, 
                           args=(pck, True), 
                           method="L-BFGS-B", 
                           jac=True, 
                           bounds=list(zip(hyp_lw, hyp_up)),
                           tol=1e-7, 
                           options={'disp': False, 'maxfun':1000})

        opt_loss[i]    = results.fun
        opt_theta[i,:] = (results.x).reshape(1, dim)

    return opt_loss, opt_theta