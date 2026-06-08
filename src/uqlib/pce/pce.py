import os, sys, shutil

import numpy as np
import math
from scipy.special import legendre, hermitenorm, genlaguerre, jacobi, gamma
from scipy.stats import qmc, uniform, norm
from scipy.linalg import cho_solve, cho_factor

from itertools import product, combinations


# PCE class
class PCE():

    pdf_types = ['U', 'N', 'G', 'B']

    def __init__(self, dim, degree, pdf_var, truncation):
        '''
        Inputs:
        - dim -> dimensionality of the input random space
        - degree -> degree of the PCE
        - pdf_var -> list of the PDF of each random input ('U','N','G','B')
        - truncation -> dictionary with the truncation method ('standard', 'rank','hyperbolic')
                        and additional parameters ('rank', 'q')
        '''

        self.dim = dim        # size of input domain
        self.deg = degree     # maximum degree of polynomials

        self.pdf_var = pdf_var      # pdf of reduced variables (U, N, G, B)
        if any(t not in self.pdf_types for t in self.pdf_var):
            raise KeyError('Invalide PDF for reduced variables')
        if len(self.pdf_var) != self.dim:
            raise KeyError('Unspecified PDF kind for input size')

        # truncation scheme
        if truncation['method'] == 'standard':
            self.multindices = trunc_standard(self.dim, self.deg)
        elif truncation['method'] == 'rank':
            self.multindices = trunc_rank(self.dim, self.deg, truncation['rank'])
        elif truncation['method'] == 'hyperbolic':
            self.multindices = trunc_hyper(self.dim, self.deg, truncation['q'])

        print('PCE surrogate with %i inputs, %i degree, %i cardinality' %
              (self.dim, self.deg, len(self.multindices)))

        # polynomial bases
        self.polynomials = []
        for j, indices in enumerate(self.multindices):

            poly = []

            for k, idx in enumerate(indices):

                if self.pdf_var[k] == 'U':
                    poly.append(legendre_norm(idx))

                elif self.pdf_var[k] == 'N':
                    poly.append(hermite_norm(idx))

                elif self.pdf_var[k] == 'G':
                    raise KeyError('Laguerre polynomials for Gamma distribution not yet implemented')
                
                elif self.pdf_var[k] == 'B':
                    raise KeyError('Jacobi polynomials for Beta distribution not yet implemented')
                
            self.polynomials.append(poly)

        return

    def compute_coeffs(self, X, y, method, weights=None):
        """
        X is a (ndata, dim) numpy.array of inputs
        y is a (ndata, noutputs) numpy.array of training outputs

        coeffs is a (degree, noutputs) numpy.array of noutputs independent PCEs
        """

        self.ndata, _ = X.shape
        if y.ndim == 1:
            self.noutputs = 1
            y = y[:,np.newaxis]
        else:
            _, self.noutputs = y.shape

        self.X_train = X
        self.y_train = y

        if method == 'LSQ' or method == 'LSQ-W':

            self.A = np.zeros((self.ndata, len(self.multindices)))
            self.W = np.zeros((self.ndata, self.ndata))

            for j, poly in enumerate(self.polynomials):

                prod = 1
                for k, p in enumerate(poly):
                    prod *= p(X[:,k])
                
                self.A[:,j] = prod

            if method == 'LSQ-W':
                self.W = np.diag(self.ndata / np.sum(self.A**2, axis=1))
            else:
                self.W = np.eye(self.ndata)

            ATWA = self.A.T @ self.W @ self.A
            ATWAinv = cho_solve((cho_factor(ATWA)),
                                np.eye(len(self.multindices)),
                                check_finite=False)

            # coeffs = (npoly x noutputs) array
            self.coeffs = ATWAinv @ self.A.T @ self.W @ self.y_train   

            # utilities for loo from Sudret
            self.h_loo          = np.diag(self.A @ ATWAinv @ self.A.T @ self.W)
            utility_loo         = np.ones(self.h_loo.shape) - self.h_loo
            self.utility_loo    = np.repeat(utility_loo[:,np.newaxis], repeats=self.noutputs, axis=1)

            self.correction = self.ndata / (self.ndata - len(self.multindices)) * (1 + np.linalg.trace(ATWAinv) / self.ndata)

        elif method == 'QUAD':

            if weights is None:

                raise ValueError('Provide weights for Gaussian quadrature!')
            
            else:

                weights = np.repeat(weights[:,np.newaxis], self.noutputs, axis=1)
            
            self.coeffs = np.zeros((len(self.polynomials), self.noutputs))

            yw = weights * self.y_train 

            for j, poly in enumerate(self.polynomials): 

                prod = 1
                for k, p in enumerate(poly):
                    prod *= p(X[:,k])
                
                prod = np.repeat(prod[:,np.newaxis], self.noutputs, axis=1)

                self.coeffs[j,:] = np.sum(prod * yw, axis=0)

        # computing a prediction on the training set (useful to be stored)
        X_train_predict = self.predict(self.X_train)
        if self.noutputs == 1:
            self.X_train_predict = X_train_predict[:,np.newaxis]
        else:
            self.X_train_predict = X_train_predict

    def moments(self):
        '''
        mean, var are (noutputs,) arrays
        '''

        if not self.coeffs.any():
            raise KeyError('Coefficients do not exist! Surrogate yet to be built!')

        mean = self.coeffs[0,:]
        var  = np.sum(self.coeffs[1:,:]**2, axis=0)

        return np.squeeze(mean), np.squeeze(var)
    
    def predict(self, X, eval_gradient=False):
        """
        X is a (nsamples, dim) numpy.array of inputs
        y is a (nsamples, noutputs) numpy.array of predictions
        grad is a (nsamples, noutputs, ndim) numpy.array of the PCE gradient for given sample for given output
        """

        if not self.coeffs.any():
            raise KeyError('Coefficients do not exist! Surrogate yet to be built!')

        nsamples, _ = X.shape
        
        y    = np.zeros((nsamples, self.noutputs))
        grad = np.zeros((nsamples, self.dim, self.noutputs))

        for j, poly in enumerate(self.polynomials):

            grad_j = np.zeros((nsamples, self.dim))
            prod_j = 1

            for k, p in enumerate(poly):
                poly_k  = p(X[:,k])
                prod_j *= poly_k

                if eval_gradient:
                    grad_k  = np.polyder(p)(X[:,k])
                    grad_j[:,k] = grad_k/poly_k

            coeffs_j = np.repeat(self.coeffs[j,:][np.newaxis,:], repeats=nsamples, axis=0)
            
            y += (coeffs_j * np.repeat(prod_j[:,np.newaxis], repeats=self.noutputs, axis=1))

            if eval_gradient:
                grad_j *= np.repeat(prod_j[:,np.newaxis], repeats=self.dim, axis=1)
                grad += (np.repeat(coeffs_j[:,np.newaxis,:], repeats=self.dim, axis=1) * 
                    np.repeat(grad_j[...,np.newaxis], repeats=self.noutputs, axis=2))

        if eval_gradient:
            return np.squeeze(y), np.squeeze(grad)
        
        else:
            return np.squeeze(y)      
    
    def sample(self, nsamples):

        X = self._sample_x(nsamples=nsamples)

        return self.predict(X, eval_gradient=False)

    def sobol_first(self):
        """
        s is a (dim, noutputs) numpy.array of first order Sobol' indices
        """
           
        if not self.coeffs.any():
            raise KeyError('Coefficients do not exist! Surrogate yet to be built!')    

        _, V = self.moments()

        s = np.zeros((self.dim, self.noutputs))
        for d in range(self.dim):

            for j, indices in enumerate(self.multindices):

                s[d,:] += self.coeffs[j,:]**2 if (indices[d] > 0 and np.sum([indices[i] for i in range(self.dim) if i != d]) == 0.0) else 0.0 

        return s / (V + np.finfo(float).eps)
    
    def sobol_second(self):
        """
        s is a (dim*(dim-1)/2, noutputs) numpy.array of second order Sobol' indices
        """

        if not self.coeffs.any():
            raise KeyError('Coefficients do not exist! Surrogate yet to be built!')    

        _, V = self.moments()

        pairs = combinations(range(self.dim), 2)

        s = np.zeros((int(self.dim * (self.dim - 1) / 2), self.noutputs))
        for d, pair in enumerate(pairs):

            for j, indices in enumerate(self.multindices):

                s[d,:] += self.coeffs[j,:]**2 if (indices[pair[0]] > 0 and indices[pair[1]] > 0 and np.sum([indices[i] for i in range(self.dim) if i not in pair]) == 0.0) else 0.0

        return s / (V + np.finfo(float).eps)
    
    def sobol_total(self):
        """
        s is a (dim, noutputs) numpy.array of total Sobol' indices
        """

        if not self.coeffs.any():
            raise KeyError('Coefficients do not exist! Surrogate yet to be built!')    

        _, V = self.moments()

        s = np.zeros((self.dim, self.noutputs))
        for d in range(self.dim):

            for j, indices in enumerate(self.multindices):

                s[d,:] += self.coeffs[j,:]**2 if indices[d] > 0 else 0.0 

        return s / (V + np.finfo(float).eps)
    
    def loo_predict(self):
        '''
        Leave-one-out prediction i.e., the prediction at x[i] from a surrogate trained without y_train[i]
        M^{PC-i} = ( M^{PC}(x^{(i)}) - h_i M(x^{(i)} ) / ( 1 - h_i )
        '''
        h_loo = np.repeat(self.h_loo[:,np.newaxis], repeats=self.noutputs, axis=1)
        loo_predict = (self.X_train_predict - h_loo * self.y_train) / self.utility_loo

        return np.squeeze(loo_predict)

    def compute_err_l1(self):
        '''
        L1 error computed through Leave-one-out.
        '''
        err_l1 = np.sum(np.abs((self.y_train - self.X_train_predict)/ self.utility_loo), axis=0)

        self.err_l1 = np.squeeze(err_l1)

        return np.squeeze(err_l1)
    
    def compute_err_l2(self):
        '''
        L2 error computed through Leave-one-out.
        '''
        err_l2 = np.sum(((self.y_train - self.X_train_predict)/ self.utility_loo)**2, axis=0)

        self.err_l2 = np.squeeze(err_l2)

        return np.squeeze(err_l2)
    
    def compute_err_mae(self):
        '''
        Mean absolute error computed through Leave-one-out.
        '''

        if not hasattr(self, 'err_l1'):
            self.compute_err_l1()

        self.err_mae = self.err_l1 / self.ndata

        return self.err_mae
    
    def compute_err_mse(self):
        '''
        Mean square error computed through Leave-one-out.
        '''

        if not hasattr(self, 'err_l2'):
            self.compute_err_l2()

        self.err_mse = self.correction * self.err_l2 / self.ndata

        return self.err_mse
    
    def compute_err_rmse(self):
        
        if not hasattr(self, 'err_mse'):
            self.compute_err_mse()

        return np.sqrt(self.err_mse)
    
    def r2_score(self):

        if not hasattr(self, 'err_mse'):
            self.compute_err_mse()

        y_train_var = np.var(self.y_train, axis=0)

        score = np.ones(self.noutputs) - self.err_mse / np.where(y_train_var == 0, 1e-12, y_train_var)

        return np.squeeze(score)
    
    def _sample_x(self, nsamples):
        '''
        Generating samples of inputs from standard distributions
        '''
        X = np.zeros((nsamples, self.dim))

        for i_var, var in enumerate(self.pdf_var):

            sampler = qmc.LatinHypercube(d = 1)
            samples = np.squeeze(sampler.random(nsamples))

            if var == 'U':
                X[:,i_var] = uniform.ppf(samples, loc=-1, scale=2)
            elif var == 'N':
                X[:,i_var] = norm.ppf(samples)

        return X
    

    

######################
# ORTHONORMAL FUNCTIONS

def legendre_norm(k):
    return legendre(k, monic=False) / np.sqrt(1 / (2*k + 1))

def hermite_norm(k):
    return hermitenorm(k, monic=False) / np.sqrt(math.factorial(k))

def laguerre_norm(k, a=1.0):
    return genlaguerre(k, a, monic=False) / np.sqrt(gamma(k+a+1) / math.factorial(k))


######################
# TRUNCATION SCHEMES

def trunc_standard(dim, deg):
    combinations = list( product( range(0, deg+1), repeat=dim) )
    multindices = [combo for combo in combinations if np.sum(combo) <= deg]
    return multindices

def trunc_rank(dim, deg, rank):
    multindices_standard = trunc_standard(dim, deg)
    multindices = []
    for indices in multindices_standard:
        if np.sum(np.where(np.asarray(indices) > 0, 1, 0)) <= rank:
            multindices.append(indices)   
    return multindices

def trunc_hyper(dim, deg, q):
    combinations = list( product( range(0, deg+1), repeat=dim) )
    multindices = [combo for combo in combinations if np.linalg.norm(combo, q) <= deg]
    return multindices