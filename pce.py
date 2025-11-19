import os, sys, shutil

import numpy as np
import math
from scipy.special import legendre, hermitenorm, genlaguerre, jacobi, gamma
from scipy.linalg import lstsq, inv, cholesky, cho_solve, cho_factor

from itertools import product, combinations


# PCE class
class PCE():

    pdf_types = ['U', 'N', 'G', 'B']

    def __init__(self, dim, degree, pdf_var, truncation):

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

        # polynomial bases
        self.polynomials = []
        for j, indices in enumerate(self.multindices):

            poly = []

            for k, idx in enumerate(indices):

                if self.pdf_var[k] == 'U':
                    poly.append(legendre_norm(idx))

                elif self.pdf_var[k] == 'N':
                    poly.append(hermite_norm(idx))

                elif self.type[k] == 'G':
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

        if method == 'LSQ':

            self.A = np.zeros((self.ndata, len(self.multindices)))

            for j, poly in enumerate(self.polynomials):

                prod = 1
                for k, p in enumerate(poly):
                    prod *= p(X[:,k])
                
                self.A[:,j] = prod

            ATA = self.A.T @ self.A
            ATAinv = cho_solve((cho_factor(ATA)),
                                np.eye(len(self.multindices)),
                                check_finite=False)

            # coeffs = (npoly x noutputs) array
            self.coeffs = ATAinv @ self.A.T @ self.y_train   

            self.h_loo = np.diag(self.A @ ATAinv @ self.A.T)

        elif method == 'PROJ':

            if weights is None:

                raise ValueError('Provide weights for Gaussian quadrature!')
            
            self.coeffs = np.zeros(len(self.polynomials))

            yw = self.y_train * weights

            for j, poly in enumerate(self.polynomials): 

                prod = 1
                for k, p in enumerate(poly):
                    prod *= p(X[:,k])

                self.coeffs[j,:] = np.sum(yw * prod, axis=0)

    def moments(self):
        '''
        mean, var are (noutputs,) arrays
        '''

        if not self.coeffs.any():
            raise KeyError('Coefficients do not exist! Surrogate yet to be built!')

        mean = self.coeffs[0,:]
        var  = np.sum(self.coeffs[1:,:]**2, axis=0)

        return np.squeeze(mean), np.squeeze(var)
    
    def predict(self, X):
        """
        X is a (nsamples, dim) numpy.array of inputs
        y is a (nsamples, noutputs) numpy.array of predictions
        """

        if not self.coeffs.any():
            raise KeyError('Coefficients do not exist! Surrogate yet to be built!')

        nsamples, _ = X.shape
        
        y = np.zeros((nsamples, self.noutputs))

        for j, poly in enumerate(self.polynomials):

            prod = 1
            for k, p in enumerate(poly):
                prod *= p(X[:,k])
            
            y += (np.repeat(self.coeffs[j,:][np.newaxis,:], repeats=nsamples, axis=0) * 
                  np.repeat(prod[:,np.newaxis], repeats=self.noutputs, axis=1))

        return np.squeeze(y)     
    
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
        loo_predict = np.zeros((self.ndata, self.noutputs))
        for i_out in range(self.noutputs):
            loo_predict[:, i_out] = (self.predict(self.X_train)[:,i_out] - self.h_loo * self.y_train[:,i_out])/((np.ones(self.h_loo.shape) - self.h_loo))
    
        return np.squeeze(loo_predict)

    def compute_err_l1(self):
        '''
        L1 error computed through Leave-one-out.
        '''
        err_l1 = np.zeros(self.noutputs)
        for i_out in range(self.noutputs):
            err_l1[i_out] = np.sum(np.abs((self.y_train[:,i_out] - self.predict(self.X_train)[:,i_out])/(np.ones(self.h_loo.shape) - self.h_loo)))

        self.err_l1 = np.squeeze(err_l1)

        return np.squeeze(err_l1)
    
    def compute_err_l2(self):
        '''
        L2 error computed through Leave-one-out.
        '''
        err_l2 = np.zeros(self.noutputs)
        for i_out in range(self.noutputs):
            err_l2[i_out] = np.sum(((self.y_train[:,i_out] - self.predict(self.X_train)[:,i_out])/(np.ones(self.h_loo.shape) - self.h_loo))**2)

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

        self.err_mse = self.err_l2 / self.ndata

        return self.err_mse
    
    def compute_err_rmse(self):
        
        if not hasattr(self, 'err_mse'):
            self.compute_err_mse()

        return np.sqrt(self.err_mse)
    
    def r2_score(self):

        if not hasattr(self, 'err_mse'):
            self.compute_err_mse()

        score = np.zeros(self.noutputs)
        for i_out in range(self.noutputs):
            score[i_out] = 1 - self.err_mse[i_out] / np.var(self.y_train[:,i_out])

        return np.squeeze(score)
    

    

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