import os, sys, shutil

import numpy as np
import math
from scipy.special import legendre, hermitenorm
from scipy.linalg import lstsq, inv

from itertools import product


# PCE class
class PCE():

    pdf_types = ['U', 'N', 'G', 'B']

    def __init__(self, dim, degree, type, truncation):

        self.dim = dim        # size of input domain
        self.deg = degree     # maximum degree of polynomials

        self.type = type      # pdf of reduced variables (U, N, G, B)
        if any(t not in self.pdf_types for t in self.type):
            raise KeyError('Invalide PDF for reduced variables')
        if len(self.type) != self.dim:
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

                if self.type[k] == 'U':
                    poly.append(legendre_norm(idx))

                elif self.type[k] == 'N':
                    poly.append(hermite_norm(idx))

                elif self.type[k] == 'G':
                    raise KeyError('Laguerre polynomials for Gamma distribution not yet implemented')
                
                elif self.type[k] == 'B':
                    raise KeyError('Jacobi polynomials for Beta distribution not yet implemented')
                
            self.polynomials.append(poly)

        return

    def compute_coeffs(self, X, y, method):
        """
        X is a (ndata, dim) numpy.array
        """

        ndata, _ = X.shape

        self.X_train = X
        self.y_train = y
        self.A = np.zeros((ndata, len(self.multindices)))

        if method == 'LSQ':

            for j, poly in enumerate(self.polynomials):

                prod = 1
                for k, p in enumerate(poly):
                    prod *= p(X[:,k])
                
                self.A[:,j] = prod

            self.coeffs, _, _, _ = lstsq(self.A, self.y_train)

        elif method == 'PROJ':

            raise NotImplementedError('Projection method using quadrature not yet implemented')

    def moments(self):

        if not self.coeffs.any():
            raise KeyError('Coefficients do not exist! Surrogate yet to be built!')

        mean = self.coeffs[0]
        var  = np.sum(self.coeffs[1:]**2)

        return mean, var
    
    def predict(self, X):

        if not self.coeffs.any():
            raise KeyError('Coefficients do not exist! Surrogate yet to be built!')

        ndata, _ = X.shape
        
        y = np.zeros(ndata)

        for j, poly in enumerate(self.polynomials):

            prod = 1
            for k, p in enumerate(poly):
                prod *= p(X[:,k])
            
            y += self.coeffs[j] * prod  

        return y     
    
    def sobol_first(self):
           
        if not self.coeffs.any():
            raise KeyError('Coefficients do not exist! Surrogate yet to be built!')    

        _, V = self.moments()

        s = np.zeros(self.dim)
        for d in range(self.dim):

            for j, indices in enumerate(self.multindices):

                s[d] += self.coeffs[j]**2 if (indices[d] > 0 and np.sum([indices[i] for i in range(self.dim) if i != d]) == 0.0) else 0.0 

        return s / (V + np.finfo(float).eps)
    
    def sobol_total(self):

        if not self.coeffs.any():
            raise KeyError('Coefficients do not exist! Surrogate yet to be built!')    

        _, V = self.moments()

        s = np.zeros(self.dim)
        for d in range(self.dim):

            for j, indices in enumerate(self.multindices):

                s[d] += self.coeffs[j]**2 if indices[d] > 0 else 0.0 

        return s / (V + np.finfo(float).eps)
    
    def compute_err_loo(self):

        h = np.diag(self.A @ inv(self.A.T @ self.A) @ self.A.T)

        ndata, _ = self.X_train.shape

        err_loo = 1/ndata * np.sum(((self.y_train - self.predict(self.X_train))/(np.ones(ndata) - h))**2)

        self.err_loo = err_loo 

        return err_loo
    
    def compute_err_emp(self):

        ndata, _ = self.X_train.shape

        err_emp = 1/ndata * np.sum((self.y_train - self.predict(self.X_train)**2))

        self.err_emp = err_emp

        return err_emp
    
    def r2_score(self):

        if not hasattr(self, 'err_emp'):
            self.compute_err_emp()

        return 1 - self.err_emp / np.var(self.y_train)
    
    def q2_score(self):

        if not hasattr(self, 'err_loo'):
            self.compute_err_loo()

        return 1 - self.err_loo / np.var(self.y_train)
    

######################
# ORTHONORMAL FUNCTIONS

def legendre_norm(k):
    return legendre(k) / np.sqrt(1 / (2*k + 1))

def hermite_norm(k):
    return hermitenorm(k) / np.sqrt(math.factorial(k))


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