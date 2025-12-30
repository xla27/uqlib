import os, sys, shutil, copy

import numpy as np

from itertools import product, combinations

from .pce import PCE, legendre_norm, hermite_norm

# MFPCE class
class MFPCE():

    pdf_types = ['U', 'N', 'G', 'B']

    def __init__(self, dim, nlevels, degrees, pdf_var, truncations):
        '''
        Lists from lowest level to highest level!

        Inputs:
        - dim -> dimensionality of the input random space
        - nlevels -> number of levels 
        - degree -> (list of) degree(s) for the PCE
        - pdf_var -> list of the PDF of each random input ('U','N','G','B')
        - truncation -> (list of) dictionary with the truncation method ('standard', 'rank','hyperbolic')
                        and additional parameters ('rank', 'q')
        '''

        self.dim     = dim  
        self.nlevels = nlevels

        if isinstance(degrees, list) and len(degrees) == nlevels:      
            self.degs = degrees
        elif not isinstance(degrees, list):
            self.degs = [degrees] * nlevels
        else:
            raise KeyError('Wrong degrees specification')
        
        if isinstance(truncations, list) and len(truncations) == nlevels:      
            self.truncations = truncations
        elif not isinstance(truncations, list):
            self.truncations = [truncations] * nlevels
        else:
            raise KeyError('Wrong truncation schemes specification')

        self.pdf_var = pdf_var      # pdf of reduced variables (U, N, G, B)
        if any(t not in self.pdf_types for t in self.pdf_var):
            raise KeyError('Invalide PDF for reduced variables')
        if len(self.pdf_var) != self.dim:
            raise KeyError('Unspecified PDF kind for input size')

        return

    def compute_coeffs(self, doe, method):
        """
        doe is a NestedDoE object

        coeffs is a (degree, noutputs) numpy.array of noutputs independent PCEs
        """

        # training the lowest level PCE
        pce_lf = copy.deepcopy(PCE(self.dim, 
                                   self.degs[0],
                                   self.pdf_var, 
                                   self.truncations[0]))
        
        X_train, y_train = doe.level(0, return_y=True)

        if y_train.ndim == 1:
            self.noutputs = 1
            y_train = y_train[:,np.newaxis]
        else:
            _, self.noutputs = y_train.shape

        pce_lf.compute_coeffs(X_train, y_train, method=method)
        
        for i_lev in range(1, self.nlevels):

            pce_i = copy.deepcopy(PCE(self.dim, 
                                    self.degs[i_lev],
                                    self.pdf_var, 
                                    self.truncations[i_lev]))

            # extracting the data
            X_train, y_train = doe.level(i_lev, return_y=True)

            # discrepancy wrt lower PCE prediction
            y_lf = pce_lf.predict(X_train)
            if self.noutputs == 1: 
                y_lf = y_lf[:,np.newaxis]
            disc_train = y_train - y_lf

            pce_i.compute_coeffs(X_train, disc_train, method=method)

            # updating the pce_lf
            pce_lf = self._update_pce(pce_lf, pce_i)

        # assigning the correct attributes
        self.multindices = pce_lf.multindices
        self.polynomials = pce_lf.polynomials
        self.coeffs      = pce_lf.coeffs            

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
    
    def _update_pce(self, pce_lf, pce_i):

        acc = {}

        multindices_lf = pce_lf.multindices
        coeffs_lf = [pce_lf.coeffs[i,:] for i in range(pce_lf.coeffs.shape[0])]
        for a, ya in zip(multindices_lf, coeffs_lf):
            key = tuple(a)
            if key in acc:
                acc[key] += ya.copy()
            else:
                acc[key] = ya.copy()   

        multindices_i = pce_i.multindices
        coeffs_i = [pce_i.coeffs[i,:] for i in range(pce_i.coeffs.shape[0])]
        for b, yb in zip(multindices_i, coeffs_i):
            key = tuple(b)
            if key in acc:
                acc[key] += yb.copy()
            else:
                acc[key] = yb.copy()  

        pce_lf.multindices = [list(k) for k in acc.keys()]

        # polynomial bases
        polynomials = []
        for j, indices in enumerate(pce_lf.multindices):

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
                
            polynomials.append(poly)
        
        pce_lf.polynomials = polynomials        
        pce_lf.coeffs      = np.array([val for val in acc.values()])

        return pce_lf