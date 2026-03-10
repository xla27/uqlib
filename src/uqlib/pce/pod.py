import os, sys, shutil
import numpy as np
from scipy.linalg import svd
import copy

from . import PCE
 

class PODPCE():
    '''
    Nomenclature:
    - X are the random input variables of the model
    - Y is the full-order output of the computational model
    - Z is the set of latent variables (obtained through isomap)
    '''

    def __init__(self, dim, degree, pdf_var, truncation):

        self.dim     = dim
        self.pdf_var = pdf_var

        # building a PCE object
        self.pce = PCE(dim, degree, pdf_var, truncation)

    def compute_pod(self, Y, delta=0.9999):
        '''
        Z is an (M x N) array with:
        - M number of snaphsots
        - N output size of the computational model
        '''
        Y = np.asarray(Y)
        if Y.ndim != 2:
            raise ValueError("Y must be 2D array of shape (M, N)")

        self.ndata, self.noutputs = Y.shape
        rank = min(self.ndata, self.noutputs)

        # mean-normalization
        self.Y_mean = np.mean(Y, axis=0)
        self.snapshots = Y - np.repeat(self.Y_mean[np.newaxis,:], self.ndata, axis=0)

        # singular value decomposition
        U, sigma, Vh = svd(self.snapshots.T)

        d = 1
        for d in range(1, sigma.shape[0]+1):
            ric = np.sum(sigma[:d]**2) / np.sum(sigma[:rank]**2)
            if ric > delta:
                self.pod_degree = d      # latent space dimensionality
                break

        self.modes = U[:,:self.pod_degree]             # POD modes (Nxd) array
        self.latent = self.modes.T @ self.snapshots.T  # latent basis (dxM) array

        return

    def compute_pce(self, X, method, weights=None):

        if not hasattr(self, 'latent'):
            raise KeyError('You to first do the POD!')
        
        # computing the PCE coefficients on the latent space basis
        self.pce.compute_coeffs(X, self.latent.T, method=method, weights=weights)
        
    def mean(self):

        mean = self.Y_mean + self.modes @ self.pce.moments()[0]

        return mean
    
    def variance(self):

        variance = np.zeros(self.noutputs)

        for i in range(self.pod_degree):
            for j in range(self.pod_degree):
                pce_variance = np.sum(self.pce.coeffs[1:,i] * self.pce.coeffs[1:,j])
                variance += self.modes[:,i] * self.modes[:,j] * pce_variance

        return variance
    
    def predict(self, X):

        nsamples, _ = X.shape

        Y_pred = np.repeat(self.Y_mean[np.newaxis,:], repeats=nsamples, axis=0)

        Y_pred += np.transpose( self.modes @ self.pce.predict(X).T )

        return Y_pred
