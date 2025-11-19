import os, sys, shutil
import numpy as np
from scipy.linalg import svd
import copy

from . import PCE
 

class PODPCE():

    def __init__(self, dim, degree, pdf_var, truncation):

        self.pce_parameters = {'dim':        dim,
                               'degree':     degree,
                               'pdf_var':    pdf_var,
                               'truncation': truncation}

    def compute_pod(self, Z, delta=0.9999):

        dim_pod, ndata_pod = Z.shape
        rank = min(dim_pod, ndata_pod)

        # mean-normalization
        self.Z_mean = np.mean(Z, axis = 1)
        Z_clean = Z - np.repeat(self.Z_mean[:,np.newaxis], repeats=ndata_pod, axis=1)

        # singular value decomposition
        U, sigma, Vh = svd(Z_clean)

        d = 1
        for d in range(1, dim_pod):
            ric = np.sum(sigma[:d]**2) / np.sum(sigma[:rank]**2)
            if ric > delta:
                break

        self.pod_degree = d

        self.modes = U[:,:self.pod_degree]
        self.latent = self.modes.T @ Z_clean

        return

    def compute_pce(self, X, method, weights=None):

        if not hasattr(self, 'latent'):
            raise KeyError('You to first do the POD!')
        
        # generating a list of pce
        self.pces = [copy.deepcopy(PCE(**self.pce_parameters)) for i in range(self.pod_degree)]
        
        for i_pce, pce in enumerate(self.pces):
            y_train = self.latent[i_pce,:]
            pce.compute_coeffs(X, y_train, method=method, weights=weights)
        
    def mean(self):
        mean = self.Z_mean

        for i_pce, pce in enumerate(self.pces):
            mean += self.modes[:, i_pce] * pce.moments()[0]
        
        return mean
    
    def variance(self):
        variance = np.zeros(self.Z_mean.shape[0])

        for i_pce, pce_i in enumerate(self.pces):
            variance += self.modes[:, i_pce]**2 * pce_i.moments()[1]

            for j_pce, pce_j in enumerate(self.pces):
                if j_pce != i_pce:
                    covariance_ij = np.sum(pce_i.coeffs[1:] * pce_j.coeffs[1:])
                    variance += self.modes[:,i_pce] * self.modes[:,j_pce] * covariance_ij
                else:
                    variance += 0.0
        
        return variance
    
    def predict(self, X):

        npred, _ = X.shape
        Z_pred = np.repeat(self.Z_mean[:,np.newaxis], repeats=npred, axis=1)

        for i_pce, pce in enumerate(self.pces): 
            Z_pred += np.repeat(self.modes[:, i_pce][:,np.newaxis], repeats=npred, axis=1) * pce.predict(X)

        return Z_pred
