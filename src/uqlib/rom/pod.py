import os, sys, shutil
import numpy as np
from abc import abstractmethod
from scipy.linalg import svd
from scipy.stats import qmc, norm, uniform
import copy

from SALib.sample import sobol as sob_sample
from SALib.analyze import sobol as sob_analyze

from ..pce import PCE
from ..gp  import GaussianProcessRegressor 

class POD():
    '''
    Nomenclature:
    - X are the random input variables of the model
    - Y is the full-order output of the computational model
    - Z is the set of latent variables (obtained through POD)
    '''

    def __init__(self):
        
        self.uq_dim = None

    def compute_pod(self, Y, delta=0.9999):
        '''
        Z is an (M x N) array with:
        - M number of snaphsots
        - N output size of the computational model
        '''
        Y = np.asarray(Y)
        if Y.ndim != 2:
            raise ValueError("Y must be 2D array of shape (M, N)")

        self.ndata, self.fom_dim = Y.shape
        rank = min(self.ndata, self.fom_dim)

        # mean-normalization
        self.Y_mean = np.mean(Y, axis=0)
        self.Y_train = Y - np.repeat(self.Y_mean[np.newaxis,:], self.ndata, axis=0)

        # singular value decomposition
        U, sigma, Vh = svd(self.Y_train.T)

        d = 1
        for d in range(1, sigma.shape[0]+1):
            ric = np.sum(sigma[:d]**2) / np.sum(sigma[:rank]**2)
            if ric > delta:
                self.pod_dim = d      # latent space dimensionality
                break

        self.modes = U[:,:self.pod_dim]               # POD modes (fom_dim, rom_dim) array
        self.Z_train = self.Y_train @ self.modes      # latent basis (ndata, rom_dim) array

    @abstractmethod
    def moments(self):
        return
    
    @abstractmethod
    def predict_latent(self, X):
        return
    
    def predict(self, X):

        nsamples, _ = X.shape

        Y_pred = np.repeat(self.Y_mean[np.newaxis,:], repeats=nsamples, axis=0)
        Y_pred += self.predict_latent(X) @ self.modes.T

        return np.squeeze(Y_pred)
    
    def sobol(self, calc_second=False, return_total=False, n_mc=1024, nproc=1):
        '''
        Sobol' indices estimation through Saltelli's sampling
        '''
        # problem definition (SALib)
        sobol_problem = {
            'num_vars': self.uq_dim,
            'names'   : [f'x{i+1}' for i in range(self.uq_dim)],
            'bounds'  : [],
            'dists'   : []
        }

        for _, var in enumerate(self.pdf_var):

            if var == 'U':
                sobol_problem['bounds'].append([-1.0, 1.0])
                sobol_problem['dists'].append('unif')

            elif var == 'N':
                sobol_problem['bounds'].append([0.0, 1.0])
                sobol_problem['dists'].append('norm') 

        # samples generation in UQ space
        X_sobol = sob_sample.sample(sobol_problem,
                                    n_mc,
                                    calc_second_order=calc_second)
        
        # FOM prediction
        Y_sobol = self.predict(X_sobol)

        # computing Sobol' indices
        s1 = np.zeros((self.uq_dim, self.fom_dim))
        st = np.zeros((self.uq_dim, self.fom_dim))
        if calc_second:
            s2 = np.zeros((int(self.uq_dim * (self.uq_dim - 1) / 2), self.fom_dim)) 

        for i_out in range(self.fom_dim):

            s = sob_analyze.analyze(sobol_problem, 
                              Y_sobol[:,i_out], 
                              calc_second_order=calc_second,
                              n_processors=nproc)
            
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



class PODPCE(POD):
    '''
    POD-based reduced order model with uncertain random input
    modeled through PCE.

    Nomenclature:
    - X are the random input variables of the model
    - Y is the full-order output of the computational model
    - Z is the set of latent variables (obtained through POD)
    '''

    def __init__(self, uq_dim, pce_degree, pdf_var, truncation):

        self.uq_dim  = uq_dim
        self.pdf_var = pdf_var

        # building a PCE object
        self.pce = PCE(uq_dim, pce_degree, pdf_var, truncation)

    def compute_pce(self, X, method, weights=None):

        if not hasattr(self, 'Z_train'):
            raise KeyError('You have to first do the POD!')
        
        # computing the PCE coefficients on the latent space basis
        self.pce.compute_coeffs(X, self.Z_train, method=method, weights=weights)

    def moments(self):

        mean = self.Y_mean + self.modes @ self.pce.moments()[0]

        variance = np.zeros(self.fom_dim)

        for i in range(self.pod_dim):
            for j in range(self.pod_dim):
                pce_variance = np.sum(self.pce.coeffs[1:,i] * self.pce.coeffs[1:,j])
                variance += self.modes[:,i] * self.modes[:,j] * pce_variance

        return mean, variance
    
    def predict_latent(self, X):

        return np.atleast_2d(self.pce.predict(X))
    


class PODGPR(POD):
    '''
    POD-based reduced order model with uncertain random input
    modeled through GPR.

    Nomenclature:
    - X are the random input variables of the model
    - Y is the full-order output of the computational model
    - Z is the set of latent variables (obtained through POD)
    '''

    def __init__(self, uq_dim, kernel, pdf_var, nproc, reg_tych):

        self.uq_dim  = uq_dim
        self.kernel  = kernel
        self.nproc   = nproc
        self.tych    = reg_tych
        self.pdf_var = pdf_var

    def compute_gpr(self, X):

        self.X_train = X

        if not hasattr(self, 'Z_train'):
            raise KeyError('You have to first do the POD!')
        
        self.gprs = []

        for d in range(self.pod_dim):

            gp = GaussianProcessRegressor(copy.deepcopy(self.kernel),
                                          nproc=self.nproc,
                                          reg_tych=self.tych)
            
            gp.fit(self.X_train, self.Z_train[:,d])

            self.gprs.append(gp)

    def predict_latent(self, X):
        
        nsamples, _ = X.shape

        Z_pred = np.zeros((nsamples, self.pod_dim))

        for d, gp in enumerate(self.gprs):

            Z_pred[:,d] = np.squeeze(gp.predict(X, return_cov=False))

        return np.atleast_2d(Z_pred)
    
    def moments(self, nsamples=1000):
        
        X_samples = self._sample_x(nsamples=nsamples)

        Y_pred = self.predict(X_samples) 

        mean = np.mean(Y_pred, axis=0)
        var  = np.var( Y_pred, axis=0)

        return mean, var    

    def _sample_x(self, nsamples):
        '''
        Generating samples of inputs from standard distributions
        '''
        X = np.zeros((nsamples, self.uq_dim))

        for i_var, var in enumerate(self.pdf_var):

            sampler = qmc.LatinHypercube(d = 1)
            samples = np.squeeze(sampler.random(nsamples))

            if var == 'U':
                X[:,i_var] = uniform.ppf(samples, loc=-1, scale=2)
            elif var == 'N':
                X[:,i_var] = norm.ppf(samples)

        return X
