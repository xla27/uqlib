import os, sys, copy
import numpy as np
from abc import abstractmethod
from scipy.stats import qmc, norm, uniform

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import TensorDataset, DataLoader, DistributedSampler

import socket

from ..pce import PCE
from ..gp  import GaussianProcessRegressor
from .common import sobol_wrapper

class AE():

    def __init__(self, uq_dim, rom_dim):

        self.uq_dim = uq_dim
        self.rom_dim = rom_dim
        self.autoencoder = None

    def load_ae(self, autoencoder, dataset, device, filename):

        self.dataset = dataset

        self.device = device

        checkpoint = torch.load(filename)

        autoencoder.load_state_dict(checkpoint)

        autoencoder.to(device)

        autoencoder.eval()

        self.autoencoder = autoencoder

        with torch.no_grad():
            Z_train = self.autoencoder.encoder(self.Y_train.to(self.device)).cpu()

        self.Z_train = Z_train

    @abstractmethod
    def predict_latent(self, X):
        '''
        Latent prediction from random input sample
        X is a (nsamples, uq_dim) numpy.array of inputs
        Z is a (nsamples, rom_dim) numpy.array of predictions
        '''
        return
    
    def predict(self, X):
        '''
        FOM prediction from random input sample
        X is a (nsamples, uq_dim) numpy.array of inputs
        Y is a (nsamples, fom_dim) numpy.array of predictions
        '''

        if X.ndim == 1:
            X = X[np.newaxis,:]

        nsamples, _ = X.shape

        Y_pred = np.zeros((nsamples, self.nfields, self.nx, self.ny))

        # predicting the latent variable through PCE 
        Z_pred = self.predict_latent(X)

        # fixing Z_pred size, decoder input size: (nsamples, rom_dim)
        if nsamples == 1 and self.rom_dim == 1:
            Z_pred = np.atleast_2d(Z_pred)
        elif nsamples > 1 and self.rom_dim == 1:
            Z_pred = Z_pred[:,np.newaxis]
        elif nsamples == 1 and self.rom_dim > 1:
            Z_pred = np.atleast_2d(Z_pred)

        with torch.no_grad():
            Y_pred = self.autoencoder.decoder(
                torch.tensor(Z_pred, dtype=torch.float32).to(self.device)
            ).numpy()

        Y_pred = self._denormalize_y()

        return Y_pred

    def moments(self, nsamples=1000):
        
        X_samples = self._sample_x(nsamples=nsamples)

        Y_pred = self.predict(X_samples) 

        mean = np.mean(Y_pred, axis=0)
        var  = np.var( Y_pred, axis=0)

        return mean[np.newaxis,...], var[np.newaxis,...]   

    def sobol(self, calc_second=False, return_total=False, n_mc=1024, nproc=1):
        '''
        Sobol' indices estimation through Saltelli's sampling
        '''
        return sobol_wrapper(self, 
                             calc_second=calc_second, 
                             return_total=return_total,
                             n_mc=n_mc,
                             nproc=nproc)

    def _denormalize_y(self, Y):
        '''
        Denormalize FOM Y predictions, dataset must be an
        instantiation of torch.utils.data.Dataset with a 
        denormalize() method
        '''

        return self.dataset.denormalize(Y)

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


class AEPCE(AE):
    '''
    AE-based reduced order model with uncertain random input
    modeled through PCE.

    Nomenclature:
    - X are the random input variables of the model
    - Y is the full-order output of the computational model
    - Z is the set of latent variables (obtained through AE)
    ''' 

    def __init__(self, uq_dim, rom_dim, pce_degree, pdf_var, truncation):

        self.uq_dim = uq_dim
        self.pdf_var = pdf_var
        self.rom_dim = rom_dim

        self.device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

        self.pce = PCE(uq_dim, pce_degree, pdf_var, truncation)

    def compute_pce(self, X, method, weights=None):

        self.X_train = X

        if not hasattr(self, 'Z_train'):
            raise KeyError('You have to first train the VAE!')
    
        # computing the PCE coefficients on the latent space basis
        self.pce.compute_coeffs(X, self.Z_train.numpy(), method=method, weights=weights)

    def predict_latent(self, X):
        '''
        Latent prediction from random input sample
        X is a (nsamples, uq_dim) numpy.array of inputs
        Z is a (nsamples, rom_dim) numpy.array of predictions
        '''
        return self.pce.predict(X)
    

class AEGPR(AE):
    '''
    AE-based reduced order model with uncertain random input
    modeled through GPR.

    Nomenclature:
    - X are the random input variables of the model
    - Y is the full-order output of the computational model
    - Z is the set of latent variables (obtained through AE)
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
            raise KeyError('You have to first train the ISOMAP!')
        
        self.gprs = []

        for d in range(self.rom_dim):

            gp = GaussianProcessRegressor(copy.deepcopy(self.kernel),
                                          nproc=self.nproc,
                                          reg_tych=self.tych)
            
            gp.fit(self.X_train, self.Z_train[:,d])

            self.gprs.append(gp)

    def predict_latent(self, X):
        '''
        Latent prediction from random input sample
        X is a (nsamples, uq_dim) numpy.array of inputs
        Z is a (nsamples, rom_dim) numpy.array of predictions
        '''
        nsamples, _ = X.shape

        Z_pred = np.zeros((nsamples, self.rom_dim))

        for d, gp in enumerate(self.gprs):

            Z_pred[:,d] = np.squeeze(gp.predict(X, return_cov=False))

        return np.squeeze(Z_pred)
