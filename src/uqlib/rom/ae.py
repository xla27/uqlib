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

from ..pce import PCE
from ..gp  import GaussianProcessRegressor
from .common import sobol_wrapper

class AE():

    def __init__(self, uq_dim, rom_dim):

        self.uq_dim = uq_dim
        self.rom_dim = rom_dim
        self.model  = None
        self.model_trained = None
        self.device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

    def train_ae(self, model_nn, Y, nproc, options):
        """
        AE training wrapper:
        - Y tensor of shape (ndata, nfields, n_x, n_y)
        - model_nn is the AE model (constructed outside the library)
        """

        if not hasattr(model_nn, 'encoder'):
            raise Exception('NN model missing encoder attribute!')
        
        if not hasattr(model_nn, 'decoder'):
            raise Exception('NN model missing decoder attribute!')

        self.model = model_nn

        self._preprocess_y(Y)

        if not options: options = {}

        if not 'epochs'     in options.keys(): options['epochs']     = 50
        if not 'batch_size' in options.keys(): options['batch_size'] = 4
        if not 'lr'         in options.keys(): options['lr']         = 1e-3
        if not 'filename'   in options.keys(): options['filename']   = "cfd_autoencoder.pt"

        # Use simple single-machine multi-GPU training without distributed setup
        # Single GPU/CPU training
        self._train(self.Y_train,
                    nproc,
                    options['epochs'],
                    options['batch_size'],
                    options['lr'],
                    options['filename'])

        self.load_ae(model_nn, Y, options['filename'])

    def load_ae(self, model_nn, Y, filename):

        model = model_nn

        checkpoint = torch.load(filename)

        model.load_state_dict(checkpoint['model_state_dict'])

        model.eval()

        self.model_trained = model

        self._preprocess_y(Y)

        with torch.no_grad():
            Z_train = self.model_trained.encoder(self.Y_train.to(self.device)).cpu()
            Y_recon = self.model_trained.decoder(Z_train.to(self.device))
            Y_recon = Y_recon[:, :, :self.nx, :self.ny]

        recon_err = torch.norm(self.Y_train - Y_recon) / torch.norm(self.Y_train)

        print(f'\tReconstruction error: {recon_err:.2e}')

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

        # fixing Z_pred size
        if nsamples == 1 and self.rom_dim == 1:
            Z_pred = np.atleast_2d(Z_pred)
        elif nsamples > 1 and self.rom_dim == 1:
            Z_pred = Z_pred[:,np.newaxis]
        elif nsamples == 1 and self.rom_dim > 1:
            Z_pred = np.atleast_2d(Z_pred)

        with torch.no_grad():
            Y_pred = self.model_trained.decoder(
                torch.tensor(Z_pred, dtype=torch.float32).to(self.device)
            ).numpy()
            Y_pred = Y_pred[:, :, :self.nx, :self.ny]

        for f in range(self.nfields):
            Y_pred[:,f,:,:] = self.Y_min[f] + Y_pred[:,f,:,:] * (self.Y_max[f] - self.Y_min[f])

        return np.squeeze(Y_pred)

    def moments(self, nsamples=1000):
        
        X_samples = self._sample_x(nsamples=nsamples)

        Y_pred = self.predict(X_samples) 

        mean = np.mean(Y_pred, axis=0)
        var  = np.var( Y_pred, axis=0)

        return mean, var   

    def sobol(self, calc_second=False, return_total=False, n_mc=1024, nproc=1):
        '''
        Sobol' indices estimation through Saltelli's sampling
        '''
        return sobol_wrapper(self, 
                             calc_second=calc_second, 
                             return_total=return_total,
                             n_mc=n_mc,
                             nproc=nproc)

    def _preprocess_y(self, Y):

        self.ndata, self.nfields, self.nx, self.ny = Y.shape

        self.Y_min = np.zeros(self.nfields)
        self.Y_max = np.zeros(self.nfields)
        Y_train = np.zeros_like(Y)

        for f in range(self.nfields):
            self.Y_min[f] = np.amin(Y[:,f,:,:])
            self.Y_max[f] = np.amax(Y[:,f,:,:])
            Y_train[:,f,:,:] = (Y[:,f,:,:] - self.Y_min[f]) / (self.Y_max[f] - self.Y_min[f])

        self.Y_train = torch.tensor(Y_train, dtype=torch.float32)

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

    def _train(self, data_tensor, num_workers=1, epochs=50, batch_size=4, lr=1e-3, filename="cfd_autoencoder.pt"):
        """
        Single GPU/CPU training without distributed setup.
        """
        device = self.device
        model = self.model.to(device)

        # Use DataParallel for multi-GPU if available
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        # ---- dataset ----
        dataset = TensorDataset(data_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        # ---- optimizer with L2 regularization ----
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

        # ---- training loop ----
        for epoch in range(epochs):
            total_loss = 0.0

            for (x,) in loader:
                x = x.to(device)

                recon = model(x)
                loss = loss_fn(recon, x)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.6f}")

            # Save checkpoint
            if epoch % 50 == 0:
                checkpoint_model = model.module if isinstance(model, nn.DataParallel) else model
                torch.save({
                    "model_state_dict": checkpoint_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }, filename)




def loss_fn(recon, x):
    mse = nn.functional.mse_loss(recon, x)
    return mse



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
