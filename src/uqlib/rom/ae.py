import os, sys
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
        
        world_size = torch.cuda.device_count() if torch.cuda.is_available() else int(nproc)

        if not options: options = {}

        if not hasattr(options, 'epochs'):     options['epochs'] = 50
        if not hasattr(options, 'batch_size'): options['batch_size'] = 4
        if not hasattr(options, 'lr'):         options['lr'] = 1e-3

        mp.spawn(
            self._train,
            args=(world_size, 
                  self.Y_train, 
                  *options),
            nprocs=world_size,
            join=True
        )

        self.load_ae(model_nn, Y, "cfd_autoencoder.pt")

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

    def _train(self, rank, world_size, data_tensor, epochs=50, batch_size=4, lr=1e-3):

        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "23456"

        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            rank=rank,
            world_size=world_size
        )

        device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

        # ---- dataset + sampler ----
        dataset = TensorDataset(data_tensor)
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
        loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

        # ---- model ----
        model = self.model.to(device)
        model = DDP(model, device_ids=[rank] if torch.cuda.is_available() else None)

        # ---- optimizer with L2 regularization ----
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

        # ---- training loop ----
        for epoch in range(epochs):
            sampler.set_epoch(epoch)

            total_loss = 0.0

            for (x,) in loader:
                
                x = x.to(device)

                recon = model(x)
                loss = loss_fn(recon, x)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if rank == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.6f}")

        if rank == 0:
            torch.save({
                "model_state_dict": model.module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, "cfd_autoencoder.pt")

        dist.destroy_process_group()



def loss_fn(recon, x):
    mse = nn.functional.mse_loss(recon, x)
    return mse




class AEPCE(AE):

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
    


