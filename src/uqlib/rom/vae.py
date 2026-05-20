import os, sys
import numpy as np
from scipy.stats import qmc, norm, uniform

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from ..pce import PCE

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=3, inner_dim=64):
        super().__init__()

        # Encoder
        self.fc1       = nn.Linear(input_dim, inner_dim)
        self.fc_mu     = nn.Linear(inner_dim, latent_dim)
        self.fc_logvar = nn.Linear(inner_dim, latent_dim)

        # Decoder
        self.fc2    = nn.Linear(latent_dim, inner_dim)
        self.fc_out = nn.Linear(inner_dim, input_dim)

    def encode(self, x):
        h      = F.relu(self.fc1(x))
        mu     = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.fc2(z))
        return self.fc_out(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z          = self.reparameterize(mu, logvar)
        x_hat      = self.decode(z)
        return x_hat, mu, logvar
    


class VAEPCE():

    def __init__(self, uq_dim, rom_dim, pce_degree, pdf_var, truncation):

        self.uq_dim = uq_dim
        self.pdf_var = pdf_var
        self.rom_dim = rom_dim

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.pce = PCE(uq_dim, pce_degree, pdf_var, truncation)

    def train_vae(self, Y, beta=1e-2, inner_layers=32):

        self.beta = beta

        _, self.fom_dim = Y.shape

        self.Y_train_mean = Y.mean(axis=0)
        self.Y_train_std  = Y.std(axis=0)

        self.Y_train = torch.tensor((Y - self.Y_train_mean) / (self.Y_train_std + 1e-8),
                                    dtype=torch.float32)

        dataset = TensorDataset(self.Y_train)
        self.loader = DataLoader(dataset, batch_size=16, shuffle=True)

        models_list    = []
        Z_train_list   = []
        recon_err_list = []

        for rom_dim in range(2, 6):
            model, Z_train, recon_err = self._train(rom_dim, inner_layers)

            print(f'\tRom dim {rom_dim} - err = {recon_err}')

            models_list.append(model)
            Z_train_list.append(Z_train)
            recon_err_list.append(recon_err)

        self.model   = models_list[np.argmin(np.array(recon_err_list))]
        self.Z_train = Z_train_list[np.argmin(np.array(recon_err_list))]     

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
    
    def predict(self, X):
        '''
        FOM prediction from random input sample
        X is a (nsamples, uq_dim) numpy.array of inputs
        Y is a (nsamples, fom_dim) numpy.array of predictions
        '''

        if X.ndim == 1:
            X = X[np.newaxis,:]

        nsamples, _ = X.shape

        Y_pred = np.zeros((nsamples, self.fom_dim))

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
            Y_pred = self.model.decode(
                torch.tensor(Z_pred, dtype=torch.float32).to(self.device)
            )

        Y_pred = self.Y_train_mean + self.Y_train_std * Y_pred.numpy()

        return np.squeeze(Y_pred)
    
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
    
    def _train(self, rom_dim, inner_layers):

        model = VAE(input_dim=self.fom_dim, latent_dim=rom_dim, inner_dim=inner_layers).to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        epochs = 2000

        for epoch in range(epochs):
            total_loss = 0

            for (y,) in self.loader:
                y = y.to(self.device)

                x_hat, mu, logvar = model(y)
                loss = loss_function(x_hat, y, mu, logvar, self.beta)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            # if epoch % 20 == 0:
            #     print(f"\tEpoch {epoch}, Loss: {total_loss:.2f}")

        model.eval()

        with torch.no_grad():
            mu, logvar = model.encode(self.Y_train.to(self.device))
            Z_train = mu.cpu()   # use mean as deterministic latent representation
            # VEDERE SE AGGIUNGERE LOGVAR COME LATENT VARIABLE
            Y_recon = model.decode(Z_train.to(self.device))

        recon_err = torch.norm(self.Y_train - Y_recon) / torch.norm(self.Y_train)

        return model, Z_train, recon_err       



def loss_function(x_hat, x, mu, logvar, beta):
    # reconstruction loss (MSE)
    recon = F.mse_loss(x_hat, x, reduction='sum')

    # KL divergence (closed form)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon + beta*kl