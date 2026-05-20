import os, sys, copy
import numpy as np
from abc import abstractmethod

from scipy.linalg import eigh, cholesky, cho_solve
from scipy.spatial.distance import pdist, cdist, squareform
from scipy.stats import qmc, norm, uniform
from scipy.sparse.csgraph import shortest_path

from SALib.sample import sobol as sob_sample
from SALib.analyze import sobol as sob_analyze

from ..pce import PCE
from ..gp  import GaussianProcessRegressor

class ISOMAP():
    '''
    Nomenclature:
    - X are the random input variables of the model
        size: uq_dim
    - Y is the full-order output of the computational model
        size: fom_dim 
    - Z is the set of latent variables
        size: rom_dim
    '''

    def __init__(self):

        self.uq_dim = None
        self.pdf_var = None
        return
        
    def compute_isomap(self, Y, delta=0.9999, distance='geodesic'):
        '''
        Y is an (M x N) array with:
        - M number of snaphsots
        - N output size of the computational model
        '''
        Y = np.asarray(Y)
        if Y.ndim != 2:
            raise ValueError("Y must be 2D array of shape (M, N)")
        
        self.ndata, self.fom_dim = Y.shape

        # removing the mean
        self.Y_train = Y

        # looping to find the optimal k_near
        k_max = max(int(np.log10(self.ndata) * 10) , self.ndata-1)
        k_list = [k for k in range(1, k_max)]
        kruskal_list = []
        eigvals_list = []
        eigvecs_list = []

        for k_near in k_list:

            try:

                # computing the dissimilarity matrix
                if distance == 'geodesic':

                    D = self._compute_dissimilarity(self.Y_train, k_near)

                elif distance == 'euclidean':

                    D = squareform(pdist(self.Y_train, metric='euclidean'))

                D_tilde = -0.5 * D**2

                # Centering the square of D i.e., D_tilde
                H = np.eye(self.ndata) - 1/self.ndata * np.ones((self.ndata, self.ndata))
                B = H @ D_tilde @ H

                # eigendecomposing B
                eigvals_k, eigvecs_k = eigh(B, subset_by_value=[0.0, np.inf], driver='evr')

                Z_tilde = eigvecs_k @ np.diag(np.sqrt(eigvals_k))

                # computing the Kruskal's stress (for optimum k)
                kruskal = self._kruskal_stress(D, Z_tilde)

                kruskal_list.append(kruskal)
                eigvals_list.append(eigvals_k)
                eigvecs_list.append(eigvecs_k)

            except ValueError as e:

                kruskal_list.append(np.nan)
                eigvals_list.append([])
                eigvecs_list.append([])

        # find the optimum k from the minimum kruskal
        ind_opt = np.nanargmin(np.array(kruskal_list))

        self.k_near = k_list[ind_opt]

        idx_sort = np.argsort(eigvals_list[ind_opt])[::-1]
        eigvals_opt = eigvals_list[ind_opt][idx_sort]
        eigvecs_opt = eigvecs_list[ind_opt][:, idx_sort]

        # relative information content to cut additional modes if unnecessary
        for d in range(1, eigvals_opt.shape[0]+1):
            ric = np.sum(eigvals_opt[:d]**2) / np.sum(eigvals_opt**2)
            if ric > delta:
                self.rom_dim = d     # latent space dimensionality
                break

        eigvals_iso  = eigvals_opt[:self.rom_dim]                        # (rom_dim,) array
        eigvecs_iso  = np.atleast_2d(eigvecs_opt[:,:self.rom_dim])       # (ndata, rom_dim) array
        self.Z_train = (eigvecs_iso @ np.diag(np.sqrt(eigvals_iso)))     # (ndata, rom_dim)  array

    def moments(self, nsamples=500):

        # sampling the random inputs X from the standard distributions
        X_samples = self._sample_x(nsamples=nsamples)

        # Isomap prediction for the samples set
        Y_pred = self.predict(X_samples) 

        mean = np.mean(Y_pred, axis=0)
        var  = np.var( Y_pred, axis=0)

        return mean, var
    
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

        for smp in range(nsamples):

            # finding the optimal weight of the training latent variables closest to the prediction
            w_opt, neigh_indices = self._backmapping(Z_pred[smp,:])

            # prediction as weighted linear linear combination of the near snaps 
            Y_pred[smp,:] += self.Y_train[neigh_indices, :].T @ w_opt 

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

    def _compute_dissimilarity(self, Y, k_near):
        """
        Given an array Y of shape (M, N) where each row is a sample y (length N):
        1) compute Euclidean distances to neighbors,
        2) find k nearest neighbors for each sample,
        3) build sparse weighted graph (vertices=samples, edges=euclidean distance to neighbors),
        4) compute all-pairs shortest path distances (sum of weights along shortest path).
        
        Returns:
        dist_matrix : ndarray (M, M) of shortest-path distances (np.inf if unreachable)
        """

        if not (0 < k_near < self.ndata):
            raise ValueError("k must satisfy 0 < k < M (number of samples)")
        
        # Pairwise Euclidean distance
        D_euclid = cdist(Y, Y, metric='euclidean')
        
        # Nearest neighbors using Euclidean distances
        G = np.full((self.ndata, self.ndata), np.inf)

        for i in range(self.ndata):

            # Exclude the point itself and get k nearest neighbors
            neighbors = np.argsort(D_euclid[i])[1:k_near+1]

            # Assign edge weights
            G[i, neighbors] = D_euclid[i, neighbors]
        
        # Make graph symmetric (undirected)
        G = np.minimum(G, G.T)

        # Zero diagonal
        np.fill_diagonal(G, 0.0)
        
        D_geo = shortest_path(
            csgraph=G,
            directed=False,
            method='D'
            )
        
        return D_geo
        
    def _kruskal_stress(self, D, Z_tilde):
        '''
        Metric to evaluate the optimum number k of nearest neighbors
        '''
        num = np.sum( ( D - squareform( pdist(Z_tilde, metric='euclidean') ) )**2 )
        den = np.sum(D**2)
        return np.sqrt(num/den)
       
    def _backmapping(self, Z_pred):
        '''
        Backmapping procedure from sampled latent variable z_pred to find the 
        optimal weights and the indexes of the k-nearest neighbors among the snapshots.
        The backmapping weights are an exact solution of a convex quadratic optimization problem subject
        to linear constraints.
        The solution exploits the Schur complement, see Franz et al..
        '''

        D_euclid = np.squeeze(cdist(np.atleast_2d(Z_pred), self.Z_train, metric='euclidean'))

        neigh_indices = np.argsort(D_euclid)[:self.k_near]

        Z_neighs = self.Z_train[neigh_indices, :]

        # gram matrix G
        tmp = np.repeat(np.atleast_2d(Z_pred), repeats=len(neigh_indices), axis=0)
        G = (tmp - Z_neighs) @ (tmp - Z_neighs).T 

        # regularization coeffs
        cvec = np.linalg.norm(tmp - Z_neighs, 2.0, axis=1)
        c = 0.01 * (cvec / np.amax(cvec))**4.0

        # forcing the regularization term
        G_tilde = G + np.diag(c)
        A = 2*G_tilde
        
        # solution of the quadratic constrained optimization problem 
        # w.T @ A @ w s.t. \sum w_i = 1
        # W_opt = (Ainv @ 1) / (1.T @ Ainv @ 1)
        while True:
            try:
                L = cholesky(A, lower=True)
                break
            except:
                l, V = eigh(A)
                A = V @ np.diag(np.clip(l, a_min=1e-8, a_max=None)) @ V.T

        one_vec = np.ones(len(neigh_indices))
        Ainv = cho_solve((L,True), np.eye(len(neigh_indices)))
        AinvOnes = Ainv @ one_vec
        S = one_vec.T @ AinvOnes
        w_opt = 1/S * AinvOnes

        return w_opt, neigh_indices



class ISOMAPPCE(ISOMAP):
    '''
    ISOMAP-based reduced order model with uncertain random input
    modeled through PCE.

    Nomenclature:
    - X are the random input variables of the model
    - Y is the full-order output of the computational model
    - Z is the set of latent variables (obtained through ISOMAP)
    ''' 

    def __init__(self, uq_dim, pce_degree, pdf_var, truncation):

        self.uq_dim  = uq_dim
        self.pdf_var = pdf_var

        # building a PCE object
        self.pce = PCE(uq_dim, pce_degree, pdf_var, truncation)

    def compute_pce(self, X, method, weights=None):

        self.X_train = X

        if not hasattr(self, 'Z_train'):
            raise KeyError('You have to first train the ISOMAP!')
    
        # computing the PCE coefficients on the latent space basis
        self.pce.compute_coeffs(X, self.Z_train, method=method, weights=weights)

    def predict_latent(self, X):
        '''
        Latent prediction from random input sample
        X is a (nsamples, uq_dim) numpy.array of inputs
        Z is a (nsamples, rom_dim) numpy.array of predictions
        '''
        return self.pce.predict(X)



class ISOMAPGPR(ISOMAP):
    '''
    ISOMAP-based reduced order model with uncertain random input
    modeled through GPR.

    Nomenclature:
    - X are the random input variables of the model
    - Y is the full-order output of the computational model
    - Z is the set of latent variables (obtained through ISOMAP)
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

        return Z_pred

    



