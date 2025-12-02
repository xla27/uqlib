import os, sys, shutil
import numpy as np
from scipy.linalg import svd, eigh, eig, cholesky, cho_solve, eigvalsh
from scipy.spatial.distance import pdist, cdist, squareform
from scipy.stats import qmc, norm, uniform
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from sklearn.neighbors import NearestNeighbors
import copy

from . import PCE

np.set_printoptions(threshold=sys.maxsize)

class ISOMAPPCE():
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
        
    def compute_isomap(self, Y, delta=0.9999, distance='geodesic'):
        '''
        Z is an (M x N) array with:
        - M number of snaphsots
        - N output size of the computational model
        '''
        Y = np.asarray(Y)
        if Y.ndim != 2:
            raise ValueError("Y must be 2D array of shape (M, N)")
        
        self.ndata, self.noutputs = Y.shape

        # removing the mean
        snapshots = Y

        # looping to finde the optimal k_near
        k_max = max(int(np.log10(self.ndata) * 10) , self.ndata-1)
        k_list = [k for k in range(1, k_max)]
        kruskal_list = []
        eigvals_list = []
        eigvecs_list = []

        for i_k, k_near in enumerate(k_list):

            try:

                # computing the dissimilarity matrix
                if distance == 'geodesic':
                    D = self._compute_dissimilarity(snapshots, k_near)
                elif distance == 'euclidean':
                    D = squareform(pdist(snapshots, metric='euclidean'))

                # Centering the square of D i.e., D_tilde
                H = np.eye(self.ndata) - 1/self.ndata * np.ones((self.ndata, self.ndata))
                B = - H @ D**2 @ H

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
        eigvals_opt = eigvals_list[ind_opt]
        eigvecs_opt = eigvecs_list[ind_opt]

        # relative information content to cut additional modes if unnecessary
        d = 1
        for d in range(1, eigvals_opt.shape[0]+1):
            ric = np.sum(eigvals_opt[:d]**2) / np.sum(eigvals_opt**2)
            if ric > delta:
                self.isomap_degree = d     # latent space dimensionality
                break
        self.isomap_degree = eigvals_opt.shape[0]
        self.snapshots = snapshots

        self.eigvals = eigvals_opt[:self.isomap_degree]                        # (d,) array
        self.eigvecs = eigvecs_opt[:,:self.isomap_degree]                      # (M,d) array
        self.latent  = (self.eigvecs @ np.diag(np.sqrt(self.eigvals))).T       # (d, M) array
    
    def compute_pce(self, X, method, weights=None):

        self.X_train = X

        if not hasattr(self, 'latent'):
            raise KeyError('You to first do the ISOMAP!')
    
        # computing the PCE coefficients on the latent space basis
        self.pce.compute_coeffs(X, self.latent.T, method=method, weights=weights)

    def moments(self, mc_samples=500):

        # sampling the random inputs X from the standard distributions
        X_samples = self._sample_x(n_samples=mc_samples)

        # Isomap prediction for the samples set
        pred_samples = self.predict(X_samples) 

        mean = np.mean(pred_samples, axis=0)
        var  = np.var( pred_samples, axis=0)

        return mean, var

    def predict(self, X):
        '''
        ROM prediction from random input sample
        X is a (nsamples, dim) numpy.array of inputs
        y is a (nsamples, noutputs) numpy.array of predictions
        '''

        if X.ndim == 1:
            X = X[np.newaxis,:]

        nsamples, _ = X.shape

        Y_pred = np.zeros((nsamples, self.noutputs))

        # predicting the latent variable through PCE 
        Z_pred = np.atleast_2d( self.pce.predict(X) )

        for smp in range(nsamples):

            # finding the optimal weight of the training latent variables closest to the prediction
            w_opt, neigh_indices = self._backmapping(Z_pred[smp,:])

            # prediction as weighted linear linear combination of the near snaps 
            Y_pred[smp,:] += self.snapshots[neigh_indices, :].T @ w_opt 

        return Y_pred
    
    def _sample_x(self, n_samples):
        '''
        Generating samples of inputs from standard distributions
        '''
        X = np.zeros((n_samples, self.dim))

        for i_var, var in enumerate(self.pdf_var):

            sampler = qmc.LatinHypercube(d = 1)
            samples = np.squeeze(sampler.random(n_samples))

            if var == 'U':
                #X[:,i_var] = qmc.scale(samples, np.array([-1]), np.array([1]))
                X[:,i_var] = uniform.ppf(samples, loc=-1, scale=2)
            elif var == 'N':
                X[:,i_var] = norm.ppf(samples)

        return X

    def _compute_dissimilarity(self, Y, k_near, symmetric=True):
        """
        Given an array Y of shape (M, N) where each row is a sample y (length N):
        1) compute Euclidean distances to neighbors,
        2) find k nearest neighbors for each sample,
        3) build sparse weighted graph (vertices=samples, edges=euclidean distance to neighbors),
        4) compute all-pairs shortest path distances (sum of weights along shortest path).
        
        Returns:
        dist_matrix : ndarray (M, M) of shortest-path distances (np.inf if unreachable)
        adj         : scipy.sparse.csr_matrix adjacency matrix used (weights = euclidean distances)
        """

        if not (0 < k_near < self.ndata):
            raise ValueError("k must satisfy 0 < k < M (number of samples)")
        
        # 1 & 2: Nearest neighbors using Euclidean distances
        nn = NearestNeighbors(n_neighbors=k_near, 
                              metric='euclidean', 
                              algorithm='auto')
        nn.fit(Y)
        distances, indices = nn.kneighbors(Y)   # shapes: (M, k)
        
        # 3: Build sparse adjacency (directed from i -> neighbor_j with weight = distance)
        rows = np.repeat(np.arange(self.ndata), k_near)
        cols = indices.ravel()
        data = distances.ravel()
        adj = csr_matrix((data, (rows, cols)), shape=(self.ndata, self.ndata))
        
        # If user wants undirected graph, make symmetric by taking the minimum weight where both exist.
        if symmetric:
            # adj.minimum(adj.T) yields elementwise minimum (keeps zeros if no edge both ways)
            adj = adj.minimum(adj.T)
            # If adjacency has zeros where no edge existed, they remain 0; set explicit zeros on diagonal
            adj.setdiag(0)
        
        # 4: All-pairs shortest paths (Dijkstra). Use undirected flag if symmetric, else directed.
        directed_flag = not symmetric
        dist_matrix = shortest_path(csgraph=adj, 
                                    method='D', 
                                    unweighted=False, 
                                    directed=directed_flag, 
                                    return_predecessors=False)
        # dist_matrix[i,j] is shortest-path distance from i to j (sum of Euclidean distances on path),
        # np.inf indicates no path between nodes.
        
        return dist_matrix
        
    def _kruskal_stress(self, D, Z_tilde):
        '''
        Metric to evaluate the optimum number k of nearest neighbors
        '''
        num = np.sum( ( D - squareform( pdist(Z_tilde, metric='euclidean') ) )**2 )
        den = np.sum(D**2)
        return np.sqrt(num/den)
       
    def _backmapping(self, z_pred):
        '''
        Backmapping procedure from sampled latent variable z_pred to find the 
        optimal weights and the indexes of the k-nearest neighbors among the snapshots.
        The backmapping weights are an exact solution of a convex quadratic optimization problem subject
        to linear constraints.
        The solution exploits the Schur complement, see Franz et al..
        '''

        # finding k-nearest neighbors of z_pred
        nn = NearestNeighbors(n_neighbors=self.k_near, 
                              metric='euclidean', 
                              algorithm='auto')
        nn.fit(self.latent.T)
        neigh_indices = np.squeeze( nn.kneighbors( np.atleast_2d(z_pred), return_distance=False) )

        neighs = self.latent[:, neigh_indices]

        # gram matrix G
        tmp = np.repeat(np.atleast_2d(z_pred).T, repeats=len(neigh_indices), axis=1)
        G = (tmp - neighs).T @ (tmp - neighs) 

        # regularization coeffs
        c = np.zeros(self.k_near)
        for i in range(self.k_near):
            c[i] = np.linalg.norm(z_pred - neighs[:,i], 2.0)
        c = 0.01 * (c / np.amax(c))**4.0

        # forcing the regularization term
        G_tilde = G + np.diag(c)
        A = 2*G_tilde
        
        # solution of the constrained optimization problem
        while True:
            try:
                L = cholesky(A, lower=True)
                break
            except:
                l, V = eigh(A)
                A = V @ np.diag(np.clip(l, a_min=1e-8, a_max=None)) @ V.T
                i+=1

        inv = cho_solve((L,True), np.eye(len(neigh_indices)))
        S = - np.ones(len(neigh_indices)).T @ inv @ np.ones(len(neigh_indices))
        w_opt = - 1/S * inv @ np.ones(len(neigh_indices))  

        return w_opt, neigh_indices




    



