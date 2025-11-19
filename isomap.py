import os, sys, shutil
import numpy as np
from scipy.linalg import svd, eigh
from scipy.spatial.distance import pdist, cdist
from scipy.optimize import minimize, LinearConstraint, Bounds
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from sklearn.neighbors import NearestNeighbors
import copy

from . import PCE

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
        self.pce_parameters = {'dim':        dim,
                               'degree':     degree,
                               'pdf_var':    pdf_var,
                               'truncation': truncation}
        
    def compute_isomap(self, Y, delta=0.9999):
        '''
        Z is an (M x N) array with:
        - M number of snaphsots
        - N output size of the computational model
        '''
        Y = np.asarray(Y)
        if Y.ndim != 2:
            raise ValueError("Y must be 2D array of shape (M, N)")
        else:
            self.snapshots = Y

        self.snapshots = Y
        self.ndata, self.noutputs = Y.shape

        #self.k_near = k_near

        # looping to finde the optimal k_near
        k_list = [k for k in range(10, 60, 5)]
        kruskal_list = []
        eigvals_list = []
        eigvecs_list = []

        for i_k, k_near in enumerate(k_list):

            # computing the dissimilarity matrix
            D = self._compute_dissimilarity(Y, k_near)

            # computing D_tilde
            D_tilde = 0.5 * D**2

            # removing row-wise and column-wise mean from D_tilde
            H = np.eye(self.ndata) - 1/self.ndata * np.ones((self.ndata, self.ndata))
            B = H @ D_tilde @ H

            # eigendecomposing B
            eigvals, eigvecs = eigh(B, subset_by_value=[0.0, np.inf], driver='evr')
            Z_tilde = eigvecs @ np.diag(np.sqrt(eigvals))

            # computing the Kruskal's stress (for optimum k)
            kruskal = self._kruskal_stress(D, Z_tilde)

            kruskal_list.append(kruskal)
            eigvals_list.append(eigvals)
            eigvecs_list.append(eigvecs)

        # find the optimum k from the minimum kruskal
        ind_opt = np.argmin(np.array(kruskal_list))

        self.k_near = k_list[ind_opt]
    
        eigvals_opt = eigvals_list[ind_opt]
        eigvecs_opt = eigvecs_list[ind_opt]

        # relative information content to cut additional modes if unnecessary
        d = 1
        for d in range(1, eigvals_opt.shape[0]+1):
            ric = np.sum(eigvals_opt[:d]**2) / np.sum(eigvals_opt**2)
            if ric > delta:
                break

        self.isomap_degree = d
        self.eigvals = eigvals_opt[:self.isomap_degree]                        # (d,) array
        self.eigvecs = eigvecs_opt[:,:self.isomap_degree]                      # (M,d) array
        self.latent  = (self.eigvecs @ np.diag(np.sqrt(self.eigvals))).T       # (d, M) array
    
    def compute_pce(self, X, method, weights=None):

        self.X_train = X

        if not hasattr(self, 'latent'):
            raise KeyError('You to first do the ISOMAP!')
        
        # generating a list of pce
        self.pces = [copy.deepcopy(PCE(**self.pce_parameters)) for i in range(self.isomap_degree)]
        
        # training a PCE for each component of the latent variable (each one having ndata samples)
        for i_pce, pce in enumerate(self.pces):
            z_train = self.latent[i_pce,:]
            pce.compute_coeffs(X, z_train, method=method, weights=weights)

    def moments(self, mc_samples=500):

        pred_samples = np.zeros((mc_samples, self.noutputs))

        # sampling the random inputs X from the standard distributions
        X_samples = self._sample_x(n_samples=mc_samples)

        for smp in range(mc_samples):

            pred_samples[smp,:] = self.predict(X_samples[smp,:]) 

        mean = np.mean(pred_samples, axis=0)
        var  = np.var( pred_samples, axis=0)

        return mean, var

    def predict(self, X):
        '''
        ROM prediction from random input sample
        '''

        if X.ndim == 1:
            X = X[np.newaxis,:]
        elif X.ndim > 1:
            if X.shape[1] > 1:
                raise ValueError('You can predict at the moment one X sample at time!')

        # predicting the latent variable through PCE 
        z_pred = np.zeros(self.isomap_degree)
        for i_pce, pce in enumerate(self.pces):
            z_pred[i_pce] = pce.predict(X)

        # finding the optimal weight of the training latent variables closest to the prediction
        w_opt, neigh_indices = self._backmapping(z_pred)

        # prediction as weighted linear linear combination of the near snaps 
        y_pred = self.snapshots[neigh_indices, :].T @ w_opt

        return y_pred
    
    def _sample_x(self, n_samples):
        '''
        Generating samples of inputs from standard distributions
        '''
        X = np.zeros((n_samples, self.dim))

        for i_var, var in enumerate(self.pdf_var):

            if var == 'U':
                X[:,i_var] = np.random.uniform(-1, 1, n_samples)

            elif var == 'N':
                X[:,i_var] = np.random.normal(0, 1, n_samples)

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
        num = np.sum((D - pdist(Z_tilde, metric='euclidean')**2))
        den = np.sum(D**2)
        return np.sqrt(num/den)
    
    def _backmapping(self, z_pred):
        '''
        Backmapping procedure from sampled latent variable z_pred to find the 
        optimal weights and the indexes of the k-nearest neighbors among the snapshots
        '''

        if z_pred.ndim < 2:
            z_pred = z_pred[np.newaxis,:]

        nn = NearestNeighbors(n_neighbors=self.k_near, 
                              metric='euclidean', 
                              algorithm='auto')
        nn.fit(self.latent.T)
        neigh_indices = np.squeeze(nn.kneighbors(z_pred, return_distance=False))

        neighs = self.latent[:, neigh_indices]

        def backmap_obj(w, eps=0.01, gamma=4.0):

            obj = np.linalg.norm(z_pred - neighs.dot(w),2.0)**2

            c = np.zeros(self.k_near)
            for i in range(c.shape):
                c[i] = eps * ( np.linalg.norm(z_pred - neighs[:,i])**2 / 
                               np.amax( np.linalg.norm( np.repeat(z_pred[:,np.newaxis],self.k_near,axis=1) - neighs, axis=0)**2 ) )**gamma
                
            obj += c.dot(w**2)
            
            return obj
        
        def backmap_con(w):
            return np.eye(w.size).dot(w) - 1

        weight_bnd = Bounds(lb=[-2]*self.k_near, ub=[2]*self.k_near)

        w_list = []
        f_list = []
        for start in range(10):
            w0 = np.random.uniform(-2, 2, self.k_near)
            opt_results = minimize(backmap_obj, 
                                   w0,
                                   method='SLSQP',
                                   bounds=weight_bnd,
                                   constraints=[{'type': 'eq', 'fun': backmap_con}],
                                   tol=1e-7,
                                   options={'maxiter':1000}) 
            w_list.append(opt_results.x)
            f_list.append(opt_results.fun)

        w_opt = np.array(w_list)[np.argmin(np.array(f_list)),:]

        return w_opt, neigh_indices




    



