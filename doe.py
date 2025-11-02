import os
import time
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 
os.environ['OPENBLAS_NUM_THREADS'] = "1"

import pickle as pkl
import itertools

import numpy          as np
from scipy.stats    import qmc
from scipy.spatial.distance import cdist
from scipy.special import roots_legendre, roots_hermitenorm
from sklearn.preprocessing import FunctionTransformer

# -------------------------------------------------------------------
#  Nested DOE class (Le Gratiet algorithm)
# -------------------------------------------------------------------

class DoE():
    '''
    Class for performing the nested DoE structure for multifidelity kriging (Le Gratiet formulation)
    '''
    methods = ['MC', 'QUADRATURE']
    pdf_types = ['U', 'N', 'G', 'B']

    def __init__(self, dim, method, pdf_var):
        '''
        Input:
        - dim is the dimensionality of the problem
        - ndata_per_level is a list of length nlevels containg the number of data required at each level
          the position in the list indicates the level i.e., from 0 to len(list)
        '''
        self.dim = dim
        if method not in self.methods:
            raise KeyError('Wrong samling method. Availables: MC, LHS, QUADRATURE') 
        else:
            self.method = method

        self.pdf_var = pdf_var      # pdf of reduced variables (U, N, G, B)
        if any(t not in self.pdf_types for t in self.pdf_var):
            raise KeyError('Invalide PDF for reduced variables')
        if len(self.pdf_var) != self.dim:
            raise KeyError('Unspecified PDF kind for input size')
        if 'G' in pdf_var:
            raise NotImplementedError('Gamma distribution not yet implemented')
        if 'B' in pdf_var:
            raise NotImplementedError('Beta distribution not yet implemented')
        
    def __call__(self, **kwargs):

        if self.method == 'MC':
            X, w = self._generate_mc(ndata=kwargs['ndata'])

        elif self.method == 'QUADRATURE':
            X, w = self._generate_quad(point_per_dim=kwargs['point_per_dim'])

        self.xdata = X
        self.weights = w

        return X, w

    def _generate_mc(self, ndata):
        '''
        Output
        - xdata is an array containing all the sampling locations
        - datalevel is a matrix o f zeros and ones telling which fidelity level should be sampled
          at a given location
        '''
        X = np.empty((ndata,0))
        w = np.ones(ndata)

        for pdf in self.pdf_var:

            if pdf == 'U':
                x_u = np.random.uniform(-1, 1, size=(ndata,1))
                X = np.hstack((X, x_u))

            elif pdf == 'N':
                x_n = np.random.normal(0, 1, (ndata,1))
                X = np.hstack((X, x_n))

        return X, w
    
    def _generate_quad(self, point_per_dim):

        points_d = []
        weights_d = []
        for pdf in self.pdf_var:
            
            if pdf == 'U':
                roots, weights = roots_legendre(point_per_dim)
                points_d.append(roots)
                weights_d.append(weights/2)

            elif pdf == 'N':
                roots, weights = roots_hermitenorm(point_per_dim)
                points_d.append(roots)
                weights_d.append(weights/np.sqrt(2*np.pi))   

        X = np.array(list(itertools.product(*points_d)))
        w = np.prod(np.array(list(itertools.product(*points_d))), axis=1)

        return X, w
                               
    def set_y(self, y):
        '''
        Setting the list of evaluation arrays of the DOE locations.
        '''
        if hasattr(self, 'ydata'):
            raise AttributeError('The DOE has already been evaluated, if you want to append other observations' \
            'use the the update() method!')
        else:
            self.ydata = y
    

# -------------------------------------------------------------------
#  Input Scaler
# -------------------------------------------------------------------

class DataScaler(FunctionTransformer):

    def __init__(self, pdf_var, bndlw, bndup):

        super().__init__(        
            func=self.to_standard,
            inverse_func=self.from_standard,
            validate=False,
            accept_sparse=False,
            check_inverse=True,
            feature_names_out=None,
            kw_args={'bndlw':bndlw, 'bndup':bndup},
            inv_kw_args={'bndlw':bndlw, 'bndup':bndup},)
        
        self.pdf_var = pdf_var

    def from_standard(self, X, bndlw, bndup):

        ndata, dim = X.shape

        X_out = np.zeros(X.shape)

        for i_pdf, pdf in enumerate(self.pdf_var):

            if pdf == 'U':
                lw = np.repeat(bndlw[i_pdf], ndata)
                up = np.repeat(bndup[i_pdf], ndata)
                X_out[:, i_pdf] = (up + lw)/2 + X[:,i_pdf] * (up - lw)/2

            elif pdf == 'N':
                mean = np.repeat(bndlw[i_pdf], ndata)
                std  = np.repeat(bndup[i_pdf], ndata)
                X_out[:, i_pdf] = mean + std * X[:, i_pdf]

        return X_out


    def to_standard(self, X, bndlw, bndup):

        ndata, dim = X.shape

        X_out = np.zeros(X.shape)

        for i_pdf, pdf in enumerate(self.pdf_var):

            if pdf == 'U':
                lw = np.repeat(bndlw[i_pdf], ndata)
                up = np.repeat(bndup[i_pdf], ndata)
                X_out[:, i_pdf] = (X[:, i_pdf] - (up + lw)/2) + (up - lw)/2

            elif pdf == 'N':
                mean = np.repeat(bndlw[i_pdf], ndata)
                std  = np.repeat(bndup[i_pdf], ndata)
                X_out[:, i_pdf] = (X[:, i_pdf] - mean) / std

        return X_out

