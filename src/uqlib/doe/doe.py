import os
import itertools

import numpy          as np
from scipy.stats    import qmc, norm, uniform
from scipy.special import roots_legendre, roots_hermitenorm, roots_genlaguerre
from scipy.spatial.distance import cdist

from .scaler import PCEDataScaler


# -------------------------------------------------------------------
#  DOE class
# -------------------------------------------------------------------

class PCEDoE():
    '''
    Class for performing DoE sampling
    '''
    methods = ['MC', 'LHS', 'QUADRATURE']
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
        if 'B' in pdf_var:
            raise NotImplementedError('Beta distribution not yet implemented')
        
    def sample(self, **kwargs):
        '''
        Method for generating samples.
        If sampling method MC or LHS use "ndata=".
        If quadrature method use "point_per_dim=".
        '''

        if self.method == 'MC':
            X, w = self._generate_mc(ndata=kwargs['ndata'])

        if self.method == 'LHS':
            X, w = self._generate_lhs(ndata=kwargs['ndata'])

        elif self.method == 'QUADRATURE':
            X, w = self._generate_quad(point_per_dim=kwargs['point_per_dim'])

        self.X = X
        self.weights = w

        return X, w

    def _generate_mc(self, ndata):

        X = np.empty((ndata,0))
        w = np.ones(ndata)

        for pdf in self.pdf_var:

            if pdf == 'U':
                x_u = np.random.uniform(-1, 1, size=(ndata,1))
                X = np.hstack((X, x_u))

            elif pdf == 'N':
                x_n = np.random.normal(0, 1, size=(ndata,1))
                X = np.hstack((X, x_n))

        return X, w
    
    def _generate_lhs(self, ndata):

        X = np.empty((ndata,0))
        w = np.ones(ndata)

        sampler = qmc.LatinHypercube(self.dim)
        X_01 = sampler.random(ndata)
        
        for i_pdf, pdf in enumerate(self.pdf_var):

            if pdf == 'U':
                x_u = uniform.ppf(X_01[:,i_pdf], loc=-1, scale=2)[:,np.newaxis]
                X = np.hstack((X, x_u))

            elif pdf == 'N':
                x_n = norm.ppf(X_01[:,i_pdf])[:,np.newaxis]
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
        w = np.prod(np.array(list(itertools.product(*weights_d))), axis=1)

        return X, w
                               
    def set_y(self, y):
        '''
        Setting the list of evaluation arrays of the DOE locations.
        '''
        if hasattr(self, 'y'):
            raise AttributeError('The DOE has already been evaluated, if you want to append other observations' \
            'use the the update() method!')
        else:
            self.y = y

    def save(self, names, bounds, filename, dimensional=False):
        '''
        Method to save the DoE in a ASCII format.
        Contains info about:
        - sampling method
        - inputs PDFs 
        - corresponding weights (if needed by the sampling method)
        - corresponding outputs (if already observed)

        Inputs:
        - names - list of variables names
        - bounds - (N,2) array of parameters of the 1D PDFs, e.g., low and upper bound, mean and variance
        - filename - string
        - dimensional - bool if the inputs are written dimensionally (default False)
        '''

        if dimensional:
           
            scaler = PCEDataScaler(self.pdf_var, 
                                   np.array(bounds[:,0]),
                                   np.array(bounds[:,1]))
            
            X = scaler.inverse_transform(self.X)

        else:

            X = self.X

        print_y = True if hasattr(self, 'y') else False

        with open(filename, 'w') as f:

            f.write('Method:\t' + self.method)
            f.write('\nDim:\t' + str(self.dim))
            f.write('\nSamples:\t' + str(self.X.shape[0]))

            f.write('\n\nUncertain inputs (pdf, name, bounds)')

            for i_var, pdf in enumerate(self.pdf_var):
                f.write(f'\n{pdf}\t{names[i_var]}\t{bounds[i_var,0]}\t{bounds[i_var,1]}')

            f.write('\n')

            line = '\nVariables are dimensional\t' if dimensional else '\nVariables are reduced\t'
            line += f'(inputs first {self.dim} columns'
            line += f', weights {self.dim+1} column' if self.method == 'QUADRATURE' else ''
            line += f', outputs last column' if print_y else ''
            line += ')'
            f.write(line)

            f.write('\n')
        
            for i in range(X.shape[0]):

                line = '\n'+'\t\t'.join(f'{x:.12f}' for x in X[i,:]) 
                line += f'\t\t{self.weights[i]:.12f}' if self.method == 'QUADRATURE' else ''
                line += f'\t\t{self.y[i]:.12f}' if print_y else ''

                f.write(line)

    def load(self, filename):
        '''
        Method to populate the DoE object from a previously saved file.
        '''

        with open(filename, 'r') as f:
            lines = f.readlines()

        for i, line in enumerate(lines):

            if 'Variables are ' in line:

                header_lines = lines[:(i+1)]

                dimensional = True if 'dimensional' in line else False

                append_w = True if 'weights' in line else False

                append_y = True if 'outputs' in line else False

                data_lines = lines[(i+1):]

                break

        # processing header
        self.method = header_lines[0].split(':')[1].lstrip('\t').rstrip('\n')
        self.dim    = int(header_lines[1].split(':')[1])
        self.ndata  = int(header_lines[2].split(':')[1])

        self.pdf_var = []
        bounds = np.zeros((self.dim, 2))

        for i in range(self.dim):
            pdf, name, par1, par2 = header_lines[5+i].strip('\n').split('\t')

            self.pdf_var.append(pdf)

            bounds[i,0] = float(par1)
            bounds[i,1] = float(par2)

        if dimensional: scaler = PCEDataScaler(self.pdf_var, bounds[:,0], bounds[:,1])

        # processing data
        X = np.empty((0, self.dim))
        w = []
        y = []

        for line in data_lines:

            if line.strip():

                vals = line.split('\t\t')

                if len(vals) >= self.dim:

                    x = np.array([[float(val) for val in vals[:self.dim]]])
                    
                    X = np.vstack((X, x))

                    if append_w: w.append(float(vals[self.dim])) 
                    
                    if append_y: y.append(float(vals[-1])) 

        self.X = scaler.transform(X) if dimensional else X

        self.weights = np.array(w) if append_w else np.ones(X.shape[0])

        if append_y: self.set_y(np.array(y))

# -------------------------------------------------------------------
#  Nested DOE class (Le Gratiet algorithm)
# -------------------------------------------------------------------

class NestedPCEDoE(PCEDoE):
    '''
    Class for performing the nested DoE structure
    '''
    methods = ['MC', 'LHS']
    pdf_types = ['U', 'N', 'G', 'B']

    def sample(self, ndata_per_level):
        '''
        Output
        - X is an array containing all the sampling locations
        - DL is a matrix o f zeros and ones telling which fidelity level should be sampled
          at a given location
        '''

        self.ndata_per_level = ndata_per_level
        self.nlevel = len(ndata_per_level)

        # obtaining the sampling function
        if self.method == 'MC':
            sampler = self._generate_mc

        if self.method == 'LHS':
            sampler = self._generate_lhs
        
        # building the HF DoE 
        X_hf = sampler(ndata=self.ndata_per_level[-1])[0]
        
        # initializing the DoE dictionary
        DL = np.ones((self.ndata_per_level[-1], self.nlevel))
        X = X_hf

        # cycling on all the levels, the level's DoE is generated and then the points closest to the higher DoE
        # are removed. Then, the DoE are put together
        for t in reversed(range(self.nlevel-1)):

            # sampling t DoE
            X_tf = sampler(ndata=self.ndata_per_level[t])[0]
            
            for i in range(X_hf.shape[0]):

                x_i = X_hf[i,:]
                dist = cdist(X_tf, x_i.reshape(1, self.dim), metric='euclidean')
                # removing from tf_doe the point closest to x_i, a point already in the doe
                X_tf = np.delete(X_tf, np.argmin(np.squeeze(dist)), 0)

            X_hf = np.append(X_hf, X_tf, axis=0)

            # updating the location matrix
            X = np.append(X, X_tf, axis=0)

            # updating the level matrix
            DL_t = np.array([1]*(t + 1) + [0]*(self.nlevel - 1 - t))
            DL_t = np.repeat(DL_t.reshape(1, self.nlevel), X_tf.shape[0], axis=0)
            DL = np.append(DL, DL_t, axis=0)

        self.X = X
        self.DL = DL

        return X, DL
    
    def level(self, level, return_y=True):
        '''
        Method to return the DOE at the desired level

        Input:
        - level, int indicating the required level

        Outputs:
        - x (ndata[level], dim) array
        - y (ndata[level],) array
        '''
        if not isinstance(level, int):
            raise ValueError('Integer required as input')
        
        if level > (self.nlevel - 1):
            raise ValueError('The required level is higher than the highest level of the DOE')
        
        i_datalevel = self.DL[:,level]
        X = self.X[i_datalevel == 1, :]
        if not return_y:
            return X
        else:
            y = self.y[level]
            return X, y
    
    def nestedlevel(self, level_high, level_low):
        '''
        Method to return the nested DOE at the desired levels i.e., the outuputs at the required levels at common design points x.

        Input:
        - level_high, the higher level
        - level_low, the lower level

        Outputs:
        - x_nested (ndata[level_high], dim) array
        - y_nested (ndata[level_high], 2) array with [:,0] indicating lower level outputs and [:,1] higher level outputs
        '''
        if not isinstance(level_high, int) or not isinstance(level_low, int):
            raise ValueError('Integers required as input')
        
        if level_high > (self.nlevel - 1):
            raise ValueError('The required level is higher than the highest level of the DOE')  
        
        if level_low >= level_high:
            raise ValueError('Lower level input is higher than the higher level input')
        
        X_nested, y_high = self.level(level_high)

        i_datalevel = self.DL[:,level_high]

        ndata, _ = X_nested.shape
        
        y_low = np.array([])
        k_low = 0
        for j in range(i_datalevel.size):
            if i_datalevel[j] == 1:
                y_low = np.append(y_low, self.y[level_low][k_low])
                k_low += 1
            elif i_datalevel[j] != self.DL[j,level_low]:
                k_low += 1

        y_nested = np.hstack(( y_low.reshape(ndata,1), 
                              y_high.reshape(ndata,1)))

        return X_nested, y_nested
    
    def save(self, names, bounds, filename, dimensional=False):
        '''
        Method to save the DoE in a ASCII format.
        Contains info about:
        - sampling method
        - inputs PDFs 
        - corresponding outputs (if already observed)
        - levels at which the user has to observe outputs if not done yet 

        Inputs:
        - names - list of variables names
        - bounds - (N,2) array of parameters of the 1D PDFs, e.g., low and upper bound, mean and variance
        - filename - string
        - dimensional - bool if the inputs are written dimensionally (default False)
        '''

        if dimensional:
           
            scaler = PCEDataScaler(self.pdf_var, 
                                   np.array(bounds[:,0]),
                                   np.array(bounds[:,1]))
            
            X = scaler.inverse_transform(self.X)

        else:

            X = self.X

        print_y = True if hasattr(self, 'y') else False

        with open(filename, 'w') as f:

            f.write('Method:\t' + self.method)
            f.write('\nDim:\t' + str(self.dim))
            f.write('\nLevels:\t' + str(self.nlevel))
            line = '\nSamples per level (low to high):\t'
            for n in self.ndata_per_level: line += str(n)+'\t'
            f.write(line)

            f.write('\n\nUncertain inputs (pdf, name, bounds)')

            for i_var, pdf in enumerate(self.pdf_var):
                f.write(f'\n{pdf}\t{names[i_var]}\t{bounds[i_var,0]}\t{bounds[i_var,1]}')

            f.write('\n')

            line = '\nVariables are dimensional\t' if dimensional else '\nVariables are reduced\t'
            line += f'(inputs first {self.dim} columns'
            if print_y:
                line += ', outputs lev. '+', '.join([str(l) for l in range(self.nlevel)]) +' last columns'
            else:
                line += ', datalevel last column'
            line += ')'
            f.write(line)

            f.write('\n')

            for i in range(X.shape[0]):

                line = '\n'+'\t\t'.join(f'{x:.12f}' for x in X[i,:]) 

                if print_y:
                    for l in range(self.nlevel):
                        try:
                            line += f'\t\t{self.y[l][i]:.12f}'
                        except:
                            continue
                else:
                    line += f'\t\t{np.sum(self.DL[i,:])}'

                f.write(line)
    
    def load(self, filename):
        '''
        Method to populate the DoE object from a previously saved file.
        '''

        with open(filename, 'r') as f:
            lines = f.readlines()

        for i, line in enumerate(lines):

            if 'Variables are ' in line:

                header_lines = lines[:(i+1)]

                dimensional = True if 'dimensional' in line else False

                append_y = True if 'outputs' in line else False

                data_lines = lines[(i+1):]

                break

        # processing header
        self.method = header_lines[0].split(':')[1].lstrip('\t').rstrip('\n')
        self.dim    = int(header_lines[1].split(':')[1])
        self.nlevel = int(header_lines[2].split(':')[1])

        line  = header_lines[3].split(':')[1]
        self.ndata_per_level = [int(n) for n in line.strip().split('\t')]

        self.pdf_var = []
        bounds = np.zeros((self.dim, 2))

        for i in range(self.dim):
            pdf, name, par1, par2 = header_lines[6+i].strip('\n').split('\t')

            self.pdf_var.append(pdf)

            bounds[i,0] = float(par1)
            bounds[i,1] = float(par2)

        if dimensional: scaler = PCEDataScaler(self.pdf_var, bounds[:,0], bounds[:,1])

        # processing data
        X  = np.empty((0, self.dim))
        DL = np.empty((0, self.dim))
        y  = [np.empty(0)] * self.nlevel

        for line in data_lines:

            if line.strip():

                vals = line.split('\t\t')

                if len(vals) >= self.dim:

                    x = np.array([[float(val) for val in vals[:self.dim]]])
                    
                    X = np.vstack((X, x))

                    dl = np.zeros(self.nlevel)

                    if append_y:

                        for l in range(self.nlevel): 
                            try:
                                y[l] = np.append(y[l], float(vals[self.dim+l]))
                                dl[l] = 1
                            except:
                                continue

                    else:

                        lev = int(vals[-1])

                        for l in range(lev): dl[l] = 1 

                    DL = np.vstack((DL, dl))

        self.X = scaler.transform(X) if dimensional else X

        self.DL = DL   

        if append_y: self.set_y(y)


