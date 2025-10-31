import os, sys, shutil
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 
os.environ['OPENBLAS_NUM_THREADS'] = "1"

import numpy as np
import math
from scipy.stats import qmc
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import r2_score

from itertools import product, repeat
import time

from pce import PCE

# Ishigami function
dim = 3
def Ishigami(x):
    x1, x2, x3 = x
    fun = np.sin(x1) + 7 * np.sin(x2)**2 + 0.1 * x3**4 * np.sin(x1)
    return fun
bndlw_obj = np.array([-np.pi]*dim)
bndup_obj = np.array([ np.pi]*dim)

# Design of experiments from Latin Hypercube
doe = qmc.LatinHypercube(d=dim)

def from_unit_uniform(X, bndlw, bndup):
    ndata, _ = X.shape
    bndlw = np.repeat(bndlw[np.newaxis,:], ndata, axis=0)
    bndup = np.repeat(bndup[np.newaxis,:], ndata, axis=0)
    X_out = (bndup + bndlw) / 2 + X * (bndup - bndlw) / 2
    return X_out

def to_unit_uniform(X, bndlw, bndup):
    ndata, _ = X.shape
    bndlw = np.repeat(bndlw[np.newaxis,:], ndata, axis=0)
    bndup = np.repeat(bndup[np.newaxis,:], ndata, axis=0)
    X_out = (X - (bndup - bndlw)/2) / ((bndup - bndlw)/2)
    return X_out

def from_unit_gaussian(X, mean, std):
    ndata, _ = X.shape
    mean = np.repeat(mean[np.newaxis,:], ndata, axis=0)
    std = np.repeat(std[np.newaxis,:], ndata, axis=0)
    X_out = mean + std * X
    return X_out

def to_unit_gaussian(X, mean, std):
    ndata, _ = X.shape
    mean = np.repeat(mean[np.newaxis,:], ndata, axis=0)
    std = np.repeat(std[np.newaxis,:], ndata, axis=0)
    X_out = (X - mean) / std
    return X_out

scaler_uniform  = FunctionTransformer(func=to_unit_uniform,  inverse_func=from_unit_uniform,  kw_args={'bndlw':bndlw_obj, 'bndup':bndup_obj}, inv_kw_args={'bndlw':bndlw_obj, 'bndup':bndup_obj})
scaler_gaussian = FunctionTransformer(func=to_unit_gaussian, inverse_func=from_unit_gaussian, kw_args={'bndlw':bndlw_obj, 'bndup':bndup_obj}, inv_kw_args={'bndlw':bndlw_obj, 'bndup':bndup_obj})

# PCE construction
p_vec = range(3, 16)

Si = np.zeros((len(p_vec), dim))
err = np.zeros(len(p_vec))
delta_t = np.zeros(len(p_vec))

for i ,p in enumerate(p_vec):

    print('\nDegree = ', p)

    # pce
    truncation = {'method':'hyperbolic', 'q': 0.5}
    pce = PCE(dim, degree=p, type='UUU', truncation=truncation)

    # cardinality
    card = math.factorial(dim+p)/(math.factorial(dim) * math.factorial(p))
    print('card = ', card)

    # doe
    ndata = int(3*card)
    X = doe.random(n=ndata)
    X = -1 + 2*X    # from [0,1] to [-1,1]

    X_obj = scaler_uniform.inverse_transform(X)

    y = np.zeros(ndata)
    for j in range(ndata):
        y[j] = Ishigami(X_obj[j,:])

    time_init = time.time()
    # coefficients
    pce.compute_coeffs(X, y, method='LSQ')
    print(pce.coeffs.shape)

    # Sobol' indices
    Si[i,:] = pce.sobol_first()
    print('Si', Si[i,:])

    # Leave-One-Out error
    err[i] = pce.compute_err_mse()
    print('err MSE = ', err[i])


