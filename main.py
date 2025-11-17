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
from doe import DoE, DataScaler

PROBLEM = input('Problem: ')
np.set_printoptions(precision=4,threshold=sys.maxsize)

# Ishigami function
def Ishigami():
    fun = lambda x: np.sin(x[:,0]) + 7 * np.sin(x[:,1])**2 + 0.1 * x[:,2]**4 * np.sin(x[:,0])
    dim = 3
    bndlw = np.array([-np.pi]*dim)
    bndup = np.array([ np.pi]*dim)
    V_i = np.array([0.3138, 0.4424, 0.0])
    return fun, dim, bndlw, bndup, V_i

# Sobol G function 
def Sobol_G():
    a = np.array([0, 1, 4.5, 9, 99, 99, 99, 99])
    dim = 8
    def fun(x):
        a = np.array([0, 1, 4.5, 9, 99, 99, 99, 99])
        a = np.repeat(a[np.newaxis,:],repeats=x.shape[0],axis=0)
        return np.prod((np.abs(4*x - 2) + a)/(1+a), axis=1)
    bndlw = np.array([0]*dim)
    bndup = np.array([1]*dim)
    V_i = np.array([1 / (3 * (1 + a[i])**2) for i in range(dim)])
    V = np.prod(V_i+1)-1
    return fun, dim, bndlw, bndup, V_i, V

if PROBLEM == 'Ishigami':
    obj, dim, bndlw, bndup, V_i = Ishigami()
    pdf_var = ['U']*dim

elif PROBLEM == 'Sobol G':
    obj, dim, bndlw, bndup, V_i, V = Sobol_G() 
    pdf_var = ['U']*dim


# Design of experiments from Latin Hypercube
doe = DoE(dim, method='MC', pdf_var=pdf_var)

scaler = DataScaler(pdf_var, bndlw, bndup)

for p in range(3, 15):
    # PCE construction

    print('\nDegree = ', p)

    # pce
    truncation = {'method':'hyperbolic', 'q': 0.75}
    pce = PCE(dim, degree=p, pdf_var=pdf_var, truncation=truncation)

    # cardinality
    card = math.factorial(dim+p)/(math.factorial(dim) * math.factorial(p))
    ndata= int(3*card)
    print('card = ', card, '\tndata = ', ndata)

    # ydata
    X, w = doe(ndata=ndata)
    ndata, dim = X.shape
    X_obj = scaler.inverse_transform(X)

    y = obj(X_obj)

    time_init = time.time()
    # coefficients
    pce.compute_coeffs(X, y, method='LSQ')

    # Moments
    if PROBLEM == 'Sobol G':
        _, var = pce.moments()
        print('Variance', var)
        print('Exact V', V)

    # Sobol' indices
    print('Si  LSQ', pce.sobol_first())
    print('Exact Si', V_i)
    print('Sij LSQ', pce.sobol_second())

    # scoring the two surrogates on test points
    ntest = 1000
    xtest = scaler.inverse_transform(doe(ndata=ntest)[0])

    ytest = obj(xtest)
    ypred = pce.predict(scaler.transform(xtest))

    from sklearn.metrics import r2_score
    print('R2 PCE = ', r2_score(ytest, ypred))

