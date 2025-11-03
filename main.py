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

# Ishigami function
dim = 3
def Ishigami(x):
    fun = np.sin(x[:,0]) + 7 * np.sin(x[:,1])**2 + 0.1 * x[:,2]**4 * np.sin(x[:,0])
    return fun
bndlw = np.array([-np.pi]*dim)
bndup = np.array([ np.pi]*dim)

pdf_var = ['U', 'U', 'U']

# Design of experiments from Latin Hypercube
doe = DoE(dim, method='QUADRATURE', pdf_var=pdf_var)

X, w = doe(point_per_dim=8)
ndata, dim = X.shape

scaler = DataScaler(pdf_var, bndlw, bndup)

# PCE construction
p = 6

print('\nDegree = ', p)

# pce
truncation = {'method':'standard'}
pce1 = PCE(dim, degree=p, pdf_var=pdf_var, truncation=truncation)
pce2 = PCE(dim, degree=p, pdf_var=pdf_var, truncation=truncation)

# cardinality
card = math.factorial(dim+p)/(math.factorial(dim) * math.factorial(p))
print('card = ', card, '\tndata = ', ndata)

# ydata
X_obj = scaler.inverse_transform(X)

y = Ishigami(X_obj)

time_init = time.time()
# coefficients
pce1.compute_coeffs(X, y, method='LSQ')
pce2.compute_coeffs(X, y, method='PROJ', weights=w)

# Sobol' indices
print('Si LSQ', pce1.sobol_first())
print('Si PROJ', pce2.sobol_first())

# scoring the two surrogates on test points
ntest = 1000
xtest = np.random.uniform(-np.pi, np.pi, size=(ntest,dim))

ytest = Ishigami(xtest)
ypred1 = pce1.predict(scaler.transform(xtest))
ypred2 = pce2.predict(scaler.transform(xtest))

from sklearn.metrics import r2_score
print('R2 PCE 1 = ', r2_score(ytest, ypred1))
print('R2 PCE 2 = ', r2_score(ytest, ypred2))

