import os, sys

import numpy as np
from sklearn.preprocessing import FunctionTransformer

class PCEDataScaler(FunctionTransformer):

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
                X_out[:, i_pdf] = (2/(up - lw)) * (X[:, i_pdf] - (up + lw)/2) 

            elif pdf == 'N':
                mean = np.repeat(bndlw[i_pdf], ndata)
                std  = np.repeat(bndup[i_pdf], ndata)
                X_out[:, i_pdf] = (X[:, i_pdf] - mean) / std

        return X_out
    

class UnitDataScaler(FunctionTransformer):

    def __init__(self, bndlw, bndup):

        super().__init__(        
            func=self.to_unit,
            inverse_func=self.from_unit,
            validate=False,
            accept_sparse=False,
            check_inverse=True,
            feature_names_out=None,
            kw_args={'bndlw':bndlw, 'bndup':bndup},
            inv_kw_args={'bndlw':bndlw, 'bndup':bndup},)

    def from_unit(self, X, bndlw, bndup):
        ndata, dim = X.shape
        X_out = np.repeat(bndlw[np.newaxis,:], ndata, axis=0) + X * (np.repeat(bndup[np.newaxis,:], ndata, axis=0) - np.repeat(bndlw[np.newaxis,:], ndata, axis=0))
        return X_out


    def to_unit(self, X, bndlw, bndup):
        ndata, dim = X.shape
        X_out = (X - np.repeat(bndlw[np.newaxis,:], ndata, axis=0)) / (np.repeat(bndup[np.newaxis,:], ndata, axis=0) - np.repeat(bndlw[np.newaxis,:], ndata, axis=0))
        return X_out
    
    def transform_jac(self):
        bndlw, bndup = self.kw_args.values()
        return 1 / (bndup - bndlw)

