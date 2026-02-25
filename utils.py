import numpy          as np
from sklearn.preprocessing import FunctionTransformer

# -------------------------------------------------------------------
#  Input Scaler
# -------------------------------------------------------------------

class DataScaler(FunctionTransformer):

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
