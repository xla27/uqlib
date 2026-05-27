import os, sys, copy, time
import numpy as np

from SALib.sample import sobol as sob_sample
from SALib.analyze import sobol as sob_analyze



def sobol_wrapper(rom, calc_second=False, return_total=False, n_mc=1024, nproc=1):

    # problem definition (SALib)
    sobol_problem = {
        'num_vars': rom.uq_dim,
        'names'   : [f'x{i+1}' for i in range(rom.uq_dim)],
        'bounds'  : [],
        'dists'   : []
    }

    for _, var in enumerate(rom.pdf_var):

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
    Y_sobol = rom.predict(X_sobol)

    # computing Sobol' indices
    s1 = np.zeros((rom.uq_dim, rom.fom_dim))
    st = np.zeros((rom.uq_dim, rom.fom_dim))
    if calc_second:
        s2 = np.zeros((int(rom.uq_dim * (rom.uq_dim - 1) / 2), rom.fom_dim)) 

    for i_out in range(rom.fom_dim):

        s = sob_analyze.analyze(sobol_problem, 
                            Y_sobol[:,i_out], 
                            calc_second_order=calc_second,
                            n_processors=nproc)
        
        s1[:,i_out] = s['S1']
        st[:,i_out] = s['ST']
        if calc_second:
            s2[:,i_out] = s['S2'][np.triu_indices(rom.uq_dim, k=1)]

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