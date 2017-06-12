#! /usr/bin/env python3

import numpy as np
from tqdm import tqdm
from grmodel.fitFuncs import getUniformStart, startH5File
from multiprocessing import Pool

try:
    import cPickle as pickle
except ImportError:
    import pickle


def samplerRun(colI):
    from emcee import EnsembleSampler
    from grmodel import GrowthModel

    # Simulation Constants
    niters, thin = 1E5, 1E3

    grM = GrowthModel.GrowthModel(colI)

    # Get uniform distribution of positions for start
    p0, ndims, nwalkers = getUniformStart(grM)

    # Set up sampler
    sampler = EnsembleSampler(nwalkers, ndims, grM.logL)

    # Run the mcmc walk
    for _ in sampler.sample(p0, iterations=niters, thin=thin):
        continue

    return (sampler, grM)


# Make iterable of columns
cols = list(range(3, 21))

# Setup the parallel pool
p = Pool()

# Open the output hdf5 file
f = startH5File('mcmc_chain.h5')

for result in tqdm(p.imap_unordered(samplerRun, cols), total=len(cols)):
    # Adds new group for this column's dataset
    grp = f.create_group('column' + str(result[1].selCol))

    # Dump class to a string to store with MCMC chain
    grp.attrs["class"] = np.void(pickle.dumps(result[1],
                                              pickle.HIGHEST_PROTOCOL))

    # Dump chain
    dset = grp.create_dataset("chain", data=result[0].chain, dtype='f2')

    # Dump log-probabilities
    grp.create_dataset("lnprob", data=result[0].lnprobability, dtype='f2')

# Close after done writing
f.close()
