#! /usr/bin/env python3

import os
from multiprocessing import Pool
from tqdm import tqdm
from grmodel.fitFuncs import getUniformStart, saveSampling

filename = './grmodel/data/first_chain.h5'


def samplerRun(colI):
    from emcee import EnsembleSampler
    from grmodel import GrowthModel

    # Simulation Constants
    niters = 2E4

    # Adjust thinning based on number of samples
    thin = niters / 100

    grM = GrowthModel.GrowthModel(colI)

    # Get uniform distribution of positions for start
    p0, ndims, nwalkers = getUniformStart(grM)

    # Set up sampler
    sampler = EnsembleSampler(nwalkers, ndims, grM.logL)

    # Run the mcmc walk
    for _, lnn, _ in sampler.sample(p0, iterations=niters, thin=thin):
        continue

    return (sampler, grM)


# Remove the sampling file if it already exists
if os.path.exists(filename):
    os.remove(filename)

# Make iterable of columns
cols = list(range(2, 16))

# Setup the progress bar
qq = tqdm(total=len(cols))

for result in Pool().map(samplerRun, cols):
    qq.update()
    saveSampling(filename, result[1], result[0])
