#! /usr/bin/env python3

import os
import numpy as np
from tqdm import tqdm
from emcee.interruptible_pool import InterruptiblePool
from grmodel.fitFuncs import getUniformStart, saveSampling

filename = './grmodel/data/first_chain.h5'

# Setup the pool so we don't have to restart it each time
pool = InterruptiblePool()


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
    sampler = EnsembleSampler(nwalkers, ndims, grM.logL, pool=pool)

    LLbest = -np.inf

    qq = tqdm(total=niters)

    # Run the mcmc walk
    for _, lnn, _ in sampler.sample(p0, iterations=niters, thin=thin):
        LLbest = np.maximum(LLbest, np.amax(lnn))

        qq.set_description('Col: ' + str(colI) + ' LL: ' + str(LLbest))
        qq.update()

    return (sampler, grM)


# Remove the sampling file if it already exists
if os.path.exists(filename):
    os.remove(filename)

# Make iterable of columns
cols = list(range(2, 6))

for result in map(samplerRun, cols):
    saveSampling(filename, result[1], result[0])
