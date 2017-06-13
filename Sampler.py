#! /usr/bin/env python3

import os
import numpy as np
from tqdm import tqdm
from grmodel.fitFuncs import getUniformStart, saveSampling

filename = 'mcmc_chain.h5'


def samplerRun(colI):
    from emcee import EnsembleSampler
    from grmodel import GrowthModel

    # Simulation Constants
    niters = 1E5

    # Adjust thinning based on number of samples
    thin = niters / 1000

    grM = GrowthModel.GrowthModel(colI, complexity=1)

    # Get uniform distribution of positions for start
    p0, ndims, nwalkers = getUniformStart(grM)

    # Set up sampler
    sampler = EnsembleSampler(nwalkers, ndims, grM.logL, threads=16)

    LLbest = -np.inf

    qq = tqdm(total=niters)

    # Run the mcmc walk
    for _, lnn, _ in sampler.sample(p0, iterations=niters, thin=thin):
        LLbest = np.maximum(LLbest, np.amax(lnn))

        qq.set_description('Col: ' + str(colI) + ' LL: ' + str(LLbest))
        qq.update()

    return (sampler, grM)


if os.path.exists(filename):
    os.remove()

# Make iterable of columns
# FIX: Only sampling first few
cols = list(range(3, 4))

for result in map(samplerRun, cols):
    saveSampling('mcmc_chain.h5', result[1], result[0])
