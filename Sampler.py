#! /usr/bin/env python3

from tqdm import tqdm
from grmodel.fitFuncs import getUniformStart, saveSampling


def samplerRun(colI):
    from emcee import EnsembleSampler
    from grmodel import GrowthModel

    # Simulation Constants
    niters = 1E3

    # Adjust thinning based on number of samples
    thin = niters / 100

    grM = GrowthModel.GrowthModel(colI)

    # Get uniform distribution of positions for start
    p0, ndims, nwalkers = getUniformStart(grM)

    # Set up sampler
    sampler = EnsembleSampler(nwalkers, ndims, grM.logL, threads=16)

    # Run the mcmc walk
    for _ in tqdm(sampler.sample(p0, iterations=niters, thin=thin),
                  total=niters, desc='Col: ' + str(colI)):
        continue

    return (sampler, grM)


# Make iterable of columns
# FIX: Only sampling first few
cols = list(range(3, 5))

for result in map(samplerRun, cols):
    saveSampling('mcmc_chain.h5', result[1], result[0])
