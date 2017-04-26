from emcee import EnsembleSampler
import numpy as np
from grmodel import GrowthModel
from grmodel.fitFuncs import getUniformStart

bestLL = -np.inf

## Load model
grM = GrowthModel.GrowthModel()

#### Run simulation
niters = 300
grM.setselCol(4)

# Get uniform distribution of positions for start
p0, ndims, nwalkers = getUniformStart(grM)

## Set up sampler
sampler = EnsembleSampler(nwalkers, ndims, grM.logL, 2.0, [], {}, None, 1)

for p, lnprob, lnlike in sampler.sample(p0, iterations=niters, storechain=False):
    if np.max(lnprob) > bestLL:
        bestLL = np.max(lnprob)
        print(bestLL)
