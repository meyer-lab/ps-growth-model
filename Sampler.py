from emcee import EnsembleSampler
import numpy as np
from grmodel import GrowthModel
from grmodel.fitFuncs import getUniformStart, startH5File

newData = True
bestLL = -np.inf

## Load model
grM = GrowthModel.GrowthModel()

#### Run simulation
niters = 100000
selCol = 4
grM.selCol = selCol

# Get uniform distribution of positions for start
p0, ndims, nwalkers = getUniformStart(grM)

## Set up sampler
sampler = EnsembleSampler(nwalkers, ndims, grM.logL, 2.0, [], {}, None, 1)

f, dset = startH5File(grM, "mcmc_chain.h5")
thinTrack = 0
thin = 200

for p, lnprob, lnlike in sampler.sample(p0, iterations=niters, storechain=False):
    if thinTrack < thin:
        thinTrack += 1
    else:
        if np.max(lnprob) > bestLL:
            bestLL = np.max(lnprob)

        matOut = np.concatenate((lnprob.reshape(nwalkers, 1),
                                 np.arange(0, nwalkers).reshape(nwalkers, 1),
                                 p.reshape(nwalkers, ndims)), axis=1)

        fShape = dset.shape
        dset.resize((fShape[0] + np.shape(matOut)[0], fShape[1]))
        dset[fShape[0]:, :] = matOut
        f.flush()

        print((dset.shape, bestLL))
        thinTrack = 1
