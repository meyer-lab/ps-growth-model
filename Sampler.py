from emcee import EnsembleSampler
import numpy as np
from tqdm import tqdm
from grmodel import GrowthModel
from grmodel.fitFuncs import getUniformStart, AddDataH5File, startH5File

bestLL = -np.inf

## Load model
grM = GrowthModel.GrowthModel()

#### Simulation Constants
niters = 100000
thinTrack, thin = -1000, 100

### Make file
f = startH5File("mcmc_chain.h5")

# Get uniform distribution of positions for start
p0, ndims, nwalkers = getUniformStart(grM)

# Make iterable of columns
cols = range(3, len(grM.data_confl.iloc[0,:])-1)

# Make a progress bar
qq = tqdm(total=niters*len(cols))

#in this case, it is column 3 to column 20, go through each
for i in cols:
    nGood, nInf = 0, 0

    grM.setselCol(i)
    
    ## Set up sampler
    sampler = EnsembleSampler(nwalkers, ndims, grM.logL)
    
    # Make new group within f with this column's dataset in the new group
    f, dset = AddDataH5File(grM, f)    
    
    for p, lnprob, lnlike in sampler.sample(p0, iterations=niters, storechain=False):
        qq.update()

        if thinTrack < thin:
            thinTrack += 1
        else:
            nGood += np.sum(np.isfinite(lnprob))
            nInf += np.sum(np.isinf(lnprob))

            if np.max(lnprob) > bestLL:
                bestLL = np.max(lnprob)
                qq.set_description("InfFrac " + str(nInf/(nGood + nInf)) + ", Col " + str(i) + ", Best LL: " + str(bestLL))
                qq.refresh()
    
            matOut = np.concatenate((lnprob.reshape(nwalkers, 1),
                                     np.arange(0, nwalkers).reshape(nwalkers, 1),
                                     p.reshape(nwalkers, ndims)), axis=1)
    
            fShape = dset.shape
            dset.resize((fShape[0] + np.shape(matOut)[0], fShape[1]))
            dset[fShape[0]:, :] = matOut
            f.flush()

            thinTrack = 1
