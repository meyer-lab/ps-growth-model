from emcee import EnsembleSampler
import numpy as np
from grmodel import GrowthModel
from grmodel.fitFuncs import getUniformStart, AddDataH5File, startH5File

bestLL = -np.inf

## Load model
grM = GrowthModel.GrowthModel()

#### Simulation Constants
niters = 100000
thinTrack = 0
thin = 200

### Make file
f = startH5File("mcmc_chain.h5")

#in this case, it is column 3 to column 20, go through each
for i in range(3, len(grM.data_confl.iloc[0,:])-1):        
    grM.setselCol(i)  
    
    # Get uniform distribution of positions for start
    #can we just use this for every column? --> less computations then
    p0, ndims, nwalkers = getUniformStart(grM)
    
    ## Set up sampler
    sampler = EnsembleSampler(nwalkers, ndims, grM.logL)
    
    #make new group within f with this column's dataset in the new group
    f, dset = AddDataH5File(grM, f)    
    
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
