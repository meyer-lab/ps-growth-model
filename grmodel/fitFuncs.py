import numpy as np
import h5py

try:
    import cPickle as pickle
except ImportError:
    import pickle

def startH5File(StoneM, filename):
    """ Dump class to a string to store with MCMC chain """
    StoneMs = pickle.dumps(StoneM, pickle.HIGHEST_PROTOCOL)

    f = h5py.File(filename, 'w', libver='latest')
    dset = f.create_dataset("data",
                            chunks=True,
                            maxshape=(None, StoneM.Nparams + 2),
                            data=np.ndarray((0, StoneM.Nparams + 2)))
    dset.attrs["class"] = np.void(StoneMs)
    f.swmr_mode = True

    return (f, dset)

def getUniformStart(StoneM):
    """ Set up parameters for parallel-tempered Ensemble Sampler """
    ndims, nwalkers = StoneM.Nparams, 4*StoneM.Nparams
    p0 = np.random.uniform(low=0, high=1, size=(nwalkers, ndims))

    for ii in range(nwalkers):
        p0[ii] = StoneM.lb + (StoneM.ub - StoneM.lb)*p0[ii]

    return (p0, ndims, nwalkers)
