import numpy as np


def startH5File(filename):
    import h5py

    return h5py.File(filename, 'w', libver='latest')


def getUniformStart(StoneM):
    """ Set up parameters for parallel-tempered Ensemble Sampler """

    ndims, nwalkers = StoneM.Nparams, 3 * StoneM.Nparams

    p0 = np.random.uniform(low=0, high=1, size=(nwalkers, ndims))

    # makes random value 2d array 76 rows by 19 columns

    for ii in range(nwalkers):
        p0[ii] = StoneM.lb + (StoneM.ub - StoneM.lb) * p0[ii]

    return (p0, ndims, nwalkers)
