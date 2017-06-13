import numpy as np


def getUniformStart(StoneM):
    """ Set up parameters for parallel-tempered Ensemble Sampler """

    ndims, nwalkers = StoneM.Nparams, 2 * StoneM.Nparams + 2

    p0 = np.random.uniform(low=0, high=1, size=(nwalkers, ndims))

    # makes random value 2d array 76 rows by 19 columns

    for ii in range(nwalkers):
        p0[ii] = StoneM.lb + (StoneM.ub - StoneM.lb) * p0[ii]

    return (p0, ndims, nwalkers)


def saveSampling(filename, classM, sampler):
    import h5py
    import pandas as pd

    try:
        import cPickle as pickle
    except ImportError:
        import pickle

    # Start constructing the dataframe
    df = pd.DataFrame(sampler.flatchain, columns=classM.pNames)
    df['LL'] = sampler.flatlnprobability

    df.to_hdf(filename,
              key='column' + str(classM.selCol) + '/chain',
              complevel=9, complib='bzip2')

    print('Best LL: ' + str(np.max(sampler.flatlnprobability)))

    # Open file to pickle class
    f = h5py.File(filename, 'a', libver='latest')

    # Adds new group for this column's dataset
    grp = f.require_group('column' + str(classM.selCol))

    # Dump class to a string to store with MCMC chain
    grp.attrs["class"] = np.void(pickle.dumps(classM,
                                              pickle.HIGHEST_PROTOCOL))

    # Done writing out pickled class
    f.close()
