import bz2
import os
import pymc3 as pm
import numpy as np
import scipy as sp
import matplotlib
from matplotlib.backends import backend_pdf

try:
    import cPickle as pickle
except ImportError:
    import pickle


def read_dataset(ff=None):
    ''' Read the specified column from the shared test file. '''

    if ff is None:
        ff = "062117_PC9"

    filename = './grmodel/data/' + ff + '_samples.pkl'

    # Read in list of classes
    classList = pickle.load(bz2.BZ2File(filename, 'r'))

    return classList


def diagnostics(classList, plott=False):
    """ Check the convergence and general posterior properties of a chain. """

    # Iterate over sampling columns
    for ii, item in enumerate(classList):

        # Calc Geweke stats
        geweke = pm.geweke(item.samples)

        # Calculate effective n
        neff = pm.effective_n(item.samples)

        # Calculate Gelman-Rubin statistics
        gr = pm.gelman_rubin(item.samples)

        gewekeOut = 0.0
        gewekeNum = 0

        for key, value in geweke.items():
            for kk, vv in value.items():
                Vec = np.absolute(vv[:, 1])

                gewekeOut += np.sum(Vec)
                gewekeNum += Vec.size

        # Let the z-score surpass 1 up to three times, or fewer with higher deviation
        # TODO: Need to come up with a null model for Geweke to test for convergence
        if False:
            print('Not converged according to Geweke.')
            print(sp.stats.chi2.cdf(gewekeOut, gewekeNum))
            print(gewekeOut)
            print(gewekeNum)
            return False

        if min(neff.values()) < 100:
            print('Effective N of sampling is less than 100.')
            print(neff)
            return False

        if max(gr.values()) > 1.1:
            print('Gelman-Rubin statistic failed.')
            print(gr)
            return False

    # Only output the posterior plot if we've converged
    if plott is True:
        saveplot(classList, pm.plot_posterior)

    # Made it to the end so consistent with converged
    return True


def saveplot(cols, func):
    """ X """
    filename = './grmodel/data/' + cols[0].loadFile + '_' + func.__name__ + '.pdf'

    # Delete the existing file if it exists
    if os.path.exists(filename):
        os.remove(filename)

    with backend_pdf.PdfPages(filename, keep_empty=False) as pdf:
        # Output sampling for each column
        for col in cols:
            fig, axis = matplotlib.pyplot.subplots(9, 2)
            axis = func(col.samples, ax=axis)
            matplotlib.pyplot.tight_layout()
            pdf.savefig(fig)
            matplotlib.pyplot.close()
