import numpy as np
import pymc3 as pm
import theano.tensor as T
import theano
import matplotlib.pyplot as plt
from .pymcGrowth import simulate


def IC(IC50, X):
    """ Define the IC50 function """
    return IC50[2] + (IC50[1] - IC50[2]) / (1 + 10**(X - np.log10(IC50[0])))


def num(IC_Div, IC_DR, d, apopfrac, ttime, X):
    """ Define the num function to count lnum, eap and dead based on given parameters """
    out = np.empty((len(X),1,4))

    for i, x in enumerate(X):
        params = np.array([IC(IC_Div, x), d, IC(IC_DR, x), apopfrac])
        out[i] = simulate(params, ttime)

    return out


def plotCurves(IC_Div, IC_DR, d, apopfrac, ttime):
    """ Plot the curves for (lnum vs. X, eap vs. X, dead vs. X) """
    X = np.linspace(0,0.5)
    result = np.array(num(IC_Div, IC_DR, d, apopfrac, ttime, X))
    lnum = result[:,0,0]
    eap = result[:,0,1]
    dead = result[:,0,2] + result[:,0,3]

    fig, ax = plt.subplots(1,3,figsize=(10,3))
    
    ax[0].set_title("lnum vs. X")
    ax[0].set_xlabel("X")
    ax[0].set_ylabel("the number of live cells")
    ax[0].plot(X, lnum)
    ax[0].set_xscale('log')

    ax[1].set_title("eap vs. X")
    ax[1].set_xlabel("X")
    ax[1].set_ylabel("the number of early apoptosis cells")
    ax[1].plot(X, eap)
    ax[1].set_xscale('log')

    ax[2].set_title("dead vs. X")
    ax[2].set_xlabel("X")
    ax[2].set_ylabel("the number of dead cells")
    ax[2].plot(X, dead)
    ax[2].set_xscale('log')
    
    plt.tight_layout()
    plt.show()




#num(np.array([0.5, 1, 0.1]), np.array([0.3, 0.6, 0]), 0.2, 0.6, np.array([72.]), np.array([0.,0.1,0.3,0.5,1]))
#plotCurves(np.array([0.5, 1, 0.1]), np.array([0.3, 0.6, 0]), 0.2, 0.6, np.array([72.]))

class doseResponseModel:

    def sample(self):
        ''' A '''
        num = 1000

        with self.model:
            self.samples = pm.sample(draws=num, tune = num, njobs=2,  # Run three parallel chains
                                     nuts_kwargs={'target_accept': 0.99})

    def build_model(self):
        '''
        Builds then returns the pyMC model.
        '''

        if not hasattr(self, 'drugCs'):
            raise ValueError("Need to import data first.")

        doseResponseModel = pm.Model()

        with doseResponseModel:
            # The three values here are div and deathrate
            # Apopfrac is on end of only IC50s
            IC50s = pm.Lognormal('IC50s', np.log(0.01), 1, shape=3)
            Emin = pm.Lognormal('Emins', np.log(0.01), 1, shape=2)
            Emax = pm.Lognormal('Emaxs', np.log(0.01), 1, shape=2)

            # Apopfrac range handled separately due to bounds
            Emin_apop = pm.Uniform('Emin_apop')
            Emax_apop = pm.Uniform('Emax_apop')

            # D should be constructed the same as in other analysis
            # TODO: Need test for d equivalence
            d = pm.Lognormal('d', np.log(0.01), 1)

            # Import drug concentrations into theano vector
            drugCs = T._shared(self.drugCs)

            EminV = T.concatenate(Emin, Emin_apop)
            EmaxV = T.concatenate(Emax, Emax_apop)

            # This is the value of each parameter, at each drug concentration
            IC50[2] + (IC50[1] - IC50[2]) / (1 + 10**(drugCs - np.log10(IC50s)))


            # Calculate the number of live cells
            lnum = T.exp(GR * self.time)

            # TODO: Fit live cell number to data


        return doseResponseModel

    # Directly import one column of data
    def importData(self):
        # Handle data import here

        # Build the model
        self.model = self.build_model()

    def __init__(self, loadFile = None):
        # If no filename is given use a default
        if loadFile is None:
            self.loadFile = "Filename here"
        else:
            self.loadFile = loadFile