"""
Dose response analysis to assess the uncertainty that exists when one only uses the live cell number.
"""
from os.path import join, dirname, abspath
import numpy as np
import pymc3 as pm
import theano.tensor as T
import pandas as pd
import matplotlib.pyplot as plt
from pymc3.backends.tracetab import trace_to_dataframe
from .pymcGrowth import simulate


def IC(IC50, X):
    """ Define the IC50 function """
    return IC50[2] + (IC50[1] - IC50[2]) / (1 + 10**(X - np.log10(IC50[0])))


def num(IC_Div, IC_DR, d, apopfrac, ttime, X):
    """ Define the num function to count lnum, eap and dead based on given parameters """
    out = np.empty((len(X), 1, 4))

    for i, x in enumerate(X):
        params = np.array([IC(IC_Div, x), d, IC(IC_DR, x), apopfrac])
        out[i] = simulate(params, ttime)

    return out


def plotCurves(IC_Div, IC_DR, d, apopfrac, ttime):
    """ Plot the curves for (lnum vs. X, eap vs. X, dead vs. X) """
    X = np.linspace(0, 0.5)
    result = np.array(num(IC_Div, IC_DR, d, apopfrac, ttime, X))
    lnum = result[:, 0, 0]
    eap = result[:, 0, 1]
    dead = result[:, 0, 2] + result[:, 0, 3]

    _, ax = plt.subplots(1, 3, figsize=(10, 3))

    ax[0].set_title('lnum vs. X')
    ax[0].set_xlabel('X')
    ax[0].set_ylabel('the number of live cells')
    ax[0].plot(X, lnum)
    ax[0].set_xscale('log')

    ax[1].set_title('eap vs. X')
    ax[1].set_xlabel('X')
    ax[1].set_ylabel('the number of early apoptosis cells')
    ax[1].plot(X, eap)
    ax[1].set_xscale('log')

    ax[2].set_title('dead vs. X')
    ax[2].set_xlabel('X')
    ax[2].set_ylabel('the number of dead cells')
    ax[2].plot(X, dead)
    ax[2].set_xscale('log')

    plt.tight_layout()
    plt.show()


def loadCellTiter(drug=None):
    """ Load Dox and NVB cellTiter Glo data. """
    filename = join(dirname(abspath(__file__)), 'data/initial-data/2017.07.10-H1299-celltiter.csv')

    data = pd.read_csv(filename)

    # Response should be normalized to the control
    data['response'] = data['CellTiter'] / np.mean(data.loc[data['Conc (nM)'] == 0.0, 'CellTiter'])

    # Put the dose on a log scale as well
    data['logDose'] = np.log10(data['Conc (nM)'] + 0.1)

    if drug is None:
        return data
    else:
        return data[data['Drug'] == drug]


def loadIncucyte(drug=None):
    """ Load Dox and NVB Incucyte data. """
    filename = join(dirname(abspath(__file__)), 'data/initial-data/2017.07.10-H1299-red.csv')

    df = pd.read_csv(filename)

    df = pd.melt(df, id_vars=['Elapsed'], var_name='Condition')

    df['Drug'], df['Concentration'] = df['Condition'].str.split('-', 1).str

    df.drop('Condition', axis=1, inplace=True)

    if drug is None:
        return df
    else:
        return df[df['Drug'] == drug]


class doseResponseModel:
    def readSamples(self):
        """ Read in sampling file and return it. """
        self.trace = pm.backends.text.load(self.fname, model=self.model)

    def sample(self):
        ''' Run sampling. '''
        with self.model:
            self.trace = pm.sample(trace=pm.backends.text.Text(self.fname))

    def build_model(self):
        ''' Builds then returns the pyMC model. '''

        if not hasattr(self, 'drugCs'):
            raise ValueError('Need to import data first.')

        doseResponseModel = pm.Model()

        with doseResponseModel:
            # The three values here are div and deathrate
            # Assume just one IC50 for simplicity
            lIC50 = pm.Normal('IC50s', 2.0)
            hill = pm.Lognormal('hill', 0.0)

            Emin_growth = pm.Uniform('Emin_growth', lower=0.0, upper=self.Emax_growth)
            # TODO: Explain large number of external assumptions
            # TODO: Didn't really want this uniform...

            Emax_death = pm.Lognormal('Emax_death', -2.0, 2.0)

            # Import drug concentrations into theano vector
            drugCs = T._shared(self.drugCs)

            # Drug term since we're using constant IC50 and hill slope
            drugTerm = 1.0 / (1.0 + T.pow(10.0, (lIC50 - drugCs) * hill))

            # Do actual conversion to parameters for each drug condition
            growthV = self.Emax_growth + (Emin_growth - self.Emax_growth) * drugTerm

            # _Assuming deathrate in the absence of drug is zero
            deathV = Emax_death * drugTerm

            # Calculate the growth rate
            GR = growthV - deathV

            # Calculate the number of live cells
            lnum = T.exp(GR * self.time)

            # Normalize live cell data to control, as is similar to measurements
            # _Should be index 0
            lExp = lnum / lnum[0]

            # Residual between model prediction and measurement
            residual = self.lObs - lExp

            pm.Normal('dataFitlnum', sd=T.std(residual), observed=residual)

        return doseResponseModel

    def traceplot(self):
        """ Create a plot of the resulting trace. """
        return pm.traceplot(self.trace)

    def __init__(self, Drug=None, filename='sampling'):
        # If no filename is given use a default
        if Drug is None:
            self.drug = 'DOX'
        else:
            self.drug = Drug

        dataLoad = loadCellTiter(self.drug)

        # Handle data import here
        self.drugCs = dataLoad['logDose'].as_matrix()
        self.time = 72.0

        # Based on control kinetic data
        self.Emax_growth = 0.0315

        self.lObs = dataLoad['response'].as_matrix()

        # Build the model
        self.model = self.build_model()

        self.fname = join(dirname(abspath(__file__)), 'data/initial-data/', filename)
