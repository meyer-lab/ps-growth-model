import numpy as np
import pymc3 as pm
import theano.tensor as T
import pandas as pd
import matplotlib.pyplot as plt
import theano
import pickle
from os.path import join, dirname, abspath
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


def loadCellTiter(drug=None):
    """ Load Dox and NVB cellTiter Glo data. """
    filename = join(dirname(abspath(__file__)), 'data/initial-data/2017.07.10-H1299-celltiter.csv')

    data = pd.read_csv(filename)

    # Response should be normalized to the control
    data['response'] = data['CellTiter'] / np.mean(data.loc[data['Conc (nM)'] == 0.0, 'CellTiter'])

    # Put the dose on a log scale as well
    data['logDose'] = np.log(data['Conc (nM)'] + 0.1)

    if drug is None:
        return data
    else:
        return data[data['Drug'] == drug]


def loadIncucyte(drug=None):
    """ Load Dox and NVB Incucyte data. """
    filename = join(dirname(abspath(__file__)), 'data/initial-data/2017.07.10-H1299-red.csv')

    df = pd.read_csv(filename)

    df = pd.melt(df, id_vars = ['Elapsed'], var_name = 'Condition')

    df['Drug'], df['Concentration'] = df['Condition'].str.split('-', 1).str

    df.drop('Condition', axis=1, inplace=True)

    if drug is None:
        return df
    else:
        return df[df['Drug'] == drug]


def save(classname, filename):
    with open(filename,"wb") as file:
         pickle.dump(classname, file, pickle.HIGHEST_PROTOCOL)


def read(filename):
    with open(filename,"rb") as file:
        try:
            while True:
                pickel_model = pickle.load(file)
                return pickel_model
        except EOFError:
            pass

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
            IC50s = pm.Lognormal('IC50s', np.log(0.01), 1, shape=3, testval=[1.0, 2.0, 3.0])
            Emin = pm.Lognormal('Emins', np.log(0.01), 1, shape=2, testval=[0.3, 0.3])
            Emax = pm.Lognormal('Emaxs', np.log(0.01), 1, shape=2, testval=[0.9, 0.1])

            # Apopfrac range handled separately due to bounds
            Emin_apop = pm.Uniform('Emin_apop', testval=0.1)
            Emax_apop = pm.Uniform('Emax_apop', testval=0.9)

            # D should be constructed the same as in other analysis
            # TODO: Test for d equivalence
            d = pm.Lognormal('d', np.log(0.01), 1)

            # Import drug concentrations into theano vector
            drugCs = T._shared(self.drugCs)

            drugsT = T.transpose(T.outer(T.transpose(drugCs), T.ones_like(IC50s, dtype=theano.config.floatX)))
            
            constV = T.ones_like(drugCs, dtype=theano.config.floatX)

            EminV = T.concatenate((Emin, T.stack(Emin_apop)))
            EmaxV = T.concatenate((Emax, T.stack(Emax_apop)))

            # This is the value of each parameter, at each drug concentration
            rangeT = T.outer(EmaxV - EminV, constV)
            lIC50T = T.outer(T.log10(IC50s), constV)

            params = T.outer(EminV, constV) + rangeT / (1 + 10**(drugsT - lIC50T))

            # Calculate the growth rate
            GR = params[0, :] - params[1, :]

            # Calculate the number of live cells
            lnum = pm.Deterministic('lnum', T.exp(GR * self.time))

            # cGDd is used later
            cGRd = (params[1, :] * params[2, :]) / (GR + d)

            # b is the rate straight to death
            b = params[1, :] * (1 - params[2, :])

            # Number of early apoptosis cells at start is 0.0
            eap = cGRd * (lnum - pm.math.exp(-d * self.time))

            # Calculate dead cells via apoptosis and via necrosis
            deadnec = b * (lnum - 1) / GR
            deadapop = d * cGRd * (lnum - 1) / GR + cGRd * (pm.math.exp(-d * self.time) - 1)

            # Conversion value is used to match lObs and lnum
            conv = pm.Lognormal('conv', mu=np.log(0.01), sd=1)

            # TODO: Fit live cell number to data
            lObs = self.lObs
            lExp = pm.Deterministic('lExp', lnum * conv)
            residual = lObs - lExp
            pm.Normal('dataFitlnum', sd = T.std(residual), observed = residual)

        return doseResponseModel

    # Traceplot
    def plot(self):
        pm.traceplot(self.samples)

    # Directly import one column of data
    def importData(self):
        dataLoad = loadCellTiter(self.drug)

        # Handle data import here
        self.drugCs = dataLoad['logDose'].as_matrix()
        self.time = 72.0

        self.lObs = dataLoad['response'].as_matrix()

        # Build the model
        self.model = self.build_model()

    def __init__(self, Drug = None):
        # If no filename is given use a default
        if Drug is None:
            self.drug = "DOX"
        else:
            self.drug = Drug
