import bz2
from os.path import join, dirname, abspath, exists
import pandas
import numpy as np
import pymc3 as pm

try:
    import cPickle as pickle
except ImportError:
    import pickle


def ssq(expc, expt):
    return np.square(expc - expt).sum()

def simulate(params, ttime):
    ''' Takes in params for parameter values and ttimes, a list or array of times
    params[0] = div
    params[1] = d
    params[2] = deathRate
    params[3] = apopfrac
    params[4] = confl_conv
    params[5] = apop_conv
    params[6] = dna_conv
    '''
    # Calculate the growth rate
    GR = params[0] - params[2]

    # Calculate the number of live cells
    lnum = np.exp(GR * ttime)

    # cGDd is used later
    cGRd = params[2] * params[3] / (GR + params[1]) 

    # Number of early apoptosis cells at start is 0.0
    eap = cGRd * (lnum - np.exp(-params[1] * ttime))

    # b is the rate straight to death
    b = params[2] * (1 - params[3])

    # Calculate dead cells
    deadapop = params[1] * cGRd * (lnum - 1) / GR + cGRd * (np.exp(-params[1] * ttime) -1)
    deadnec = b * (lnum -1) / GR

    out = np.concatenate((np.expand_dims(lnum, axis=1),
                         np.expand_dims(eap, axis=1),
                         np.expand_dims(deadapop, axis=1),
                         np.expand_dims(deadnec, axis=1)), axis=1)
    return out


class MultiSample:

    def __init__(self):
        self.cols = list()

    def loadCols(self, firstCols, filename=None):
        """ Load columns in. Run through columns until the loading process errors. """
        try:
            while (True):
                # Create new class
                temp = GrowthModel()

                # Import data column
                temp.importData(firstCols, filename)

                # Stick the class into the list
                self.cols.append(temp)

                # If filename for output not yet set, save it
                if not hasattr(self, 'filePrefix'):
                    self.filePrefix = './grmodel/data/' + temp.loadFile

                # Increment to next column
                firstCols += 1
        except IndexError:
            if len(self.cols) < 2:
                raise ValueError("Didn't find many columns.")

        return firstCols

    def sample(self):
        ''' Map over sampling runs. '''
        for result in map(lambda x: x.sample(), self.cols):
            continue

    def save(self):
        ''' Open file and dump pyMC3 objects through pickle. '''
        import os
        if exists(self.filePrefix + '_samples.pkl'):
            os.remove(self.filePrefix + '_samples.pkl')

        pickle.dump(self.cols, bz2.BZ2File(self.filePrefix + '_samples.pkl', 'wb'))


class GrowthModel:

    def sample(self):
        ''' A '''

        with self.model:
            self.samples = pm.sample(njobs=3,  # Run three parallel chains
                                     nuts_kwargs={'target_accept': 0.99})

    def build_model(self):
        '''
        Builds then returns the pyMC model.
        '''

        if not hasattr(self, 'timeV'):
            raise ValueError("Need to import data first.")

        growth_model = pm.Model()

        with growth_model:
            # Growth rate
            div = pm.Lognormal('div', np.log(0.02), 1)

            # Rate of moving from apoptosis to death
            d = pm.Lognormal('d', np.log(0.01), 1)

            # Rate of entering apoptosis or skipping straight to death
            deathRate = pm.Lognormal('deathRate', np.log(0.01), 1)

            # Fraction of dying cells that go through apoptosis
            apopfrac = pm.Uniform('apopfrac')

            # Calculate the growth rate
            GR = div - deathRate

            # Calculate the number of live cells
            lnum = pm.math.exp(GR * self.timeV)

            # cGDd is used later
            cGRd = deathRate * apopfrac / (GR + d)

            # Number of early apoptosis cells at start is 0.0
            eap = cGRd * (lnum - pm.math.exp(-d * self.timeV))

            # b is the rate straight to death
            b = deathRate * (1 - apopfrac)

            # Calculate dead cells
            deadapop = d * cGRd * (lnum - 1) / GR + cGRd * (pm.math.exp(-d * self.timeV) -1)
            deadnec = b * (lnum -1) / GR
#            dead = ((b * lnum + cGRd * d * lnum - b - cGRd * d) / GR +
#                    cGRd * pm.math.exp(-d * self.timeV) - cGRd)

            ssqErr = 0.0

            # Set up conversion rates
            confl_conv = pm.Lognormal('confl_conv', np.log(self.conv0), 0.1)
            apop_conv = pm.Lognormal('apop_conv', np.log(self.conv0 * 0.25), 0.4)
            dna_conv = pm.Lognormal('dna_conv', np.log(self.conv0 * 0.144), 0.4)

            # Priors on conv factors
            pm.Lognormal('confl_apop', np.log(0.25), 0.2, observed=apop_conv / confl_conv)
            pm.Lognormal('confl_dna', np.log(0.144), 0.2, observed=dna_conv / confl_conv)
            
            # TODO: Account for the fact that apop and dna can't exceed confl
            if 'confl' in self.expTable.keys():
                ssqErr += ssq((lnum + eap + deadapop + deadnec), self.expTable['confl'] / confl_conv)
            if 'apop' in self.expTable.keys():
                ssqErr += ssq((eap + deadapop), self.expTable['apop'] / apop_conv)
            if 'dna' in self.expTable.keys():
                ssqErr += ssq((deadapop + deadnec), self.expTable['dna'] / dna_conv)
            if 'overlap' in self.expTable.keys():
                ssqErr += ssq(deadapop, self.expTable['overlap'] / dna_conv)

            # Save the sum of squared error
            ssqErr = pm.Deterministic('ssqErr', ssqErr)

            # Error distribution for the expt observations
            pm.ChiSquared('dataFit', self.nobs,
                          observed=ssqErr / pm.Lognormal('std', -2, 1))

        return growth_model

    def importData(self, selCol, loadFile=None, drop24=True):
        # If no filename is given use a default
        if loadFile is None:
            self.loadFile = "042017_PC9"
        else:
            self.loadFile = loadFile

        # Property list
        properties = {'confl': '_confluence_phase.csv',
                      'apop': '_confluence_green.csv',
                      'dna': '_confluence_red.csv',
                      'overlap': '_confluence_overlap.csv'}

        # Find path for csv files in the repository.
        pathcsv = join(dirname(abspath(__file__)), 'data/' + self.loadFile)

        # Pull out selected column data
        self.selCol = selCol

        # Get dict started
        self.expTable = dict()

        # Read in both observation files. Return as formatted pandas tables.
        # Data tables to be kept within class.
        for key, value in properties.items():
            # Read input file
            try:
                dataset = pandas.read_csv(pathcsv + value)
                if drop24:
                    data = dataset.loc[dataset['Elapsed'] >= 24]
                else:
                    data = dataset
                if key == 'confl':
                    data0 = dataset.loc[dataset['Elapsed'] == 0]
                    self.conv0 = np.mean(data0.iloc[:, self.selCol])
            except FileNotFoundError:
                print("No file for key: " + key)
                continue

            # Write data into array
            self.expTable[key] = data.iloc[:, self.selCol].as_matrix()

            if hasattr(self, 'timeV'):
                # Compare to existing vector
                if np.max(self.timeV - data.iloc[:, 1].as_matrix()) > 0.1:
                    raise ValueError("File time vectors don't match up.")
            else:
                # Set the time vector
                self.timeV = data.iloc[:, 1].as_matrix()

                # Set the name of the condition we're considering
                self.condName = data.columns.values[self.selCol]

        # Record the number of observations we've made
        self.nobs = len(self.expTable) * len(self.timeV)

        # Build the model
        self.model = self.build_model()
