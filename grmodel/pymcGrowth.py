import bz2
from os.path import join, dirname, abspath
import pandas
import numpy as np
import pymc3 as pm

try:
    import cPickle as pickle
except ImportError:
    import pickle


def ssq(expc, expt):
    return np.square((expc - expt) / expc).sum()


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
        pickle.dump(self.cols, bz2.BZ2File(self.filePrefix + '_samples.pkl', 'w'))


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
            div = pm.Lognormal('div', np.log(0.01), 1)

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
            dead = ((b * lnum + cGRd * d * lnum - b - cGRd * d) / GR +
                    cGRd * pm.math.exp(-d * self.timeV) - cGRd)

            ssqErr = 0.0

            # Set up conversion rates
            confl_conv = pm.Lognormal('confl_conv', np.log(self.conv0), 0.1)
            apop_conv = pm.Lognormal('apop_conv', np.log(self.conv0 * 0.204), 0.2)
            dna_conv = pm.Lognormal('dna_conv', np.log(self.conv0 * 0.144), 0.2)

            # Priors on conv factors
            pm.Lognormal('confl_apop', np.log(0.204), 0.1, observed=apop_conv / confl_conv)
            pm.Lognormal('apop_dna', np.log(0.144), 0.1, observed=dna_conv / confl_conv)

            if 'confl' in self.expTable.keys():
                ssqErr += ssq((lnum + dead + eap) * confl_conv, self.expTable['confl'])
            if 'apop' in self.expTable.keys():
                ssqErr += ssq((dead + eap) * apop_conv, self.expTable['apop'])
            if 'dna' in self.expTable.keys():
                ssqErr += ssq(dead * dna_conv, self.expTable['dna'])

            # Save the sum of squared error
            ssqErr = pm.Deterministic('ssqErr', ssqErr)

            # Error distribution for the expt observations
            pm.ChiSquared('dataFit', self.nobs,
                          observed=ssqErr / pm.Lognormal('std', -2, 1))

        return growth_model

    def importData(self, selCol, loadFile=None):
        # If no filename is given use a default
        if loadFile is None:
            self.loadFile = "062117_PC9"
        else:
            self.loadFile = loadFile

        # Property list
        properties = {'confl': '_confluence_phase.csv',
                      'apop': '_confluence_green.csv',
                      'dna': '_confluence_red.csv'}

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
                data = dataset.loc[dataset['Elapsed'] >= 24]
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
