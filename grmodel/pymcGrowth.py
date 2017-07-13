import matplotlib
matplotlib.use('Agg')
import numpy as np
import pymc3 as pm


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

                # Increment column
                firstCols = firstCols + 1
        except IndexError:
            if len(self.cols) < 2:
                raise ValueError("Didn't find many columns.")

        return firstCols

    def sample(self):
        ''' Map over sampling runs. '''
        for result in map(lambda x: x.sample(), self.cols):
            continue

    def save(self, filename):
        ''' Map over saving runs. '''
        import os

        if os.path.exists(filename):
            os.remove(filename)

        for result in map(lambda x: x.saveTable(filename), self.cols):
            continue

class GrowthModel:

    def sample(self):
        '''
        A
        '''
        with self.model:
            self.samples = pm.sample(500, start=self.getMAP())

    def getMAP(self):
        '''
        Find the MAP point as a starting point.
        '''
        return pm.find_MAP(model=self.model)

    def saveTable(self, filename):
        '''
        Saves a table of sampling results.
        '''
        import h5py

        # Raise an error if there are no sampling results.
        if not hasattr(self, 'samples'):
            raise ValueError("Need to sample first.")

        # Start constructing the dataframe
        df = pm.backends.tracetab.trace_to_dataframe(self.samples)

        df.to_hdf(filename,
                  key='column' + str(self.selCol) + '/chain',
                  complevel=9, complib='bzip2')

        # Open file to pickle class
        f = h5py.File(filename, 'a', libver='latest')

        # Done writing out pickled class
        f.close()


    def build_model(self):
        '''
        Builds then returns the pyMC model.
        '''

        if not hasattr(self, 'timeV'):
            raise ValueError("Need to import data first.")

        growth_model = pm.Model()

        with growth_model:
            div = pm.Lognormal('div', -2, 2)
            b = pm.Lognormal('b', -5, 3)
            c = pm.Lognormal('c', -5, 3)
            d = pm.Lognormal('d', -5, 3)
            
            # Set up conversion rates
            conv = np.mean(self.expTable['confl'][0:25:75])
            confl_conv = pm.Lognormal('confl_conv', np.log(conv), 0.1)
            apopcon = confl_conv * 0.25
            dnacon = confl_conv * 0.125

#            # Priors on conv factors
#            pm.Lognormal('confl_apop', np.log(10.0), 0.1, observed=apop_conv / confl_conv)
#            pm.Lognormal('apop_dna', np.log(2.0), 0.1, observed=apop_conv / dna_conv)

            # Calculate the growth rate
            GR = div - b - c

            # cGDd is used later
            cGRd = c / (GR + d)

            # Calculate the number of live cells
            lnum = pm.math.exp(GR * self.timeV)

            # Number of early apoptosis cells at start is 0.0
            eap = cGRd * (lnum - pm.math.exp(-d * self.timeV))

            # Calculate dead cells
            dead = ((b * lnum + cGRd * d * lnum - b - cGRd * d) / GR +
                    cGRd * pm.math.exp(-d * self.timeV) - cGRd)

            ssqErr = 0.0

            if 'confl' in self.expTable.keys():
                expc = (lnum + dead + eap) * confl_conv
                diff = expc - self.expTable['confl']
                ssqErr = ssqErr + (np.square(diff) / expc).sum()
            if 'apop' in self.expTable.keys():
                expc = (dead + eap) * apopcon + 10**(-2)
                diff = expc - self.expTable['apop']
                ssqErr = ssqErr + (np.square(diff) / expc).sum()
            if 'dna' in self.expTable.keys():
                expc = dead * dnacon + 10**(-2)
                diff = expc - self.expTable['dna']
                ssqErr = ssqErr + (np.square(diff) / expc).sum()

            # Save the sum of squared error
            ssqErr = pm.Deterministic('ssqErr', ssqErr)

            # Error distribution for the expt observations
            if self.selCol <= 8:
                sd = -1
                sdd = 0.5
            else:
                sd = np.log(0.1)
                sdd = 0.2
            pm.ChiSquared('dataFit', self.nobs,
                          observed=ssqErr / pm.Lognormal('std', sd, sdd))

        return growth_model


    def old_model(self, params, confl_conv):
        """
        Solves the ODE function given a set of initial values (y0),
        over a time interval (self.timeV)
        """
        GR = params[0] - params[1] - params[2]

        lnum = np.exp(GR * self.timeV)

        # Number of early apoptosis cells at start is 0.0
        Cone = -params[2] / (GR + params[3])

        eap = params[2] / (GR + params[3]) * np.exp(GR * self.timeV)
        eap = eap + Cone * np.exp(-params[3] * self.timeV)

        dead = (params[1] / GR * np.exp(GR * self.timeV) +
                params[2] * params[3] / (GR * (GR + params[3])) * np.exp(GR * self.timeV) +
                params[2] / (GR + params[3]) * np.exp(-params[3] * self.timeV) -
                params[1] / GR - params[2] * params[3] / (GR * (GR + params[3])) -
                params[2] / (GR + params[3]))

        out = np.concatenate((np.expand_dims(lnum, axis=1),
                              np.expand_dims(dead, axis=1),
                              np.expand_dims(eap, axis=1)), axis=1)
        out = out * confl_conv
        ssqErr = 0.0

        # Run likelihood function with modeled and experiemental data, with
        if 'confl' in self.expTable.keys():
            diff = (lnum + dead + eap) * confl_conv - self.expTable['confl']
            ssqErr = ssqErr + np.sum(np.square(diff)/((lnum + dead + eap) * confl_conv))
        if 'apop' in self.expTable.keys():
            expc = (dead + eap) * 0.25 * confl_conv + 10**(-2)
            diff = expc - self.expTable['apop']
            ssqErr = ssqErr + np.sum(np.square(diff)/expc)
        if 'dna' in self.expTable.keys():
            expc = dead * 0.125 * confl_conv + 10**(-2)
            diff = expc - self.expTable['dna']
            ssqErr = ssqErr + np.sum(np.square(diff)/ expc)
        return (ssqErr, out) 


    def importData(self, selCol, loadFile=None):
        from os.path import join, dirname, abspath
        import pandas

        # If no filename is given use a default
        if loadFile is None:
            loadFile = "030317-2_H1299"

        # Property list
        properties = {'confl': '_confluence_phase.csv',
                      'apop': '_confluence_green.csv',
                      'dna': '_confluence_red.csv'}

        # Find path for csv files in the repository.
        pathcsv = join(dirname(abspath(__file__)), 'data/' + loadFile)

        # Pull out selected column data
        self.selCol = selCol

        # Get dict started
        self.expTable = dict()

        # Read in both observation files. Return as formatted pandas tables.
        # Data tables to be kept within class.
        for key, value in properties.items():
            # Read input file
            try:
                data = pandas.read_csv(pathcsv + value)
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
