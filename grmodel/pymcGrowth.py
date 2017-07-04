import numpy as np
import pymc3 as pm


class MultiSample:

    def __init__(self, filename):
        return None



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

    def getTable(self):
        '''

        '''
        return None

    def saveTable(self, filename):
        '''

        '''

        noned = 0.0

    def build_model(self):
        '''
        A
        '''

        if not hasattr(self, 'timeV'):
            raise ValueError("Need to import data first.")

        growth_model = pm.Model()

        with growth_model:
            div = pm.Lognormal('div', -2, 0.5)
            b = pm.Lognormal('b', -3, 1)
            c = pm.Lognormal('c', -3, 1)
            d = pm.Lognormal('d', -3, 1)

            confl_conv = pm.Lognormal('confl_conv', 2, 1)

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
                diff = (lnum + dead + eap) * confl_conv - self.expTable['confl']
                ssqErr = ssqErr + diff.norm(2)

            if 'apop' in self.expTable.keys():
                diff = (lnum + eap) * confl_conv - self.expTable['apop']
                ssqErr = ssqErr + diff.norm(2)

            if 'dna' in self.expTable.keys():
                diff = dead * confl_conv - self.expTable['dna']
                ssqErr = ssqErr + diff.norm(2)

            nobs = len(self.expTable) * len(self.timeV)

            pm.StudentT('dataFit', nu=nobs,
                        lam=pm.Lognormal('std', -2, 1) / nobs,
                        observed=ssqErr)

        return growth_model

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

        # Build the model
        self.model = self.build_model()
