import numpy as np
import pymc3 as pm


def simulate(params, ts):
    """
    Solves the ODE function given a set of initial values (y0),
    over a time interval (ts)

    params:
    params  list of parameters for model (a, b, c, d, e)
    ts  time interval over which to solve the function

    y0  list with the initial values for each state

    ODE function of cells living/dying/undergoing early apoptosis

        params:
        ss   the number of cells in a particular state
                (LIVE, DEAD, EARLY_APOPTOSIS)
        t   time
        a   parameter between LIVE -> LIVE (cell division)
        b   parameter between LIVE -> DEAD
        c   parameter between LIVE -> EARLY_APOPTOSIS
        d   parameter between EARLY_APOPTOSIS -> DEATH
        e   parameter between DEATH -> GONE
    """
    GR = params[0] - params[1] - params[2]

    lnum = np.exp(GR * ts)

    # Number of early apoptosis cells at start is 0.0
    Cone = -params[2] / (GR + params[3])

    eap = params[2] / (GR + params[3]) * np.exp(GR * ts)
    eap = eap + Cone * np.exp(-params[3] * ts)

    dead = (params[1] / GR * np.exp(GR * ts) +
            params[2] * params[3] / (GR * (GR + params[3])) * np.exp(GR * ts) +
            params[2] / (GR + params[3]) * np.exp(-params[3] * ts) -
            params[1] / GR - params[2] * params[3] / (GR * (GR + params[3])) -
            params[2] / (GR + params[3]))

    # Add numbers to the output matrix
    out = np.concatenate((np.expand_dims(lnum, axis=1),
                          np.expand_dims(dead, axis=1),
                          np.expand_dims(eap, axis=1)), axis=1)

    return out

# Previous code for comparing to data
    # # Calculate model data table
    # model = simulate(paramV, self.timeV)

    # # Run likelihood function with modeled and experiemental data, with
    # # standard deviation given by last two entries in paramV
    # if 'confl' in self.expTable.keys():
    #     confl_mod = paramV[-3] * np.sum(model, axis=1)

    #     log_likelihood = np.sum(logpdf_sum(self.expTable['confl'],
    #                             loc=confl_mod, scale=paramV[-2]))

    # if 'apop' in self.expTable.keys():
    #     green_mod = paramV[-3] * (model[:, 1] + model[:, 2])

    #     log_likelihood += np.sum(logpdf_sum(self.expTable['apop'],
    #                              loc=green_mod, scale=paramV[-1]))

    # if 'dna' in self.expTable.keys():
    #     dna_mod = paramV[-3] * model[:, 1]

    #     log_likelihood += np.sum(logpdf_sum(self.expTable['dna'],
    #                              loc=dna_mod, scale=paramV[-1]))


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

            # Save the sum of squared error
            ssqErr = pm.Deterministic('ssqErr', ssqErr)

            # Error distribution for the expt observations
            pm.ChiSquared('dataFit', self.nobs,
                          observed=ssqErr / pm.Lognormal('std', -2, 1))

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

        # Record the number of observations we've made
        nobs = len(self.expTable) * len(self.timeV)

        # Build the model
        self.model = self.build_model()
