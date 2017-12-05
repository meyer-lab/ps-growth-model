import bz2
from os.path import join, dirname, abspath, exists
import pandas
import numpy as np
import pymc3 as pm
import theano.tensor as T
import theano
from theano.sandbox.linalg import ops as linOps

try:
    import cPickle as pickle
except ImportError:
    import pickle


def lRegRes(inputs, outputs):
    """
    Computers the least squares estimator (LSE) B_hat that minimises the sum of the
    squared errors. Returns the residuals.
    Returns:
        residuals: output_hat - output
        B_hat: [conv_factor, offset]
    """
    X = T.transpose(T.stack([inputs, T.ones([inputs.shape[0]])], axis=0))

    B_hat = T.dot(T.dot(linOps.matrix_inverse(T.dot(X.T, X)),X.T), outputs)

    return (T.dot(X, B_hat) - outputs, B_hat)


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
        self.models = list()

    def loadModels(self, firstCols, fileName = None, seldrugs = None, comb = None):
        """  Initialize GrowthModel for each drug. Load data for each drug in."""
        # Get LoadFile from GrowthModel()
        gr = GrowthModel(fileName)
        gr.importData(firstCols, comb = comb)
        self.models.append(gr)
        return gr.drugs

    def sample(self):
        ''' Map over sampling runs. '''
        for result in map(lambda x: x.sample(), self.models):
            continue

    def save(self):
        ''' Open file and dump pyMC3 objects through pickle. '''
        import os
        if exists(self.filePrefix + '_samples.pkl'):
            os.remove(self.filePrefix + '_samples.pkl')

        pickle.dump(self.models, bz2.BZ2File(self.filePrefix + '_samples.pkl', 'wb'))


class GrowthModel:

    def sample(self):
        ''' A '''
        num = 500
        print(len(self.doses))
        with self.model:
            self.samples = pm.sample(draws=num, tune = num, njobs=2,  # Run three parallel chains
                                     nuts_kwargs={'target_accept': 0.99})

    def build_model(self):
        '''
        Builds then returns the pyMC model.
        '''

        if not hasattr(self, 'timeV'):
            raise ValueError("Need to import data first.")

        growth_model = pm.Model()

        with growth_model:
            # Rate of moving from apoptosis to death, assumed invariant wrt. treatment
            d = pm.Lognormal('d', np.log(0.01), 1)

            # Specify vectors of prior distributions
            # Growth rate
            div = pm.Lognormal('div', np.log(0.02), 1, shape=len(self.doses))

            # Rate of entering apoptosis or skipping straight to death
            deathRate = pm.Lognormal('deathRate', np.log(0.01), 1, shape=len(self.doses))

            # Fraction of dying cells that go through apoptosis
            apopfrac = pm.Uniform('apopfrac', shape=len(self.doses))


            # Make a vector of time and one for time-constant values
            timeV = T._shared(self.timeV)
            constV = T.ones_like(timeV, dtype=theano.config.floatX)


            # Calculate the growth rate
            GR = T.outer(div - deathRate, constV)

            # cGDd is used later
            cGRd = T.outer(deathRate * apopfrac, constV) / (GR + d)

            # b is the rate straight to death
            b = T.outer(deathRate * (1 - apopfrac), constV)


            # Calculate the number of live cells
            lnum = T.exp(GR * timeV)

            # Number of early apoptosis cells at start is 0.0
            eap = cGRd * (lnum - pm.math.exp(-d * self.timeV))

            # Calculate dead cells via apoptosis and via necrosis
            deadnec = b * (lnum - 1) / GR
            deadapop = d * cGRd * (lnum - 1) / GR + cGRd * (pm.math.exp(-d * self.timeV) - 1)

            # Convert to measurement units
            confl_exp = lnum + eap + deadapop + deadnec
            apop_exp = eap + deadapop
            dna_exp = (deadapop + deadnec)
            ovlap_exp = deadapop


            # Fit model to confl, apop, dna, and overlap measurements 
            if ('confl') in self.expTable.keys():
                # Observed error values for confl
                confl_obs, conflBhat = lRegRes(T.reshape(confl_exp, (-1, )), self.expTable['confl'])
                pm.Deterministic('conflBhat', conflBhat)
                pm.Normal('dataFit', sd=T.std(confl_obs), observed=confl_obs)

            if ('apop') in self.expTable.keys():
                # Observed error values for apop
                apop_obs, apopBhat = lRegRes(T.reshape(apop_exp, (-1, )), self.expTable['apop'])
                pm.Deterministic('apopBhat', apopBhat)
                pm.Normal('dataFita', sd=T.std(apop_exp), observed=apop_obs)

            if ('dna') in self.expTable.keys():
                # Observed error values for dna
                dna_obs, dnaBhat = lRegRes(T.reshape(dna_exp, (-1, )), self.expTable['dna'])
                pm.Deterministic('dnaBhat', dnaBhat)
                pm.Normal('dataFitd', sd=T.std(dna_obs), observed=dna_obs)

            if ('overlap') in self.expTable.keys():
                # Observed error values for overlap
                ovlap_obs, ovlapBhat = lRegRes(T.reshape(ovlap_exp, (-1, )), self.expTable['overlap'])
                pm.Deterministic('ovlapBhat',ovlapBhat)
                pm.Normal('dataFito', sd=T.std(ovlap_obs), observed=ovlap_obs)
            pm.Deterministic('logp', growth_model.logpt)

        return growth_model

    # Directly import one column of data
    def importData(self, firstCols, drop24=False, comb = None):
        

        # Property list
        properties = {'confl': '_confluence_phase.csv',
                      'apop': '_confluence_green.csv',
                      'dna': '_confluence_red.csv',
                      'overlap': '_confluence_overlap.csv'}

        # Find path for csv files in the repository.
        pathcsv = join(dirname(abspath(__file__)), 'data/' + self.loadFile)

        # Pull out selected column data
        self.selCols = []
        self.condNames = []
        self.doses = []
        self.drugs = []
        selconv0 =[]

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
                    conv0 = np.mean(data0.iloc[:, firstCols:])
            except FileNotFoundError:
                print("No file for key: " + key)
                continue

            if hasattr(self, 'timeV'):
                # Compare to existing vector
                if np.max(self.timeV - data.iloc[:, 1].as_matrix()) > 0.1:
                    raise ValueError("File time vectors don't match up.")
            else:
                # Set the time vector
                self.timeV = data.iloc[:, 1].as_matrix()

            if not hasattr(self, 'totalCols'):
                self.totalCols = len(data.columns)
            if self.totalCols < firstCols + 2:
                raise ValueError("Didn't find many columns.")

            for col in list(range(firstCols, self.totalCols)):
                # Set the name of the condition we're considering
                condName = data.columns.values[col]

                if comb != None: # For data with combination therapies
                    # Select columns with drug/combination of interest
                    if condName.split(' ')[0] == self.drug or condName.split(' ')[0] == comb or 'Control' in condName:
                        # Represent dose with a tuple of len(2) in each case
                        if 'Control' in condName:
                            drug = 'Control'
                            dose1 = 0
                            dose2 = 0
                        elif condName.split(' ')[0] == comb:
                            dose1 = 0
                            dose2 = float(condName.split(' ')[1])
                        elif condName.split(' ')[0] == self.drug:
                            try:
                                drug1str = condName.split(', ')[0]
                                dose1 = float(drug1str.split(' ')[1])
                                combstr = condName.split(', ')[1]
                                dose2 = float(combstr.split(' ')[1])
                            except IndexError:
                                dose2 = 0
                        dose = (dose1, dose2)

                        # Add data to expTable
                        self.expTable.setdefault(key, []).append(data.iloc[:, col].as_matrix())

                        # Add conv0
                        if key == 'confl':
                            self.doses.append(dose)
                            self.condNames.append(condName)
                            self.selCols.append(col)
                            selconv0.append(conv0[col-firstCols])

                else: # For data without combinations
                    if 'Blank' not in condName:
                        # Add the name of the condition we're considering
                        try:
                            drug = condName.split(' ')[0]
                            dose = condName.split(' ')[1]
                        except IndexError:
                            drug = 'Control'
                            dose = 0

                        # Add data to expTable
                        self.expTable.setdefault(key, []).append(data.iloc[:, col].as_matrix())

                        # Add conv0
                        if key == 'confl':
                            self.drugs.append(drug)
                            self.doses.append(dose)
                            self.condNames.append(condName)
                            self.selCols.append(col)
                            selconv0.append(conv0[col-firstCols])
            self.expTable[key] = np.array(self.expTable[key]).reshape((-1, ))

        # Record averge conv0 for confl prior
        self.conv0 = np.mean(selconv0)

        # Build the model
        self.model = self.build_model()

    def __init__(self, loadFile = None):
        # If no filename is given use a default
        if loadFile is None:
            self.loadFile = "101117_H1299"
        else:
            self.loadFile = loadFile
