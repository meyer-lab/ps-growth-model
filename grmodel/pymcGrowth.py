"""
This module handles experimental data, by fitting a growth and death rate for each condition separately.
"""
import bz2
import os
from os.path import join, dirname, abspath, exists
import pandas
import pickle
import numpy as np
import pymc3 as pm
import theano.tensor as T
import theano


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
    deadapop = params[1] * cGRd * (lnum - 1) / GR + cGRd * (np.exp(-params[1] * ttime) - 1)
    deadnec = b * (lnum - 1) / GR

    out = np.concatenate((np.expand_dims(lnum, axis=1),
                          np.expand_dims(eap, axis=1),
                          np.expand_dims(deadapop, axis=1),
                          np.expand_dims(deadnec, axis=1)), axis=1)
    return out


def build_model(conv0, doses, timeV, expTable):
    ''' Builds then returns the pyMC model. '''
    growth_model = pm.Model()

    with growth_model:
        # Set up conversion rates
        confl_conv = pm.Lognormal('confl_conv', np.log(conv0),      0.1)
        apop_conv  = pm.Lognormal('apop_conv',  np.log(conv0)-2.06, 0.2)
        dna_conv   = pm.Lognormal('dna_conv',   np.log(conv0)-1.85, 0.2)

        # Priors on conv factors
        pm.Lognormal('confl_apop', -2.06, 0.0647, observed=apop_conv / confl_conv)
        pm.Lognormal('confl_dna', -1.85, 0.125, observed=dna_conv / confl_conv)
        pm.Lognormal('apop_dna', 0.222, 0.141, observed=dna_conv / apop_conv)
        
        # Offset values for apop and dna
        apop_offset = pm.Lognormal('apop_offset', -1., 0.1)
        dna_offset  = pm.Lognormal('dna_offset',  -1., 0.1)
        
        # Rate of moving from apoptosis to death, assumed invariant wrt. treatment
        d = pm.Lognormal('d', np.log(0.01), 1)

        # Specify vectors of prior distributions
        # Growth rate
        div = pm.Lognormal('div', np.log(0.02), 1, shape=len(doses))

        # Rate of entering apoptosis or skipping straight to death
        deathRate = pm.Lognormal('deathRate', np.log(0.01), 1, shape=len(doses))

        # Fraction of dying cells that go through apoptosis
        apopfrac = pm.Beta('apopfrac', 2., 2., shape=len(doses))


        # Make a vector of time and one for time-constant values
        timeV = T._shared(timeV)
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
        eap = cGRd * (lnum - pm.math.exp(-d * timeV))

        # Calculate dead cells via apoptosis and via necrosis
        deadnec = b * (lnum - 1) / GR
        deadapop = d * cGRd * (lnum - 1) / GR + cGRd * (pm.math.exp(-d * timeV) - 1)

        # Convert model calculations to experimental measurement units
        confl_exp = (lnum + eap + deadapop + deadnec) * confl_conv
        apop_exp = (eap + deadapop) * apop_conv + apop_offset
        dna_exp = (deadapop + deadnec) * dna_conv + dna_offset

        # Fit model to confl, apop, dna, and overlap measurements
        if ('confl') in expTable.keys():
            # Observed error values for confl
            confl_obs = T.reshape(confl_exp, (-1, )) - expTable['confl']

            pm.Normal('dataFit', sd=T.std(confl_obs), observed=confl_obs)
        if ('apop') in expTable.keys():
            # Observed error values for apop
            apop_obs = T.reshape(apop_exp, (-1, )) - expTable['apop']

            pm.Normal('dataFita', sd=T.std(apop_obs), observed=apop_obs)
        if ('dna') in expTable.keys():
            # Observed error values for dna
            dna_obs = T.reshape(dna_exp, (-1, )) - expTable['dna']

            pm.Normal('dataFitd', sd=T.std(dna_obs), observed=dna_obs)

        pm.Deterministic('logp', growth_model.logpt)

    return growth_model


class GrowthModel:
    def fit(self):
        ''' Run NUTS sampling'''
        print('Building the model')
        model = build_model(self.conv0, self.doses, self.timeV, self.expTable)

        print('Performing inference')
        self.fit = pm.variational.inference.fit(n=80000, model=model)


    def save(self):
        ''' Open file and dump pyMC3 objects through pickle. '''
        if self.interval:
            filePrefix = './grmodel/data/' + self.loadFile
        else:
            filePrefix = './grmodel/data/' + self.loadFile + '_ends'

        if exists(filePrefix + '_samples.pkl'):
            os.remove(filePrefix + '_samples.pkl')

        pickle.dump(self, bz2.BZ2File(filePrefix + '_samples.pkl', 'wb'))


    # Directly import one column of data
    def importData(self, firstCols, comb=None, interval=True):
        """Import experimental data"""
        self.interval = interval

        # Property list
        properties = {'confl': '_confluence_phase.csv',
                      'apop': '_confluence_green.csv',
                      'dna': '_confluence_red.csv'}

        # Find path for csv files in the repository.
        pathcsv = join(dirname(abspath(__file__)), 'data/' + self.loadFile)

        # Pull out selected column data
        self.selCols = []
        self.condNames = []
        self.doses = []
        self.drugs = []
        selconv0 = []

        # Get dict started
        self.expTable = dict()

        # Read in both observation files. Return as formatted pandas tables.
        # Data tables to be kept within class.
        for key, value in properties.items():
            # Read input file
            try:
                dataset = pandas.read_csv(pathcsv + value)
                # If interval=False, get endpoint data
                if not interval:
                    endtime = max(dataset['Elapsed'])
                    data1 = dataset.loc[dataset['Elapsed'] == 0]
                    data2 = dataset.loc[dataset['Elapsed'] == endtime]
                    data = pandas.concat([data1, data2])
                # Otherwise get entire data set
                else:
                    data = dataset
                # Get phase confl was t=0 for confl_conv calculation
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

                # For data with combination therapies
                if comb != None:
                    # Represent dose with a tuple of len(2) in each case
                    # If control
                    if 'Control' in condName:
                        drug = 'Control'
                        dose1 = 0
                        dose2 = 0
                    # If only the combination drug
                    elif condName.split(' ')[0] == comb:
                        drug = comb
                        dose1 = 0
                        dose2 = float(condName.split(' ')[1])
                    # If contains drug besides the combination drug
                    elif 'blank' not in condName:
                        try:  # Both combination drug and another drug
                            drug1str = condName.split(', ')[0]
                            dose1 = float(drug1str.split(' ')[1])
                            combstr = condName.split(', ')[1]
                            dose2 = float(combstr.split(' ')[1])
                            drug = drug1str.split(' ')[0] + '+' + combstr.split(' ')[0]
                        except IndexError:  # Only the other drug
                            drug = condName.split(' ')[0]
                            dose1 = condName.split(' ')[1]
                            dose2 = 0
                    dose = (dose1, dose2)

                    # Add data to expTable
                    self.expTable.setdefault(key, []).append(data.iloc[:, col].as_matrix())

                    # Append to class variables once per column of data
                    if key == 'confl':
                        self.drugs.append(drug)
                        self.doses.append(dose)
                        self.condNames.append(condName)
                        self.selCols.append(col)
                        selconv0.append(conv0[col - firstCols])

                else:  # For data without combinations
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

                        # Append to class variables once per column of data
                        if key == 'confl':
                            self.drugs.append(drug)
                            self.doses.append(dose)
                            self.condNames.append(condName)
                            self.selCols.append(col)
                            selconv0.append(conv0[col - firstCols])
            # Reshape experimental data into 1D array
            self.expTable[key] = np.array(self.expTable[key]).reshape((-1, ))

        # Record averge conv0 for confl prior
        self.conv0 = np.mean(selconv0)

    def __init__(self, loadFile = None):
        # If no filename is given use a default
        if loadFile is None:
            self.loadFile = "101117_H1299"
        else:
            self.loadFile = loadFile
