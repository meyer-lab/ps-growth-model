import bz2
from os.path import join, dirname, abspath, exists
import pandas
import numpy as np
import pymc3 as pm
import theano.tensor as T

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
        self.models = list()

    def loadModels(self, firstCols, fileName = None, seldrugs = None, comb = None):
        """  Initialize GrowthModel for each drug. Load data for each drug in."""
        # Get LoadFile from GrowthModel()
        temp = GrowthModel(fileName)

        if not hasattr(self, 'filePrefix'):
            self.filePrefix = './grmodel/data/' + temp.loadFile

        # Find path for csv files in the repository.
        pathcsv = join(dirname(abspath(__file__)), 'data/' + temp.loadFile)
        dataset = pandas.read_csv(pathcsv + '_confluence_phase.csv')
        if seldrugs:
            drugs = seldrugs
        else:
            conditions = dataset.columns.values[firstCols:]
            alldrugs = [cond.split(' ')[0] for cond in conditions]
            drugs = []
            for drug in alldrugs:
                if drug not in drugs:
                    drugs.append(drug)
            drugs = [x for x in drugs if 'Control' and 'Blank' not in x]
        for drug in drugs:
            gr = GrowthModel()
            gr.importData(firstCols, drug, comb = comb)
            self.models.append(gr)
        return drugs

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
        num = 1000
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
            # Set up conversion rates
            confl_conv = pm.Lognormal('confl_conv', np.log(self.conv0), 0.1)
            apop_conv = pm.Lognormal('apop_conv', np.log(self.conv0 * 0.25), 0.2)
            dna_conv = pm.Lognormal('dna_conv', np.log(self.conv0 * 0.144), 0.2)

            # Priors on conv factors
            pm.Lognormal('confl_apop', np.log(0.25), 0.1, observed=apop_conv / confl_conv)
            pm.Lognormal('confl_dna', np.log(0.144), 0.1, observed=dna_conv / confl_conv)
            
            # Offset values for green and red
            gos = pm.Uniform('gos', lower=0, upper=0.2)
            ros = pm.Uniform('ros', lower=0, upper=0.2)
            oos = pm.Uniform('oos', lower=0, upper=0.05)

            # Rate of moving from apoptosis to death, assumed invariant wrt. treatment
            d = pm.Lognormal('d', np.log(0.01), 1)

            # Iterate over each dose
            for dose in self.doses:
                # Specify vectors of prior distributions
                # Growth rate
                div = pm.Lognormal('div '+str(dose), np.log(0.02), 1)

                # Rate of entering apoptosis or skipping straight to death
                deathRate = pm.Lognormal('deathRate '+str(dose), np.log(0.01), 1)

                # Fraction of dying cells that go through apoptosis
                apopfrac = pm.Uniform('apopfrac '+str(dose))
                
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

                # Calculate dead cells via apoptosis and via necrosis
                deadapop = d * cGRd * (lnum - 1) / GR + cGRd * (pm.math.exp(-d * self.timeV) -1)
                deadnec = b * (lnum -1) / GR

                # Fit model to confl, apop, dna, and overlap measurements 
                if (dose, 'confl') in self.expTable.keys():
                    # Observed error values for confl
                    confl_obs = (lnum + eap + deadapop + deadnec) * confl_conv - self.expTable[(dose, 'confl')]

                    pm.Normal('dataFit '+str(dose), sd = T.std(confl_obs), observed = confl_obs)
                if (dose, 'apop') in self.expTable.keys():
                    # Observed error values for apop
                    apop_obs = (eap + deadapop) * apop_conv + gos - self.expTable[(dose, 'apop')]

                    pm.Normal('dataFita '+str(dose), sd = T.std(apop_obs), observed = apop_obs)
                if (dose, 'dna') in self.expTable.keys():
                    # Observed error values for dna
                    dna_obs = (deadapop + deadnec) * dna_conv + ros - self.expTable[(dose, 'dna')]

                    pm.Normal('dataFitd '+str(dose), sd = T.std(dna_obs), observed = dna_obs)
                if (dose, 'overlap') in self.expTable.keys():
                    # Observed error values for overlap
                    ovlap_obs = deadapop * dna_conv + oos - self.expTable[(dose, 'overlap')]

                    pm.Normal('dataFito '+str(dose), sd = T.std(ovlap_obs), observed = ovlap_obs)
            logp = pm.Deterministic('logp', growth_model.logpt)

        return growth_model

    # Directly import one column of data
    def importData(self, firstCols, drug, drop24=False, comb = None):
        self.drug = drug

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

            doses = []
            for col in list(range(firstCols, self.totalCols)):
                # Set the name of the condition we're considering
                condName = data.columns.values[col]

                if comb != None: # For data with combination therapies
                    # Select columns with drug/combination of interest
                    if condName.split(' ')[0] == self.drug or condName.split(' ')[0] == comb or 'Control' in condName:
                        # Represent dose with a tuple of len(2) in each case
                        if 'Control' in condName:
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

                        doses.append(dose)

                        # Add data to expTable
                        self.expTable[(dose,key)] = data.iloc[:, col].as_matrix()

                        # Add conv0
                        if key == 'confl':
                            self.doses.append(dose)
                            self.condNames.append(condName)
                            self.selCols.append(col)
                            selconv0.append(conv0[col-firstCols])

                else: # For data without combinations
                    if condName.split(' ')[0] == self.drug or 'Control' in condName:
                        # Add the name of the condition we're considering
                        try:
                            dose = condName.split(' ')[1]
                        except IndexError:
                            dose = 0

                        doses.append(dose)

                        # Add data to expTable
                        self.expTable[(dose,key)] = data.iloc[:, col].as_matrix()

                        # Add conv0
                        if key == 'confl':
                            self.doses.append(dose)
                            self.condNames.append(condName)
                            self.selCols.append(col)
                            selconv0.append(conv0[col-firstCols])

        # Record averge conv0 for confl prior
        self.conv0 = np.mean(selconv0)

        # Record the number of observations we've made
        self.nobs = len(properties) * len(self.timeV)

        # Build the model
        self.model = self.build_model()

    def __init__(self, loadFile = None):
        # If no filename is given use a default
        if loadFile is None:
            self.loadFile = "101117_H1299"
        else:
            self.loadFile = loadFile
