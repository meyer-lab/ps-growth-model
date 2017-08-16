import bz2
import os
import pymc3 as pm
import numpy as np
import scipy as sp
import pandas as pd
from .sampleAnalysis import readCols
try:
    import cPickle as pickle
except ImportError:
    import pickle


class MultiDrugs:
    def __init__(self, columns, drugs, fitparams):
        self.columns = columns
        self.drugs = drugs
        self.fitparams = fitparams

    def get_tables(self):
        """ Get tables from base model sampling results """
        # Read columns and get columns for params
        df = readCols(self.columns)[1]
        params = self.fitparams[:]
        params.append('Column')
        params.append('Condition')
        df = df.loc[:, params]
        # Set up table for each drug
        drugdict = dict()
        for drug in self.drugs:
            dfd = df[df['Condition'].str.contains(drug+' ')]
            # Break if drug not in dataset
            if dfd.empty:
                print("Error: Drug not in dataset")
                break

            # Add dose to table
            dfd = dfd.copy()
            dfd[drug+'-dose'] = dfd.loc[:, 'Condition'].str.split(' ').str[1]
            dfd.loc[:, drug+'-dose'] = pd.to_numeric(dfd[drug+'-dose'])
            dfd[drug+'-logdose'] = np.log10(dfd.loc[:, drug+'-dose'])
            
            # Convert self.fitparams to ln space
            for param in self.fitparams:
                dfd.loc[:, param] = dfd[param].apply(np.log)
            drugdict[drug] = dfd
        return drugdict

    def fitCurves(self):
        """ For each drug, fit dose response curves to each param in self.fitparams """
        # Get tables and initialize dictionary
        drugdict = self.get_tables()
        curvefits = dict()
        # Iiterate over each drug
        for key in drugdict:
            # Get table
            val = drugdict[key]
            # Fit curve to each param
            for param in self.fitparams:
                fit = CurveFit(key, val, param)
                fitmap = fit.sampling()
                curvefits[str(key)+'-'+str(param)] = fitmap
        self.curvefits = curvefits
        return curvefits

    def saveCurves(self, filePrefix):
        """ Saves curve fits """
        # Remove file if already exists 
        if os.path.exists(filePrefix + '_curves.pkl'):
            os.remove(filePrefix + '_curves.pkl')
        # Save sampling data
        pickle.dump(self.curvefits, bz2.BZ2File(filePrefix + '_curves.pkl', 'w'))


class CurveFit:
    def __init__(self, drug, drugdf, fitparam): 
        self.drug = drug
        self.drugdf = drugdf
        self.fitparam = fitparam

    def get_dicts(self):
        """ 
        Returns a dictionary for sampling data at each dose, 
        and a dictionary of kdes for sampling distribution at each dose
        """
        # Initialize variables
        dosedata = dict()
        kdes = dict()
        drugdf = self.drugdf.copy()
        # Get drug doses
        logdoses = drugdf.loc[:,self.drug+'-logdose']
        self.logdoses = list(logdoses.drop_duplicates(keep='first'))
        # Iterate over each dose
        for logdose in logdoses:
            df = drugdf.loc[drugdf[self.drug+'-logdose'] == logdose]
            kde = sp.stats.gaussian_kde(df.loc[:, self.fitparam])
            dosedata[logdose] = df.loc[:, self.fitparam]
            kdes[logdose] = kde
        return (dosedata, kdes)

    def build_model(self):
        """ Builds and returns pymc model """
        curve_model = pm.Model()
        with curve_model:
            # Set up parameters for prior distribution specification
            dosedata, kdes = self.get_dicts()
            mindose = np.amin(self.logdoses)
            maxdose = np.amax(self.logdoses)
            mindoseparam = np.mean(dosedata[mindose])
            maxdoseparam = np.mean(dosedata[maxdose])
            sdparam = np.std(self.drugdf[self.fitparam])/5
            if mindoseparam < maxdoseparam:
                minparam = np.amin(dosedata[mindose])
                maxparam = np.amax(dosedata[maxdose])
                minrange = minparam - 2 * sdparam
                maxrange = maxparam + 2 * sdparam
            else: 
                minparam = np.amax(dosedata[mindose])
                maxparam = np.amin(dosedata[maxdose])
                minrange= maxparam - 2 * sdparam
                maxrange = minparam + 2 * sdparam
            # Specify priors
            bottom = pm.Lognormal('bottom', (mindoseparam+minparam)/2, sdparam)
            top = pm.Lognormal('top', (maxdoseparam+maxparam)/2, sdparam)
            logIC50 = pm.Normal('logIC50', np.mean(self.logdoses), np.std(self.logdoses))
            hillslope = pm.Lognormal('hillslope', 0, 1)
            
            # Set up data input for interpolated distribution
            datarange = np.arange(minrange, maxrange, (maxrange-minrange)/200)
            for logdose in self.logdoses:
                kde = kdes[logdose]
                # Interpolates kde distribution
                y = bottom + (top - bottom) / (1 + np.power(10., (logIC50-logdose)*hillslope))
                pm.Interpolated('dis'+str(logdose), np.exp(datarange), kde.pdf(datarange), observed = y)

        return curve_model

    def sampling(self):
        """ Run pymc sampling """
        self.model = self.build_model()
        get_MAP = pm.find_MAP(model=self.model)
        return get_MAP
#        print(get_MAP)
#        with self.model:
#            self.samples = pm.sample()
#        return self.samples