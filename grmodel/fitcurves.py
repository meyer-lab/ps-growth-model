import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from .sampleAnalysis import readModels
try:
    import cPickle as pickle
except ImportError:
    import pickle

def sigmoid(p,x):
    """ Calculate value of sigmoid function y given equation and x value """
    logIC50,bottom,top,hillslope = p
    y = bottom + (top - bottom) / (1 + np.power(10., (logIC50-x)*hillslope))
    return y

def residuals(p,x,y):
    '''Return residue of sigmoid curve'''
    return y - sigmoid(p,x)

def prepdata(drugs, params, log = False):
    """ 
    Read in MCMC sampling results
    Output dfdict: a dictionary, with keys (drug, param), of pandas dataframes
    with shapes (len(samples), len(dose))
    """
    # Read in dataframe
    conditions = drugs[:]
    classdict, df = readModels(conditions)
    # Initiate dictionary
    dfdict = {}

    # Interate over each drug
    for drug in drugs:
        dfd = df[drug]
        if dfd.shape[0]>1000:
            dfd = dfd.sample(1000)
        # Break if drug not in dataset
        if dfd.empty:
            print("Error: Drug not in dataset")
            break

        # Set up list of doses
        classM = classdict[drug]
        doses = classM.doses
        doses.remove(0.0)
        
        # Make one pandas table for each parameter
        for param in params:
            dftemp = pd.DataFrame()
            # Append one column per dose, in order of increasing dosage
            for dose in doses:
                dftemp = pd.concat([dftemp, dfd[param+' '+str(dose)]], axis=1)
            dftemp.columns = np.log10(doses)
            # Log10 transformation
            if log and param in ['div', 'deathRate', 'd']:
                dftemp = dftemp.apply(np.log10)
            # Save table in dictionary
            dfdict[(drug,param)] = dftemp

    return dfdict

def fitcurves(dfdict, drugs, params):
    """
    Fit dose-response curves to each parameter in params, for each drug in drugs.
    Return curveparamsdict: a dictionary, with keys (drug, param), of curve
    fit parameters. Each set of curve fit parameters corresponds to one sample.
    """
    curveparamsdict = {}
    # Iterate over dfdict.keys
    for key in dfdict.keys():
        table = dfdict[key]
        data = []
        # doses
        x = table.columns.tolist()
        # Set limits on parameter values
        mins = np.array([min(x),min(table.iloc[:,0]), min(table.iloc[:,-1]),0.5])
        maxs = np.array([max(x),max(table.iloc[:,0]), max(table.iloc[:,-1]),5])
        
        # Fit sigmoid to each sample in dfdict[key]
        for row in table.iterrows():
            y = np.array(row[1].as_matrix())
            p_guess= (np.mean(x), y[0], y[-1], 1)
            res_1 = sp.optimize.least_squares(residuals,p_guess, bounds=(mins,maxs), args=(x,y))
            # Save solution
            data.append(list(res_1.x))
        # Save curve fits
        df = pd.DataFrame(data, columns = ['logIC50', 'bottom', 'top', 'hillslope'])
        curveparamsdict[key] = df
    return curveparamsdict

def plotcurves(drugs, params, log = False):
    """ Plot dose-response curves overlayed with sampling results"""
    # Get dictionary of curve fits
    dfdict = prepdata(drugs, params, log = log)
    curveparamsdict = fitcurves(dfdict, drugs, params)

    # Initialize subplots
    f, axis = plt.subplots(len(drugs), len(params),figsize=(10,2.5*len(drugs)), sharex=False, sharey='col')
    # Iterate over each parameter in each drug
    for drug in drugs:
        for param in params:
            # Plot curve distribution
            # Set up variables
            df = curveparamsdict[(drug, param)]
            table = dfdict[(drug, param)]
            doses = table.columns.tolist()
            doserange = np.arange(min(doses), max(doses), (max(doses) - min(doses))/200)
            calcset = np.full((df.shape[0], len(doserange)), np.inf)
            
            # Get values for IogIC50 and hill-slope
            logIC50 = round(np.mean(df['logIC50']), 2)
            hillslope = round(np.mean(df['hillslope']), 2)
            varr = 0
            # Iterate over each curve
            for row in df.iterrows():
                # Calculate sigmoid curve value across doserange
                curvefit = sigmoid(list(row[1].as_matrix()), doserange)
                calcset[varr, :] = curvefit
                varr += 1
            # Get median & 90% confidence interval for each time point
            qqq = np.percentile(calcset, [5, 25, 50, 75, 95], axis=0)

            i = drugs.index(drug)
            j = params.index(param)
            # Plot curve distribution
            axis[i, j].plot(doserange, qqq[2, :], color = 'b')
            axis[i, j].fill_between(doserange, qqq[1, :], qqq[3, :], alpha=0.5)
            axis[i, j].fill_between(doserange, qqq[0, :], qqq[4, :], alpha=0.2)

            # Plot 90% CI of MCMC sampling results
            # Set up mean and confidence interval
            dfmean = table.mean(axis=0)
            dferr1 = dfmean-table.quantile(0.05, axis=0)
            dferr2 = table.quantile(0.95,axis=0)-dfmean

            # Plot 90% CI
            axis[i, j].errorbar(doses, dfmean, [dferr1,dferr2],
                               fmt='o',ms=5,capsize=5,capthick=1)
            axis[i, j].set_title(drug+', logIC50 = '+str(logIC50)+', hillslope = '+str(hillslope))
            axis[i, j].set_xlabel(drug+' logdose')
            axis[i, j].set_ylabel(param)
    plt.tight_layout()