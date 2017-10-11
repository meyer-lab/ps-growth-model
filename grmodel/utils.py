import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from .sampleAnalysis import readModels

try:
    import cPickle as pickle
except ImportError:
    import pickle

def makePlot(cols, drugs):
    hist_plot(cols)
    PCA(cols)
    dose_response_plot(drugs)
    violinplot(drugs)

def reformatData(dfd, drug, doses, params, log = True, nsamples = 1000):
    """
    Sample nsamples number of points from sampling results,
    Reformat dataframe so that columns of the dataframe are params
    """
    dfplot = pd.DataFrame()

    if dfd.shape[0] > nsamples:
        dfd = dfd.sample(nsamples)

    # Interate over each dose
    # Columns: div, d, deathRate, apopfrac, condition
    for dose in doses:
        dftemp = pd.DataFrame()
        for param in params:
            dftemp[param] = dfd[param+' '+str(dose)]
        dftemp['drug'] = drug 
        dftemp['dose'] = dose
        dfplot = pd.concat([dfplot, dftemp], axis = 0)

    # Log transformation
    logparams = ['div', 'd', 'deathRate']
    for param in logparams:
        dfplot[param] = dfplot[param].apply(np.log10)
    return dfplot

def hist_plot(drug):
    """
    Display histograms of parameter values across conditions
    """
    import seaborn as sns
    # Read in dataframe
    classdict, df = readModels([drug])

    params = ['div', 'd', 'deathRate', 'apopfrac']

    # Set up table for the drug
    dfd = df[drug]
    # Set up list of doses
    classM = classdict[drug]
    doses = classM.doses

    dfplot = reformatData(dfd, drug, doses, params)
    
    dfplot['Condition'] = dfplot.apply(lambda row: row.drug + ' ' + str(row.dose), axis=1)

    #Set context for seaborn
    sns.set_context("paper", font_scale=1.4)

    # Main plot organization
    sns.pairplot(dfplot, diag_kind="kde", hue='Condition', vars=params,
                 plot_kws=dict(s=5, linewidth=0),
                 diag_kws=dict(shade=True), size = 2)

    # Get conditions
    cond = dfplot.Condition.unique()
    condidx = dict(zip(cond, sns.color_palette()))
    # Make legend
    patches = list()
    for key, val in zip(cond, [condidx[con] for con in cond]):
        patches.append(matplotlib.patches.Patch(color=val, label=key))
    # Show legend
    plt.legend(handles=patches, bbox_to_anchor=(-3, 4.5), loc=2)

    # Draw plot
    plt.show()


def PCA(drugs):
    """
    Principle components analysis of sampling results for parameter values
    """
    from sklearn.decomposition import PCA
    import seaborn as sns
    from matplotlib.patches import Patch

    # Read in dataframe
    conditions = drugs[:]
    classdict, df = readModels(conditions)

    params = ['div', 'd', 'deathRate', 'apopfrac']

    # Interate over each drug
    dfplot = pd.DataFrame()
    for drug in drugs:
        # Set up table for the drug
        dfd = df[drug]

        # Set up list of doses
        classM = classdict[drug]
        doses = classM.doses

        dftemp = reformatData(dfd, drug, doses, params, nsamples = 100)
        dfplot = pd.concat([dfplot, dftemp], axis = 0)

    # Keep columns in params
    dfmain = dfplot.loc[:,params]

    # Run PCA
    pca = PCA(n_components=3)
    pca.fit(dfmain)
    # Get explained variance ratio
    expvar = pca.explained_variance_ratio_
    # Get PCA Scores
    dftran = pca.fit_transform(dfmain)
    dftran = pd.DataFrame(dftran, columns = ['PC 1', 'PC 2', 'PC 3'])
    # Add condition column to PCA scores
    alldrugs = np.asarray(dfplot.loc[:,'drug'])
    dftran['drug'] = alldrugs
    alldoses = np.asarray(dfplot.loc[:,'dose'])
    dftran['dose'] = alldoses

    # Plot first 2 principle components
    plt.figure(figsize=[6,6])
    markers = ['o','^', '+', '_', 'x', '*']
    colors = ['b', 'g', 'r', 'm', 'y', 'c']
    for drug in drugs:
        dfp = dftran.loc[dftran['drug'] == drug]
        classM = classdict[drug]
        doses = classM.doses
        for dose in doses:
            i = doses.index(dose)
            dfdose = dfp.loc[dfp['dose'] == dose]
            plt.scatter(dfdose['PC 1'], dfdose['PC 2'], c = colors[drugs.index(drug)], marker = markers[i], s=10)
    #ax = sns.lmplot('PC 1', 'PC 2', data = dftran, hue = 'drug', markers = ['o', '.'], fit_reg = False, scatter_kws={"s": 10})
    # Set axis labels
    patches = [Patch(color=colors[j], label=drugs[j]) for j in range(len(drugs))]
    plt.legend(handles = patches, bbox_to_anchor=(0, 1.2))
    plt.xlabel('PC 1 ('+str(round(float(expvar[0])*100, 0))[:-2]+'%)', fontsize = 20)
    plt.ylabel('PC 2 ('+str(round(float(expvar[1])*100, 0))[:-2]+'%)', fontsize = 20)
    plt.title('PCA', fontsize = 20)
    plt.show()


def dose_response_plot(drugs = None, log=True, logdose = False, show = True):
    '''
    Takes in a list of drugs
    Makes 1*num(parameters) plots for each drug
    ''' 
    # Read in dataframe
    if drugs == None: 
        classdict, df = readModels()
        drugs = list(classdict.keys())
    else:
        conditions = drugs[:]
        classdict, df = readModels(conditions)

    params = ['div ', 'd ', 'deathRate ', 'apopfrac ']

    # Make plots for each drug
    f, axis = plt.subplots(len(drugs),4,figsize=(12,2.5*len(drugs)), sharex=False, sharey='col')

    # Interate over each drug
    for drug in drugs:
        # Set up table for the drug
        dfd = df[drug]
        if dfd.shape[0] > 1000:
            dfd = dfd.sample(1000)

        # Set up list of doses
        classM = classdict[drug]
        doses = classM.doses
        if logdose:
            doses.remove(0.0)

        # Break if drug not in dataset
        if dfd.empty:
            print("Error: Drug not in dataset")
            break

        dfj = pd.DataFrame()
        for param in params:
            for dose in doses: 
                dfj = pd.concat([dfj, dfd[param+str(dose)]], axis=1)

        # Convert sampling values for specific parameters to logspace
        if log:
            logparams = ['div ', 'd ', 'deathRate ']
            for param in logparams:
                dfj.loc[:, dfj.columns.to_series().str.contains(param).tolist()] = dfj[dfj.columns[dfj.columns.to_series().str.contains(param)]].apply(np.log10)

        # Set up mean and confidence interval
        dfmean = dfj.mean(axis=0)
        dferr1 = dfmean-dfj.quantile(0.05, axis=0)
        dferr2 = dfj.quantile(0.95, axis=0)-dfmean
        
        # Set up table for plots
        dfplots = []
        for dftemp in [dfmean, dferr1, dferr2]:
            dfplot = pd.DataFrame()
            dfplot['dose'] = doses
            for param in params:
                temp = []
                for dose in doses:
                    temp.append(dftemp[param+str(dose)])
                dfplot[param] = temp
            dfplots.append(dfplot)
        dfmean = dfplots[0]
        dferr1 = dfplots[1]
        dferr2 = dfplots[2]

        # Plot params vs. drug dose
        j = drugs.index(drug)
        if logdose:
            doses = np.log10(doses)
        for i in range(len(params)):
            axis[j,i].errorbar(doses,dfmean[params[i]],
                               [dferr1[params[i]],dferr2[params[i]]],
                               fmt='.',capsize=5,capthick=1)
            axis[j,i].set_xlabel(drug+'-dose')
            axis[j,i].set_ylabel(params[i])

    plt.tight_layout()
    plt.title('Dose-response Plot (Drugs: '+str(drugs)[1:-1]+')', x = -3, y = 5.1)
    if show:
        plt.show()
    else:
        return (f, axis) 


def violinplot(drugs,log=True):
    '''
    Takes in a list of drugs
    Makes 1*num(parameters) boxplots for each drug
    '''
    import seaborn as sns
    sns.set_context("paper", font_scale=1.4)
    # Read in dataframe
    conditions = drugs[:]
    classdict, df = readModels(conditions)

    params = ['div', 'd', 'deathRate', 'apopfrac']
    
    # Make plots for each drug
    f, axis = plt.subplots(len(drugs),4,figsize=(12,2.5*len(drugs)), sharex=False, sharey='col')


    # Interate over each drug
    for drug in drugs:
        # Set up table for the drug
        dfd = df[drug]

        # Set up list of doses
        classM = classdict[drug]
        doses = classM.doses

        # Reshape table for violinplot
        # Columns: div, d, deathRate, apopfrac, dose
        dfplot = reformatData(dfd, drug, doses, params)

        # Plot params vs. drug dose
        j = drugs.index(drug)
        for i in range(len(params)):
            if params[i] == 'apopfrac':
                axis[j,i].set_ylim([0,1])
            sns.violinplot(dfplot['dose'],dfplot[params[i]],ax=axis[j,i],cut=0)

    plt.tight_layout()
    plt.title('Violinplot (Drugs: '+str(drugs)[1:-1]+')', x = -3, y = 5.1)
    plt.show()


def plotSimulation(self, paramV):
    """
    Plots the results from a simulation.
    TODO: Run simulation when this is called, and also plot observations.
    TODO: If selCol is None, then plot simulation but not observations.
    """

    # Calculate model data table
    params = mcFormat(paramV[:-4])
    t_interval = np.arange(
        0, self.data_confl.iloc[-1, 1], (self.data_confl.iloc[2, 1] - self.data_confl.iloc[1, 1]))

    state = simulate(params, t_interval)

    # plot simulation results; if selCol is not None, also plot observations
    if self.selCol is not None:
        # print(self.selCol)
        data_confl_selCol = self.data_confl.iloc[:, self.selCol]
        data_green_selCol = self.data_green.iloc[:, self.selCol]
        t_interval_observ = self.data_confl.iloc[:, 1]

        # get conversion constants
        conv_confl, conv_green = np.power(10, paramV[-4:-2])

        # adjust simulation values
        simulation_confl = state.iloc[:, 1] * conv_confl
        simulation_green = (state.iloc[:, 2] + state.iloc[:, 3]) * conv_green

        f, axarr = plt.subplots(3, figsize=(10, 10))
        axarr[0].set_title('Simulation Results')
        t_interval = state.iloc[:, 0].values
        axarr[0].plot(t_interval, state.iloc[:, 1], 'b-', label="live")
        axarr[0].plot(t_interval, state.iloc[:, 2], 'r-', label="dead")
        axarr[0].plot(t_interval, state.iloc[:, 3],
                      'g-', label="early apoptosis")
        axarr[0].plot(t_interval, state.iloc[:, 4], 'k-', label="gone")
        axarr[0].legend(bbox_to_anchor=(1.04, 0.5),
                        loc="center left", borderaxespad=0)

        axarr[1].set_title('Observed: data_confl')
        axarr[1].plot(t_interval_observ, data_confl_selCol, label='data_confl')
        axarr[1].plot(t_interval, simulation_confl, label='simulation_confl')
        axarr[1].legend(bbox_to_anchor=(1.04, 0.5),
                        loc="center left", borderaxespad=0)

        axarr[2].set_title('Observed: data_green')
        axarr[2].plot(t_interval_observ, data_green_selCol, label='data_green')
        axarr[2].plot(t_interval, simulation_green, label='simulation_green')
        axarr[2].legend(bbox_to_anchor=(1.04, 0.5),
                        loc="center left", borderaxespad=0)
        plt.tight_layout()
        plt.show()
    else:
        figure()
        xlabel('Time')
        ylabel('Number of Cells')
        t_interval = state.iloc[:, 0].values
        plt.plot(t_interval, state.iloc[:, 1], 'b-', label="live")
        plt.plot(t_interval, state.iloc[:, 2], 'r-', label="dead")
        plt.plot(t_interval, state.iloc[:, 3], 'g-', label="early apoptosis")
        plt.plot(t_interval, state.iloc[:, 4], 'k-', label="gone")
        plt.legend(loc='upper right')
        show()
