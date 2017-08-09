import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from .sampleAnalysis import readCols

try:
    import cPickle as pickle
except ImportError:
    import pickle

def makePlot(cols, drugs):
    hist_plot(cols)
    PCA(cols)
    dose_response_plot(drugs)
    violinplot(drugs)

def hist_plot(cols):
    """
    Display histograms of parameter values across conditions
    """
    import seaborn as sns
    # Read in dataset to Pandas data frame
    df = readCols(cols)[1]

    print(df.columns)

    # Get conditions
    cond = df.loc[:,'Condition']
    cond = list(cond.drop_duplicates(keep='first'))
    condidx = dict(zip(cond, sns.color_palette()))

    # Log transformation
    params = ['div', 'd', 'deathRate', 'apopfrac', 'confl_conv', 'std']
    logparams = ['div', 'd', 'deathRate', 'confl_conv', 'std']
    for param in logparams:
        df[param] = np.log10(df[param])

    # Main plot organization
    sns.pairplot(df, diag_kind="kde", hue='Condition', vars=params,
                 plot_kws=dict(s=5, linewidth=0),
                 diag_kws=dict(shade=True), size = 2)

    # Shuffle positions to show legend
    patches = list()
    for key, val in zip(cond, [condidx[con] for con in cond]):
        patches.append(matplotlib.patches.Patch(color=val, label=key))
    plt.legend(handles=patches, bbox_to_anchor=(0, 6.5), loc=2)

    # Draw plot
    plt.show()


def PCA(cols):
    """
    Principle components analysis of sampling results for parameter values
    """
    from sklearn.decomposition import PCA
    import seaborn as sns

    df = readCols(cols)[1]
    print(df.columns)

    # Log transformation
    params = ['div', 'd', 'deathRate', 'apopfrac', 'confl_conv', 'std']
    for param in params:
        df[param] = np.log10(df[param])

    # Keep columns in params
    dfmain = df.loc[:,params]

    # Run PCA
    pca = PCA(n_components=3)
    pca.fit(dfmain)
    # Get explained variance ratio
    expvar = pca.explained_variance_ratio_
    # Get PCA Scores
    dftran = pca.fit_transform(dfmain)
    dftran = pd.DataFrame(dftran, columns = ['PC 1', 'PC 2', 'PC 3'])
    # Add condition column to PCA scores
    condition = np.asarray(df.loc[:,'Condition'])
    dftran['Conditions'] = condition 

    # Plot first 2 principle components
    ax = sns.lmplot('PC 1', 'PC 2', data = dftran, hue = 'Conditions', fit_reg = False, scatter_kws={"s": 10})
    # Set axis labels
    ax.set_xlabels('PC 1 ('+str(round(float(expvar[0])*100, 0))[:-2]+'%)')
    ax.set_ylabels('PC 2 ('+str(round(float(expvar[1])*100, 0))[:-2]+'%)')
    plt.title('PCA')
    plt.show()


def dose_response_plot(drugs, log=True, columns = None, logdose = False, show = True):
    '''
    Takes in a list of drugs
    Makes 1*num(parameters) plots for each drug
    ''' 
    # Read in dataframe
    if columns == None: 
        df = readCols(list(range(2,19)))[1]
    else:
        df = readCols(columns)[1]

    params = ['div', 'd', 'deathRate', 'apopfrac', 'confl_conv', 'std']
    
    # Make plots for each drug
    f, axis = plt.subplots(len(drugs),6,figsize=(15,2.5*len(drugs)), sharex=False, sharey='col')

    # Get control parameter fits
    dfc = df.loc[df['Condition'].str.contains('Control')]

    # Interate over each drug
    for drug in drugs:
        # Set up table for the drug
        dfd = df[df['Condition'].str.contains(drug+' ')]
        # Break if drug not in dataset
        if dfd.empty:
            print("Error: Drug not in dataset")
            break

        # Add dose to table
        dfd = dfd.copy()
        dfd[drug+'-dose'] = dfd.loc[:, 'Condition'].str.split(' ').str[1]
        dfd.loc[:, drug+'-dose'] = pd.to_numeric(dfd[drug+'-dose'])
        # Add control
        if logdose == False:
            dfcon = dfc.copy()
            dfcon[drug+'-dose'] = 0
            dfd = pd.concat([dfd, dfcon])
        
        # Set up mean and confidence interval
        if log: 
            logparams = ['div', 'd', 'deathRate', 'confl_conv', 'std']
            for param in logparams:
                dfd.loc[:, param] = dfd[param].apply(np.log10)
        if logdose:
            dfd.loc[:, drug+'-dose'] = dfd[drug+'-dose'].apply(np.log10)
        dfmean = dfd.groupby([drug+'-dose'])[params].mean().reset_index()
        dferr1 = dfmean-dfd.groupby([drug+'-dose'])[params].quantile(0.05).reset_index()
        dferr2 = dfd.groupby([drug+'-dose'])[params].quantile(0.95).reset_index()-dfmean

        # Plot params vs. drug dose
        j = drugs.index(drug)
        for i in range(len(params)):
            axis[j,i].errorbar(dfmean[drug+'-dose'],dfmean[params[i]],
                               [dferr1[params[i]],dferr2[params[i]]],
                               fmt='.',capsize=5,capthick=1)
            axis[j,i].set_xlabel(drug+'-dose')
            axis[j,i].set_ylabel(params[i])

    plt.tight_layout()
    plt.title('Dose-response Plot (Drugs: '+str(drugs)[1:-1]+')', x = -5, y = 4.9)
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
    df = readCols(list(range(2,19)))[1]

    params = ['div', 'd', 'deathRate', 'apopfrac', 'confl_conv', 'std']
    logparams = ['div', 'd', 'deathRate', 'confl_conv', 'std']
    
    # Make plots for each drug
    f, axis = plt.subplots(len(drugs),6,figsize=(18,3*len(drugs)), sharex=False, sharey='col')
    # Get control parameter fits
    dfc = df.loc[df['Condition'] == 'Control']
    dfc = dfc.copy()
    # Iterate over each drug 
    for drug in drugs:
        # Set up table for the drug
        dfd = df[df['Condition'].str.contains(drug+' ')]
        # Break if drug not in dataset
        if dfd.empty:
            print("Error: Drug not in dataset")
            break 

        # Add dose to table
        dfd = dfd.copy()
        dfd[drug+'-dose'] = dfd.loc[:, 'Condition'].str.split(' ').str[1]
        dfd.loc[:, drug+'-dose'] = pd.to_numeric(dfd[drug+'-dose'])
        # Add control
        dfc[drug+'-dose'] = 0
        dfd = pd.concat([dfd, dfc])

        # Plot params vs. drug dose
        j = drugs.index(drug)
        for i in range(len(params)):
            if log and params[i] in logparams:
                sns.violinplot(dfd[drug+'-dose'],np.log10(dfd[params[i]]),ax=axis[j,i])
            else:
                sns.violinplot(dfd[drug+'-dose'],dfd[params[i]],ax=axis[j,i])

    plt.tight_layout()
    plt.title('Violinplot (Drugs: '+str(drugs)[1:-1]+')', x = -5, y = 4.9)
    plt.show()

def plot_dose_fits(columns, drugs, params, dic, dist = False):
    df = readCols(columns)[1]
    
    fig, axis = plt.subplots(len(drugs),len(params),figsize=(3*len(params),3*len(drugs)), sharex= 'row', sharey='col')
    for drug in drugs:
        # Set up table for the drug
        dfd = df[df['Condition'].str.contains(drug+' ')]
        # Break if drug not in dataset
        if dfd.empty:
            print("Error: Drug not in dataset")
            break

        # Add dose to table
        dfd = dfd.copy()
        dfd[drug+'-dose'] = dfd.loc[:, 'Condition'].str.split(' ').str[1]
        dfd.loc[:, drug+'-dose'] = pd.to_numeric(dfd[drug+'-dose'])
        # log10 transform drug dosage
        dfd.loc[:, drug+'-dose'] = dfd[drug+'-dose'].apply(np.log10)
        doses = list(dfd.loc[:,drug+'-dose'].drop_duplicates(keep='first'))
        mindose = min(doses)
        maxdose = max(doses)

        # log10 transform parameters besides 'apopfrac'
        for param in params:
            if param != 'apopfrac':
                dfd.loc[:, param] = dfd[param].apply(np.log10)

        # Set up mean and confidence interval
        dfmean = dfd.groupby([drug+'-dose'])[params].mean().reset_index()
        dferr1 = dfmean-dfd.groupby([drug+'-dose'])[params].quantile(0.05).reset_index()
        dferr2 = dfd.groupby([drug+'-dose'])[params].quantile(0.95).reset_index()-dfmean

        # Plot params vs. drug dose
        j = drugs.index(drug)
        for i in range(len(params)):
            # Plot dose response
            axis[j,i].errorbar(dfmean[drug+'-dose'],dfmean[params[i]],
                               [dferr1[params[i]],dferr2[params[i]]],
                               fmt='.',capsize=5,capthick=1)
            # Plot dose response curves
            if not dist: # plot MAP curve
                paramfits = dic[str(drug)+'-'+str(params[i])]
                bottom = np.exp(paramfits['bottom_log__'])
                top = np.exp(paramfits['top_log__'])
                logIC50 = paramfits['logIC50']
                hillslope = paramfits['hillslope']
                doserange = np.arange(mindose, maxdose, (maxdose - mindose)/100)
                paramfit = []
                for x in doserange:
                    y = bottom + (top - bottom) / (1 + np.power(10., (logIC50 - x)*hillslope))
                    paramfit.append(y)
                axis[j,i].plot(doserange, paramfit)

            axis[j,i].set_xlabel(drug+'-dose')
            axis[j,i].set_ylabel(params[i])

    plt.tight_layout()
    plt.title('Dose-response Curves (Drugs: '+str(drugs)[1:-1]+')', x = -5, y = 4.9)
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
