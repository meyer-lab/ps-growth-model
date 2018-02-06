import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from collections import OrderedDict
from .sampleAnalysis import readModel


def reformatData(dfd, doseidx, params, dTypes=False):
    """
    Sample nsamples number of points from sampling results,
    Reformat dataframe so that columns of the dataframe are params
    Returns dataframe of columns: parameters and dose
    """
    # Initialize a dataframe
    dfplot = pd.DataFrame()
    # Randomly sample 1000 rows from pymc sampling results
    if dfd.shape[0] > 1000:
        dfd = dfd.sample(1000)

    # Interate over each dose
    # Columns: div, d, deathRate, apopfrac, dose
    for dose in doseidx:
        dftemp = pd.DataFrame()
        for param in params:
            dftemp[param] = dfd[param+'__'+str(doseidx[dose])]
        dftemp['dose'] = dose
        if dTypes:
            dftemp['Data Type'] = dfd['Data Type']
        dfplot = pd.concat([dfplot, dftemp], axis=0)

    # Log transformation
    for param in ['div', 'deathRate']:
        dfplot[param] = dfplot[param].apply(np.log10)
    return dfplot


def violinplot(filename, drugs=None):
    '''
    Takes in a list of drugs
    Makes 1*len(parameters) violinplots for each drug
    '''
    import seaborn as sns
    sns.set_context("paper", font_scale=1.2)
    # Read in dataframe
    classM, df = readModel(filename)
    alldrugs = classM.drugs
    alldoses = classM.doses
    # Get a list of drugs
    if drugs == None:
        drugs = list(sorted(set(classM.drugs)))
        drugs.remove('Control')

    params = ['div', 'deathRate', 'apopfrac']
    
    dfdict = {}

    # Interate over each drug
    for drug in drugs:
        # Set up ordered dictionary for dose:idx
        doseidx = OrderedDict()
        flag = True
        # Iterate from the last condition to the first condition
        for i in range(len(alldrugs)-1, -1, -1):
            # Condition matches drug of interest
            if alldrugs[i] == drug:
                doseidx[alldoses[i]] = i
            # Include the first control after drug conditions
            elif alldrugs[i] == 'Control' and flag and bool(doseidx):
                doseidx[alldoses[i]] = i
                flag = False
        # Put dictionary items in order of increasing dosage
        doseidx = OrderedDict(reversed(list(doseidx.items())))

        # Reshape table for violinplot
        # Columns: div, deathRate, apopfrac, dose
        dfplot = reformatData(df, doseidx, params)
        
        dfdict[drug] = dfplot
    return (dfdict, drugs, params)


def violinplot_split(filename, drugs=None):
    """
    Make split violin plots for comparison of sampling distributions from 
    analyses of kinetic data and endpoint data.
    """
    import seaborn as sns
    # Read in model and kinetic dataframe
    classM, df = readModel(filename)
    
    # Append a Data Type variable to dataframe
    df['Data Type'] = 'Kinetic'
    # Read in dataframe for endpoint data
    _, df2 = readModel(filename+'_ends')
    df2['Data Type'] = 'Endpoints'
    # Concatinate the two data frames
    df = pd.concat([df, df2], axis=0)
    
    # Get variables from model
    alldrugs = classM.drugs
    alldoses = classM.doses
    # Get a list of drugs
    if drugs == None:
        drugs = list(sorted(set(classM.drugs)))
        drugs.remove('Control')

    params = ['div', 'deathRate', 'apopfrac']

    # Set up a len(drugs)*len(params) grid of subplots
    _, axis = plt.subplots(len(drugs), len(params), figsize=(12, 3*len(drugs)), sharex=False, sharey='col')

    # Interate over each drug
    for drug in drugs:
        # Set up ordered dictionary for dose:idx
        doseidx = OrderedDict()
        flag = True
        # Iterate from the last condition to the first condition
        for i in range(len(alldrugs)-1, -1, -1):
            # Condition matches drug of interest
            if alldrugs[i] == drug:
                doseidx[alldoses[i]] = i
            # Include the first control after drug conditions
            elif alldrugs[i] == 'Control' and flag and bool(doseidx):
                doseidx[alldoses[i]] = i
                flag = False
        # Put dictionary items in order of increasing dosage
        doseidx = OrderedDict(reversed(list(doseidx.items())))

        # Reshape table for violinplot
        # Columns: div, deathRate, apopfrac, dose
        dfplot = reformatData(df, doseidx, params, dTypes=True)

        # Plot params vs. drug dose
        # Get drug index
        j = drugs.index(drug)
        # Iterate over each parameter in params
        for i in range(len(params)):
            # For apopfrac, set y-axis limit to [0,1]
            if params[i] == 'apopfrac':
                axis[j, i].set_ylim([0, 1])
            # Make violin plots
            sns.violinplot(x="dose", y=params[i], hue="Data Type",
                                data=dfplot, palette="muted", split=True, ax=axis[j, i], cut=0)
            axis[j, i].set_xlabel(drug+' dose')
            axis[j, i].legend_.remove()

    plt.tight_layout()

    return axis


def hist_plot(drug):
    """
    Display histograms of parameter values across conditions
    *Note: Not up-to-date
    """
    import seaborn as sns
    # Read in dataframe
    classdict, df = readModel([drug])

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
                 diag_kws=dict(shade=True), size=2)

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


def PCA_plot(drugs):
    """
    Principle components analysis of sampling results for parameter values
    *Note: Not up-to-date
    """
    from sklearn.decomposition import PCA
    import seaborn as sns
    from matplotlib.patches import Patch

    # Read in dataframe
    conditions = drugs[:]
    classdict, df = readModel(conditions)

    params = ['div', 'd', 'deathRate', 'apopfrac']

    # Interate over each drug
    dfplot = pd.DataFrame()
    for drug in drugs:
        # Set up table for the drug
        dfd = df[drug]

        # Set up list of doses
        classM = classdict[drug]
        doses = classM.doses

        dftemp = reformatData(dfd, drug, doses, params, nsamples=100)
        dfplot = pd.concat([dfplot, dftemp], axis=0)

    # Keep columns in params
    dfmain = dfplot.loc[:, params]

    # Run PCA
    pca = PCA(n_components=3)
    pca.fit(dfmain)
    # Get explained variance ratio
    expvar = pca.explained_variance_ratio_
    # Get PCA Scores
    dftran = pca.fit_transform(dfmain)
    dftran = pd.DataFrame(dftran, columns=['PC 1', 'PC 2', 'PC 3'])
    # Add condition column to PCA scores
    alldrugs = np.asarray(dfplot.loc[:, 'drug'])
    dftran['drug'] = alldrugs
    alldoses = np.asarray(dfplot.loc[:, 'dose'])
    dftran['dose'] = alldoses

    # Plot first 2 principle components
    plt.figure(figsize=[6, 6])
    markers = ['o', '^', '+', '_', 'x', '*']
    colors = ['b', 'g', 'r', 'm', 'y', 'c']
    for drug in drugs:
        dfp = dftran.loc[dftran['drug'] == drug]
        classM = classdict[drug]
        doses = classM.doses
        for dose in doses:
            i = doses.index(dose)
            dfdose = dfp.loc[dfp['dose'] == dose]
            plt.scatter(dfdose['PC 1'], dfdose['PC 2'], c=colors[drugs.index(drug)],
                        marker=markers[i], s=10)
#    ax = sns.lmplot('PC 1', 'PC 2', data = dftran, hue = 'drug', markers = ['o', '.'],
#                    fit_reg = False, scatter_kws={"s": 10})
    # Set axis labels
    patches = [Patch(color=colors[j], label=drugs[j]) for j in range(len(drugs))]
    plt.legend(handles=patches, bbox_to_anchor=(0, 1.2))
    plt.xlabel('PC 1 ('+str(round(float(expvar[0])*100, 0))[:-2]+'%)', fontsize=20)
    plt.ylabel('PC 2 ('+str(round(float(expvar[1])*100, 0))[:-2]+'%)', fontsize=20)
    plt.title('PCA', fontsize=20)
    plt.show()


def dose_response_plot(drugs=None, log=True, logdose=False, show=True):
    '''
    Takes in a list of drugs
    Makes 1*num(parameters) plots for each drug
    *Note: Not up-to-date
    ''' 
    # Read in dataframe
    if drugs is None:
        classdict, df = readModel()
        drugs = list(classdict.keys())
    else:
        conditions = drugs[:]
        classdict, df = readModel(conditions)

    params = ['div ', 'd ', 'deathRate ', 'apopfrac ']

    # Make plots for each drug
    f, axis = plt.subplots(len(drugs), 4, figsize=(12, 2.5*len(drugs)), sharex=False, sharey='col')

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
            axis[j, i].errorbar(doses, dfmean[params[i]],
                                [dferr1[params[i]], dferr2[params[i]]],
                                fmt='.', capsize=5, capthick=1)
            axis[j, i].set_xlabel(drug+'-dose')
            axis[j, i].set_ylabel(params[i])

    plt.tight_layout()
    plt.title('Dose-response Plot (Drugs: '+str(drugs)[1:-1]+')', x=-3, y=5.1)
    if show:
        plt.show()
    else:
        return (f, axis)
