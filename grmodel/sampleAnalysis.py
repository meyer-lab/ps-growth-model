import bz2
import pickle
from collections import deque
import pymc3 as pm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


from .pymcGrowth import simulate


def read_dataset(ff=None):
    '''
    Read the pymc model from a sampling file
    Makes traceplots if traceplot=True
    '''

    if ff is None:
        ff = "101117_H1299"

    filename = './grmodel/data/' + ff + '_samples.pkl'

    # Read in class
    return pickle.load(bz2.BZ2File(filename, 'rb'))


def readModel(ff=None, trim=False):
    """
    Calls read_dataset to load pymc model
    Outputs: (model, table for the sampling results)
    """
    model = read_dataset(ff)

    model.samples = model.fit.sample(1000)

    df = pm.backends.tracetab.trace_to_dataframe(model.samples)

    # TODO: Get rid of needing samples
    return (model, df)


def getvalues(dic):
    """
    Take dic, a dictionary with lists or item as values
    Return vals, a flattened list of values in the dictionary
    """
    values = list(dic.values())
    vals = []
    for sublist in values:
        try:
            vals.append(float(sublist))
        except TypeError:
            vals.extend(sublist.tolist())
    return vals


def calcset(pdset, idx, time, idic):
    """Calculate model predictions based on parameter fits from sampling data"""
    # Initialize counter
    varr = 0
    # Initialize 3 numpy 2D arrays
    calcset, calcseta, calcsetd = (np.full((pdset.shape[0], len(time)), np.inf) for _ in range(3))
    # Interate over each row of sampling data
    for row in pdset.iterrows():
        # Get parameter values
        mparm = np.copy(np.array(row[1].values[[idic['div__' + idx],
                                                idic['d'],
                                                idic['deathRate__' + idx],
                                                idic['apopfrac__' + idx]]]))
        # Get raw model predictions
        simret = simulate(mparm, time)
        # Apply conversion factors to model predictions
        # Fill-in one row of the numpy arrays
        calcset[varr, :] = (np.sum(simret, axis=1)
                            * row[1].values[idic['confl_conv']])
        calcseta[varr, :] = (np.sum(simret[:, 1:3], axis=1)
                             * row[1].values[idic['apop_conv']]
                             + row[1].values[idic['apop_offset']])
        calcsetd[varr, :] = (np.sum(simret[:, 2:4], axis=1)
                             * row[1].values[idic['dna_conv']]
                             + row[1].values[idic['dna_offset']])
        varr += 1
    return (calcset, calcseta, calcsetd)


def simulation(filename, drug, ax=None, unit='nM'):
    """Make simulation plots of experimental data overlayed with model predictions"""
    # Load model and dataset
    classM, pdset = readModel(ff=filename)
    # A list of all conditions
    alldrugs = classM.drugs
    # A list of all doses, one for each condition
    alldoses = classM.doses
    # Initialize variables
    doses = deque()  # Drug doses
    doseidx = deque()  # Indces of conditions that matches drug of interest or is control
    flag = True

    # Iterate over conditions in reverse order
    for i in range(len(alldrugs) - 1, -1, -1):
        # Condition matches drug of interest
        if alldrugs[i] == drug:
            doses.appendleft(alldoses[i])
            doseidx.appendleft(i)
        # Include the first control after drug conditions
        elif alldrugs[i] == 'Control' and len(doses) != 0 and flag:
            doses.appendleft(alldoses[i])
            doseidx.appendleft(i)
            flag = False

    # Initialize an axis variable of dimension (1,3)
    if not ax:
        _, ax = plt.subplots(1, 3, figsize=(10.5, 3.5), sharex=True, sharey=False)

    # Set up idic, a dictionary of parameter:column index (eg: div_1:3)
    idic = {}
    # Shared parameters
    idic['d'] = pdset.columns.get_loc('d')
    for param in ['confl_conv', 'apop_conv', 'dna_conv', 'apop_offset', 'dna_offset']:
        idic[param] = pdset.columns.get_loc(param)
    # Vectorized parameters, interate over each idx in doseidx
    for param in ['div', 'deathRate', 'apopfrac']:
        for i in doseidx:
            idx = '__' + str(i)
            idic[param + idx] = pdset.columns.get_loc(param + idx)

    # The time increments for which model prediction is calculated
    time = np.arange(min(classM.timeV), max(classM.timeV))

    # Initialize variables
    colors = sns.color_palette('hls', 10)
    pltparams = ['confl', 'apop', 'dna']
    patches = []

    # Iterate over each dose
    for cidx in range(len(doseidx)):
        # Get model predictions over time
        calcsets = calcset(pdset, str(doseidx[cidx]), time, idic)
        # Iterate over phase, apop, and dna
        for i in range(3):
            calc = calcsets[i]
            # Get median & 90% confidence interval for each time point
            qqq = np.percentile(calc, [5, 25, 50, 75, 95], axis=0)
            # Plot confidence interval
            ax[i].plot(time, qqq[2, :], c=colors[cidx], linewidth=1)
            ax[i].fill_between(time, qqq[1, :], qqq[3, :], alpha=0.5, color=colors[cidx])
            ax[i].fill_between(time, qqq[0, :], qqq[4, :], alpha=0.2, color=colors[cidx])

            # Plot observation
            idx1 = doseidx[cidx] * len(classM.timeV)
            idx2 = (doseidx[cidx] + 1) * len(classM.timeV)
            ax[i].scatter(classM.timeV, classM.expTable[pltparams[i]][idx1:idx2],
                          c=colors[cidx], marker='.', s=5)

            # Label Plot
            ax[i].set_title(pltparams[i])
            ax[i].set_xticks([0, 25, 50, 75])
            ax[i].set_xlabel('Time (hr)')
            if i != 0:
                ax[i].set_ylim([0, 1.3])
        # For each dose, add a patch to legend
        patches.append(mpatches.Patch(color=colors[cidx], label=str(doses[cidx]) + ' ' + unit))
    ax[0].set_ylabel(drug + ' Confluence')

    return (ax, patches)


def sim_plots(filename, drugs=None, unit='nM'):
    ''' Plot sampling predictions overlaying experimental data for multiple drugs '''
    sns.set_context("paper", font_scale=2)
    # If drugs given, make simulation plots for selected drugs
    if drugs != None:
        for drug in drugs:
            simulation(filename, drug, unit=unit)
    # If drugs not given, plot all drugs
    else:
        classM, _ = readModel()
        drugs = list(set(classM.drugs))
        drugs.remove('Control')
        for drug in drugs:
            simulation(filename, drug, unit=unit)
