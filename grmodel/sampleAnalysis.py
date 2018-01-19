import bz2
import os
import pymc3 as pm
import numpy as np
import scipy as sp
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends import backend_pdf
from collections import deque

from .pymcGrowth import simulate

try:
    import cPickle as pickle
except ImportError:
    import pickle


def read_dataset(ff=None, traceplot = False):
    ''' Read the specified column from the shared test file. '''

    if ff is None:
        ff = "101117_H1299_ends"

    filename = './grmodel/data/' + ff + '_samples.pkl'

    # Read in list of classes
    classList = pickle.load(bz2.BZ2File(filename,'rb'))
    
    # Save traceplots if traceplot = True
    if traceplot:
        tracefile = './grmodel/data/' + ff + '_' + 'traceplot.pdf'

        # Delete the existing file if it exists
        if os.path.exists(tracefile):
            os.remove(tracefile)

        with backend_pdf.PdfPages(tracefile, keep_empty=False) as pdf:
            # Output sampling for each column
            pm.plots.traceplot(classList.samples)
            fig = plt.gcf()
            pdf.savefig(fig)
            matplotlib.pyplot.close()

    return classList

def readModel(ff=None, trim=False):
    model = read_dataset(ff)
    df = pm.backends.tracetab.trace_to_dataframe(model.samples)
    if trim:
        for name in df.columns.values:
            if 'logp' in str(name):
                cutoff = np.amin(df[name])+50
                df = df.loc[df[name] < cutoff, :]
    return (model, df)

def diagnostics(item, plott=False):
    """ Check the convergence and general posterior properties of a chain. """
    flag = True
    # Iterate over sampling classes
        # Calc Geweke stats
    geweke = pm.geweke(item.samples)

    # Calculate effective n
    neff = pm.effective_n(item.samples)

    # Calculate Gelman-Rubin statistics
    gr = pm.gelman_rubin(item.samples)

    # Initialize variables 
    gewekeOut = 0.0
    gewekeNum = 0
    gewekeDivnum = 0
    gewekeDiv = []
    gewekeDivparam = []

    #Keep track of diverging chains for each column
    divparamnum = 0
    divparam = []

    for _, value in geweke.items():
        for kk, vv in value.items():
            try: # Single chain
                Vec = np.absolute(vv[:, 1])
            except TypeError: # Multiple chains, flatten the np array
                Vec = np.concatenate([x for x in vv])
                Vec = np.absolute(Vec[:, 1])
            
            intervals = len(Vec)
            VecDiv = [val for val in Vec if val >= 1]
            divnum = len([val for val in Vec if val >= 1.96])

            lenVecDiv = len(VecDiv)
            gewekeDivnum += lenVecDiv
            if lenVecDiv > 0:
                gewekeDiv.extend(VecDiv)
                gewekeDivparam.append(kk)
            gewekeOut += np.sum(Vec)
            gewekeNum += Vec.size

            # Hypothesis testing for each parameter
            z = (divnum - intervals*0.05) / np.sqrt(intervals*0.05*0.95)
            p_value = 1 - sp.stats.norm.cdf(z)
            if p_value <= 0.05:
                divparamnum += 1
                divparam.append(kk)

    # Let the z-score surpass 1 up to three times, or fewer with higher deviation
    # TODO: Need to come up with a null model for Geweke to test for convergence
    if gewekeDivnum > 3:
        print('Column ' + str(item.selCols) + ' sampling not converged according to Geweke.')
        print('z-score surpassed 1 for ' + str(gewekeDivnum)
              + ' times for parameters ' + str(gewekeDivparam) + ': \n' + str(gewekeDiv))

        if divparamnum > 0:
            print('divparamnum = ' + str(divparamnum) + ' for param(s) ' + str(divparam))
        print('\n')
        flag = False

    # Get a flattened list of effective n
    neffvals = getvalues(neff)
    # 
    if min(neffvals) < 100:
        print('Column ' + str(item.selCols) + ' effective N of sampling is less than 100.')
        print(neff)
        print('\n')
        flag = False

    # Get a flattened list of Gelman-Rubin statistics
    grvals = getvalues(gr)
    if max(grvals) > 1.1:
        print('Gelman-Rubin statistic failed for column ' + str(item.selCols))
        print(gr)
        print('\n')
        flag = False

    # Only output the posterior plot if we've converged
    if plott and flag:
        saveplot(item, pm.plot_posterior)

    # Made it to the end so consistent with converged
    return flag

def getvalues(dic):
    """Take dic, a dictionary with lists or item as values
    Return vals, a flattened list of values in the dictionary"""
    values = list(dic.values())
    vals = []
    for sublist in values:
        try:
            vals.append(float(sublist))
        except TypeError:
            vals.extend(sublist.tolist())
    return vals
    
def saveplot(cols, func):
    """ Take in cols, pymc models, and func, a plotting funciton
    Make and save the plots"""
    filename = './grmodel/data/' + cols[0].loadFile + '_' + func.__name__ + '.pdf'

    # Delete the existing file if it exists
    if os.path.exists(filename):
        os.remove(filename)

    with backend_pdf.PdfPages(filename, keep_empty=False) as pdf:
        # Output sampling for each column
        for col in cols:
            fig, axis = matplotlib.pyplot.subplots(9, 1)
            axis = func(col.samples, ax=axis)
            matplotlib.pyplot.tight_layout()
            pdf.savefig(fig)
            matplotlib.pyplot.close()

def calcset(pdset, idx, time, idic):
    """Calculate model predictions based on parameter fits from """
    # Initialize counter
    varr = 0
    # Initialize 3 numpy 2D arrays 
    calcset, calcseta, calcsetd = (np.full((pdset.shape[0], len(time)), np.inf) for _ in range(3))
    # Interate over each row of sampling data
    for row in pdset.iterrows():
        # Get parameter values
        mparm = np.copy(np.array(row[1].as_matrix()[[idic['div__'+idx],
                                                     idic['d'],
                                                     idic['deathRate__'+idx],
                                                     idic['apopfrac__'+idx]]]))
        # Get raw model predictions
        simret = simulate(mparm, time)
        # Apply conversion factors to model predictions
        # Fill-in one row of the numpy arrays
        calcset[varr, :] = (np.sum(simret, axis=1)
                            * row[1].as_matrix()[idic['confl_conv']])
        calcseta[varr, :] = (np.sum(simret[:, 1:3], axis=1)
                             * row[1].as_matrix()[idic['apop_conv']]
                             + row[1].as_matrix()[idic['apop_offset']])
        calcsetd[varr, :] = (np.sum(simret[:, 2:4], axis=1)
                             * row[1].as_matrix()[idic['dna_conv']]
                             + row[1].as_matrix()[idic['dna_offset']])
        varr += 1
    return (calcset, calcseta, calcsetd)

def simulation(drug, unit='nM'):
    """Make simulation plots of experimental data overlayed with model predictions"""
    # Load model and dataset
    classM, pdset = readModel()
    # A list of all conditions
    alldrugs = classM.drugs
    # A list of all doses, one for each condition
    alldoses = classM.doses
    # Initialize variables
    doses = deque() # Drug doses
    doseidx = deque() # Indces of conditions that matches drug of interest or is control
    flag = True

    # Iterate over conditions in reverse order
    for i in range(len(alldrugs)-1, -1, -1):
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
            idic[param+idx] = pdset.columns.get_loc(param+idx)

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
            idx2 = (doseidx[cidx]+1) * len(classM.timeV)
            ax[i].scatter(classM.timeV, classM.expTable[pltparams[i]][idx1:idx2],
                          c=colors[cidx], marker='.', s=20)

            # Label Plot
            ax[i].set_title(pltparams[i])
            ax[i].set_xticks([0,25,50,75])
            ax[i].set_xlabel('Time (hr)')
            if i != 0:
                ax[i].set_ylim([0,1.3])
        # For each dose, add a patch to legend
        patches.append(mpatches.Patch(color=colors[cidx], label=str(doses[cidx])+' '+unit))
    ax[0].set_ylabel(drug+' Confluence')
    # Show legend 
    plt.legend(handles=patches, fontsize=10, bbox_to_anchor=(1.5, 1))
    plt.tight_layout()
    return ax

def sim_plots(drugs=None, unit='nM'):
    ''' Plot sampling predictions overlaying experimental data for multiple drugs '''
    import seaborn as sns
    sns.set_context("paper", font_scale=2)
    # If drugs given, make simulation plots for selected drugs
    if drugs != None:
        for drug in drugs:
            simulation(drug, unit=unit)
    # If drugs not given, plot all drugs
    else:
        classM, _ = readModel()
        drugs = list(set(classM.drugs))
        drugs.remove('Control')
        for drug in drugs:
            simulation(drug, unit=unit)
