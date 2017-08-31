import bz2
import os
import pymc3 as pm
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt 
from matplotlib.backends import backend_pdf
from .pymcGrowth import simulate

try:
    import cPickle as pickle
except ImportError:
    import pickle


def read_dataset(ff=None):
    ''' Read the specified column from the shared test file. '''

    if ff is None:
        ff = "030317-2_H1299"

    filename = './grmodel/data/' + ff + '_samples.pkl'

    # Read in list of classes
    classList = pickle.load(bz2.BZ2File(filename, 'rb'))

    return classList

def readModels(drugs=None, trim = False):
    ''' 
    Load specified drugs
    grModels: dict of drug : GrowthModel object
    grdf: dict of drug: df
    '''
    grModels = dict()
    grdf = []
    classList = read_dataset()
    for ii, item in enumerate(classList):
        if drugs==None or item.drug in drugs:
            grModels[item.drug] = item
            df = pm.backends.tracetab.trace_to_dataframe(item.samples)
            df['Columns'] = str(item.selCols)
            df['Drug'] = item.drug

            if trim:
                for name in df.columns.values:
                    if 'ssqErr' in str(name):
                        cutoff = np.amin(df[name])+50
                        df = df.loc[df[name] < cutoff,:]
            grdf.append(df)
    grdf = pd.concat(grdf)

    return (grModels, grdf)

def diagnostics(classList, plott=False):
    """ Check the convergence and general posterior properties of a chain. """
    flag = True
    # Iterate over sampling columns
    for ii, item in enumerate(classList):

        # Calc Geweke stats
        geweke = pm.geweke(item.samples)

        # Calculate effective n
        neff = pm.effective_n(item.samples)

        # Calculate Gelman-Rubin statistics
        gr = pm.gelman_rubin(item.samples)

        gewekeOut = 0.0
        gewekeNum = 0
        gewekeDivnum = 0
        gewekeDiv = []
        gewekeDivparam = []
        #Keeps track of diverging chains for each column
        divparamnum = 0
        divparam = []

        for key, value in geweke.items():
            for kk, vv in value.items():
                Vec = np.absolute(vv[:, 1])
                intervals = len(Vec)
                VecDiv= [val for val in Vec if val >= 1]
                divnum = len([val for val in Vec if val >= 1.96])
                
                gewekeDivnum += len(VecDiv)
                if len(VecDiv) > 0:
                    gewekeDiv.extend(VecDiv)
                    gewekeDivparam.append(kk)
                gewekeOut += np.sum(Vec)
                gewekeNum += Vec.size

                # Hypothesis testing for single chain
                z = (divnum - intervals*0.05) / np.sqrt(intervals*0.05*0.95)
                p_value = 1 - sp.stats.norm.cdf(z)
                if p_value <= 0.05:
                    divparamnum += 1
                    divparam.append(kk)

        # Let the z-score surpass 1 up to three times, or fewer with higher deviation
        # TODO: Need to come up with a null model for Geweke to test for convergence
        if gewekeDivnum > 3:
            print('Column ' + str(item.selCols) + ' sampling not converged according to Geweke.')
            print('z-score surpassed 1 for ' + str(gewekeDivnum) + ' times for parameters ' + str(gewekeDivparam) + ': \n' + str(gewekeDiv)) 
#            print('p-value = ' +str(1-sp.stats.chi2.cdf(gewekeOut, gewekeNum)))
#            print('gewekeOut = ' + str(gewekeOut))
#            print('gewekeNum = ' + str(gewekeNum))
            if divparamnum > 0:
                print('divparamnum = ' + str(divparamnum) + ' for param(s) ' + str(divparam))
            print('\n')
            flag = False

        if min(neff.values()) < 100:
            print('Column ' + str(item.selCols) + ' effective N of sampling is less than 100.')
            print(neff)
            print('\n')
            flag = False

        if max(gr.values()) > 1.1:
            print('Gelman-Rubin statistic failed for column ' + str(item.selCols))
            print(gr)
            print('\n')
            flag = False

    # Only output the posterior plot if we've converged
    if plott and flag:
        saveplot(classList, pm.plot_posterior)

    # Made it to the end so consistent with converged
    return flag


def saveplot(cols, func):
    """ X """
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


def sim_plot(drug, confl = True, rep=None, printt=False):
    """ 
    Given column, plots simulation of predictions overlaying observed data 
    rep: number of replicates per column
    printt: print dataset and dataset shape
    show: show figure
    axis: pass in figure axis
    """
    # Read in dataset to Pandas data frame
    classdict, pdset = readModels([drug])
    classM = classdict[drug]

    if printt:
        print(pdset.shape)
        print(pdset)

    #Initialize variables 
    if rep != None:
        time = classM.timeV.reshape(rep, int(len(classM.timeV) / rep))[0, :]
    else:
        time = classM.timeV

    # Get indexes for params
    idic = {}
    for param in ['confl_conv', 'apop_conv', 'dna_conv', 'std', 'stdad']:
        idic[param] = pdset.columns.get_loc(param)
    for param in ['div ', 'd ', 'deathRate ', 'apopfrac ']:
        for dose in classM.doses: 
            idic[param+str(dose)] = pdset.columns.get_loc(param+str(dose))

    # Set up colors and subplots
    if confl:
        colors = ['b', 'g', 'r']
    else:
        colors = ['g', 'r']
    f, ax = plt.subplots(1, len(classM.doses), figsize = (3*len(classM.doses), 3), sharex = True, sharey = False)
    for dose in classM.doses:
        calcset = np.full((pdset.shape[0], len(time)), np.inf)
        calcseta = np.full((pdset.shape[0], len(time)), np.inf)
        calcsetd = np.full((pdset.shape[0], len(time)), np.inf)
        varr = 0
        # Evaluate predictions for each set of parameter fits
        for row in pdset.iterrows():
            mparm = np.copy(np.array(row[1].as_matrix()[[idic['div '+str(dose)], idic['d '+str(dose)], idic['deathRate '+str(dose)], idic['apopfrac '+str(dose)]]]))

            # Use old_model to calculate lnum, eap, and dead over time
            simret = simulate(mparm, time)
            if rep:
                simret = simret[:len(time), :]
                simret = simret.reshape((len(time), 4))

            # Calculate predictions for total, apop, and dead cells over time
            if confl:
                calcset[varr, :] = np.sum(simret, axis=1) * row[1].as_matrix()[idic['confl_conv']]
            calcseta[varr, :] = np.sum(simret[:, 1:3], axis=1) * row[1].as_matrix()[idic['apop_conv']]
            calcsetd[varr, :] = np.sum(simret[:, 2:4], axis=1) * row[1].as_matrix()[idic['dna_conv']]

            varr = varr + 1

        # Iterate over total, apop, and dead cels
        if confl: 
            calcsets = [calcset, calcseta, calcsetd]
        else:
            calcsets = [calcseta, calcsetd]
        for i in list(range(len(calcsets))):
            calc = calcsets[i]
            c = colors[i]
            # Get median & 90% confidence interval for each time point
            qqq = np.percentile(calc, [5, 25, 50, 75, 95], axis=0)
            # Plot confidence interval
            if len(classM.doses) > 1:
                ax[classM.doses.index(dose)].plot(time, qqq[2, :], color = c)
                ax[classM.doses.index(dose)].fill_between(time, qqq[1, :], qqq[3, :], alpha=0.5)
                ax[classM.doses.index(dose)].fill_between(time, qqq[0, :], qqq[4, :], alpha=0.2)
            else:
                ax.plot(time, qqq[2, :], color = c)
                ax.fill_between(time, qqq[1, :], qqq[3, :], alpha=0.5)
                ax.fill_between(time, qqq[0, :], qqq[4, :], alpha=0.2)
        # Plot observation 
        if len(classM.doses) > 1: 
            if confl: 
                ax[classM.doses.index(dose)].scatter(classM.timeV, classM.expTable[(dose,'confl')], color = 'b', marker = '.')
            ax[classM.doses.index(dose)].scatter(classM.timeV, classM.expTable[(dose,'apop')], color = 'g', marker = '.')
            ax[classM.doses.index(dose)].scatter(classM.timeV, classM.expTable[(dose,'dna')], color = 'r', marker = '.')
            ax[classM.doses.index(dose)].set_xlabel('Time (hr)')
            ax[classM.doses.index(dose)].set_title(drug + ' ' + str(dose))
            if classM.doses.index(dose) == 0:
                ax[0].set_ylabel('% Confluence')
        else:
            if confl: 
                ax.scatter(classM.timeV, classM.expTable[(dose,'confl')], color = 'b', marker = '.')
            ax.scatter(classM.timeV, classM.expTable[(dose,'apop')], color = 'g', marker = '.')
            ax.scatter(classM.timeV, classM.expTable[(dose,'dna')], color = 'r', marker = '.')
            ax.set_xlabel('Time (hr)')
            ax.set_ylabel('% Confluence')
            plt.title('Sim_plot '+str(drug))

    plt.tight_layout()
    plt.show()


def sim_plots(drugs = None, confl = True, rep = None):
    if drugs!= None:
        for drug in drugs:
            sim_plot(drug, confl = confl, rep = rep)
    else:
        classdict, pdset = readModels()
        for drug in classdict:
            sim_plot(drug, confl = confl, rep = rep)