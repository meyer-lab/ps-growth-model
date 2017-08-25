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
        ff = "042017_PC9"

    filename = './grmodel/data/' + ff + '_samples.pkl'

    # Read in list of classes
    classList = pickle.load(bz2.BZ2File(filename, 'rb'))

    return classList

def readCols(cols, trim = True):
    ''' Load specified columns '''
    grCols = []
    grdf = []
    classList = read_dataset()
    for ii, item in enumerate(classList):
        if item.selCol in cols:
            grCols.append(item)

    for item in grCols:
        df = pm.backends.tracetab.trace_to_dataframe(item.samples)
        df['Column'] = item.selCol
        if item.condName[-2:] == '.1':
            df['Condition'] = item.condName[:-2]
        else:
            df['Condition'] = item.condName

        if trim:
            cutoff = np.amin(df['ssqErr'])+200
            df = df.loc[df['ssqErr'] < cutoff,:]
        grdf.append(df)

    grdf = pd.concat(grdf)

    if len(cols) == 1:
        grCols = grCols[0]

    return (grCols, grdf)

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
            print('Column ' + str(item.selCol) + ' sampling not converged according to Geweke.')
            print('z-score surpassed 1 for ' + str(gewekeDivnum) + ' times for parameters ' + str(gewekeDivparam) + ': \n' + str(gewekeDiv)) 
#            print('p-value = ' +str(1-sp.stats.chi2.cdf(gewekeOut, gewekeNum)))
#            print('gewekeOut = ' + str(gewekeOut))
#            print('gewekeNum = ' + str(gewekeNum))
            if divparamnum > 0:
                print('divparamnum = ' + str(divparamnum) + ' for param(s) ' + str(divparam))
            print('\n')
            flag = False

        if min(neff.values()) < 100:
            print('Column ' + str(item.selCol) + ' effective N of sampling is less than 100.')
            print(neff)
            print('\n')
            flag = False

        if max(gr.values()) > 1.1:
            print('Gelman-Rubin statistic failed for column ' + str(item.selCol))
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


def sim_plot(column, rep=None, data=None, printt=False, show=True, axis=None):
    """ 
    Given column, plots simulation of predictions overlaying observed data 
    rep: number of replicates per column
    printt: print dataset and dataset shape
    show: show figure
    axis: pass in figure axis
    """
    # Read in dataset to Pandas data frame
    if data == None:
        classM, pdset = readCols([column])
    else:
        classM, pdset = data

    if printt:
        print(pdset.shape)
        print(pdset)

    #Initialize variables 
    if rep != None:
        time = classM.timeV.reshape(rep, int(len(classM.timeV) / rep))[0, :]
    else:
        time = classM.timeV
    calcset = np.full((pdset.shape[0], len(time)), np.inf)
    calcseta = np.full((pdset.shape[0], len(time)), np.inf)
    calcsetd = np.full((pdset.shape[0], len(time)), np.inf)

    varr = 0
    # Get indexes for params
    idic = {}
    for param in ['div', 'd', 'deathRate', 'apopfrac', 'confl_conv', 'apop_conv', 'dna_conv', 'std']:
        idic[param] = pdset.columns.get_loc(param)
    # Evaluate predictions for each set of parameter fits
    for row in pdset.iterrows():
        mparm = np.copy(np.array(row[1].as_matrix()[[idic['div'], idic['d'], idic['deathRate'], idic['apopfrac']]]))
#        try:
            # Use old_model to calculate lnum, eap, and dead over time
        simret = simulate(mparm, time)
        if rep:
            simret = simret[:len(time), :]
            simret = simret.reshape((len(time), rep))

            # Calculate predictions for total, apop, and dead cells over time
        calcset[varr, :] = np.sum(simret, axis=1) * row[1].as_matrix()[idic['confl_conv']]
        calcseta[varr, :] = np.sum(simret[:, 1:3], axis=1) * row[1].as_matrix()[idic['apop_conv']]
        calcsetd[varr, :] = np.sum(simret[:, 2:4], axis=1) * row[1].as_matrix()[idic['dna_conv']]

        varr = varr + 1
#        except:
#            print('Failed')
#            continue
    
    # Plot prediction distribution and observation
    # Set up axis and color
    if axis == None:
        plt.figure()
        axis = plt.gca()
    colors = ['b', 'g', 'r']

    # Iterate over total, apop, and dead cels
    calcsets = [calcset, calcseta, calcsetd]
    for i in list(range(len(calcsets))):
        calc = calcsets[i]
        c = colors[i]
        # Get median & 90% confidence interval for each time point
        qqq = np.percentile(calc, [5, 25, 50, 75, 95], axis=0)
        # Plot confidence interval 
        axis.plot(time, qqq[2, :], color = c)
        axis.fill_between(time, qqq[1, :], qqq[3, :], alpha=0.5)
        axis.fill_between(time, qqq[0, :], qqq[4, :], alpha=0.2)
    # Plot observation 
    axis.scatter(classM.timeV, classM.expTable['confl'], color = 'b', marker = '.')
    axis.scatter(classM.timeV, classM.expTable['apop'], color = 'g', marker = '.')
    axis.scatter(classM.timeV, classM.expTable['dna'], color = 'r', marker = '.')

    if show:
        plt.show()
        axis.set_xlabel('Time (hr)')
        axis.set_ylabel('% Confluence')
        axis.set_title('Sim_plot '+pdset['Condition'].as_matrix()[0])
    else:
        title = pdset['Condition'].as_matrix()[0]
        return (axis, title)


def sim_plots(columns=None, rep = None):
    if columns is None:
        columns = list(range(3,19))
    f, ax = plt.subplots(len(columns)//4, 4, figsize = (9.2, 2.3*(len(columns)//4)), sharex=True, sharey= False)
    for i in list(range(len(columns))):
        axis, title = sim_plot(columns[i], show=False, axis = ax[i//4, i%4], rep = rep)
        ax[i//4, i%4] = axis
        ax[i//4, i%4].set_title(title)
        if i%4 == 0:
            ax[i//4, i%4].set_ylabel('% Confluence')
        if i//4 == len(columns)//4-1:
            ax[i//4, i%4].set_xlabel('Time (hr)')
    plt.tight_layout()
    plt.title('Sim_Plots', x = -3, y = 6)
    plt.show()


def fit_plot(param, column, replica = False):
    '''
    Inputs: param = a list of len(7) in normal space, column = column for corresponding observation
    Plot model prediction overlaying observation
    '''
    # Import an instance of GrowthModel
    classM, _ = readCols([column])

    # Initialize variables and parameters 
    if replica:
        ltime = int(len(classM.timeV)/3)
    else:
        ltime = int(len(classM.timeV))
    calcset = np.full((ltime), np.inf)
    calcseta = np.full((ltime), np.inf)
    calcsetd = np.full((ltime), np.inf)
    mparm = param[0:4]

    # Use old model to calculate cells nubmers
    simret = classM.old_model(mparm, param[4], param[5], param[6])[1]
    if replica:
        simret = simret[:ltime,:]
    simret = simret.reshape(ltime,3)

    # Calculate total, apop, and dead cells 
    calcset[:] = np.sum(simret,axis = 1) * param[4]
    calcseta[:] = np.sum(simret[:,1:3], axis = 1) * param[5]
    calcsetd[:] = simret[:,2] * param[6]
    
    # Plot prediction curves overlayed with observation 
    if replica:
        plt.plot(classM.timeV.reshape(3,ltime)[0,:], calcset)
        plt.plot(classM.timeV.reshape(3,ltime)[0,:], calcseta)
        plt.plot(classM.timeV.reshape(3,ltime)[0,:], calcsetd)
    else:
        plt.plot(classM.timeV, calcset)
        plt.plot(classM.timeV, calcseta)
        plt.plot(classM.timeV, calcsetd)
    plt.scatter(classM.timeV, classM.expTable['confl'])
    plt.scatter(classM.timeV, classM.expTable['apop'])
    plt.scatter(classM.timeV, classM.expTable['dna'])
    plt.title('Fit_plot '+str(column))
    plt.xlabel('Time (hr)')
    plt.ylabel('% Confluence')
    plt.show()