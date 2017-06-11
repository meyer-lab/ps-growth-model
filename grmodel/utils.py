import numpy as np
import scipy as sp
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt

try:
    import cPickle as pickle
except ImportError:
    import pickle

def geweke_single_chain(chain1, chain2=None):
    """
    Perform the Geweke Diagnostic between two univariate chains. If two chains are input 
    instead of one, Student's t-test is performed instead. Returns p-value.
    """
    from scipy.stats import ttest_ind

    len0 = chain1.shape[0]
    if chain2 is None:
        chain2 = chain1[int(np.ceil(len0/2)):len0]
        chain1 = chain1[0:int(np.ceil(len0*0.1))]

    return ttest_ind(chain1, chain2)[1]


def confidence_interval(data, confidence=0.95):
    ''' Return the median and confidence interval. '''
    a = np.array(data, dtype=np.float64)

    return np.median(a), np.percentile(a, 1 - confidence), np.percentile(a, confidence)


def read_dataset(column, filename=None):
    ''' Read the specified column from the shared test file. '''
    import os
    import h5py

    if filename is None:
        filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), "./data/first_chain.h5")

    # Open hdf5 file
    f = h5py.File(filename, 'r')

    # Create pointer to main data set
    dset = f['/column' + str(column) +'/data']

    if dset is None:
        raise AssertionError("Dataset from hdf5 was read as empty.")

    # Read in StoneModel and unpickle
    classM = pickle.loads(dset.attrs['class'].tobytes())

    # Turn matrix into dataframe
    pdset = pd.DataFrame(dset.value, columns = classM.pNames)

    print(str(pdset.shape[0]) + ' rows read.')

    f.close()

    return (classM, pdset)


def growth_rate_plot(complexity, column):

    # Read in dataset to Pandas data frame
    classM, pdset = read_dataset(column)
    
    # Get columns with growth rate
    growth_rates = pdset.iloc[:,2:(2+complexity)]

    # Time interval to solve over
    t_interval = np.arange(0, np.max(classM.uniqueT), 0.2)
    
    # Evaluate each of the growth rates over the time interval
    rows_list = []
    for i in range(growth_rates.shape[0]):
        growth_rate = growth_rates.iloc[i,:].tolist()
        yest = np.polyval(growth_rate, t_interval)
        rows_list.append(yest)
        #rows_list.append(np.exp(yest)) #Initially had this here but I got a "FloatingPointError: overflow encountered in exp"
    df = pd.DataFrame(rows_list)
    
    # Get median & 95% confidence interval for each time point
    rows_list_2 = []
    for i in range(df.shape[1]):
        timepoint = df.iloc[:,i]
        rows_list_2.append(np.exp(list(confidence_interval(timepoint))))
    df_2 = pd.DataFrame(rows_list_2)
    
    plt.figure(figsize=(10,10))
    plt.plot(t_interval, np.array(df_2.iloc[:,0]))
    plt.fill_between(t_interval, df_2.iloc[:,1], df_2.iloc[:,2], alpha = 0.3)
    plt.show()



def plotSimulation(self, paramV):
    """
    Plots the results from a simulation.
    TODO: Run simulation when this is called, and also plot observations.
    TODO: If selCol is None, then plot simulation but not observations.
    """
    
    #calculate model data table
    params = mcFormat(paramV[:-4])
    t_interval = np.arange(0, self.data_confl.iloc[-1, 1], (self.data_confl.iloc[2, 1] - self.data_confl.iloc[1,1]))
    
    state = simulate(params, t_interval)
    
    #plot simulation results; if selCol is not None, also plot observations
    if self.selCol is not None:
        #print(self.selCol)
        data_confl_selCol = self.data_confl.iloc[:,self.selCol]
        data_green_selCol = self.data_green.iloc[:,self.selCol]
        t_interval_observ = self.data_confl.iloc[:,1]
        
        #get conversion constants
        conv_confl, conv_green = np.power(10, paramV[-4:-2])
        
        #adjust simulation values
        simulation_confl = state.iloc[:,1] * conv_confl
        simulation_green = (state.iloc[:,2] + state.iloc[:,3])* conv_green
        
        
        f, axarr = plt.subplots(3, figsize=(10,10))
        axarr[0].set_title('Simulation Results')
        t_interval = state.iloc[:, 0].values
        axarr[0].plot(t_interval, state.iloc[:, 1], 'b-', label="live")
        axarr[0].plot(t_interval, state.iloc[:, 2], 'r-', label="dead")
        axarr[0].plot(t_interval, state.iloc[:, 3], 'g-', label="early apoptosis")
        axarr[0].plot(t_interval, state.iloc[:, 4], 'k-', label="gone")
        axarr[0].legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
        
        axarr[1].set_title('Observed: data_confl')
        axarr[1].plot(t_interval_observ, data_confl_selCol, label = 'data_confl')
        axarr[1].plot(t_interval, simulation_confl, label = 'simulation_confl')
        axarr[1].legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
        
        axarr[2].set_title('Observed: data_green')
        axarr[2].plot(t_interval_observ, data_green_selCol, label = 'data_green')
        axarr[2].plot(t_interval, simulation_green, label = 'simulation_green')
        axarr[2].legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
        plt.tight_layout()
        plt.show()
    else:
        figure()
        xlabel('Time')
        ylabel('Number of Cells')
        t_interval = state.iloc[:, 0].values
        plot(t_interval, state.iloc[:, 1], 'b-', label="live")
        plot(t_interval, state.iloc[:, 2], 'r-', label="dead")
        plot(t_interval, state.iloc[:, 3], 'g-', label="early apoptosis")
        plot(t_interval, state.iloc[:, 4], 'k-', label="gone")
        legend(loc='upper right')
        show()

