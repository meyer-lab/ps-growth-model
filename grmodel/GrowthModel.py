from scipy.integrate import odeint
import pandas as pd
import numpy as np

np.seterr(over='raise')

def rate_values(parameters, time):
    '''Calculates the rates of each cellular process at a given time

    Returns a list of values of e raised to the power of teh calculated rate
    (for each set of parameters) as to ensure no negative value. The input is
    a list of lists, each index of the inner list a coeficient of time to the power
    of that index. The outer list represents the number of functions.

    Arguments:
        parameters (list of lists): contains number of functions and parameters of each funciton
        time (float): time at which rate is calculated

    Returns:
        list of the rates for each each function
    '''

    if time < 0:
        raise ValueError

    tt = np.power(time, np.arange(0, parameters.shape[0], dtype=np.float))

    return np.exp(np.matmul(parameters.T, tt.T))

def logpdf_sum(x, loc, scale):
    """ Calculate the likelihood of an observation applied to a normal distribution. """
    prefactor = - np.log(scale * np.sqrt(2*np.pi))
    summand = -np.square((x - loc)/(np.sqrt(2) * scale))
    return prefactor + summand

def likelihood(table_merge, sigma_sim_live, sigma_sim_dead):
    """
    Calculates the log likelihood of a simulation given the experimental data.

    parameters:
    table_sim	merged table with the transformed simulation predictions
    sigma_sim_live	the sigma used for the normal distribution for live cells
    sigma_sim_dead	the sigma used for the normal distribution for dead cells
    """

    logSqrErr = 0

    for i, row in table_merge.iterrows():
        try:
            logSqrErr += logpdf_sum(row.Confl_data, loc=row.Confl_model, scale=sigma_sim_live)
            logSqrErr += logpdf_sum(row.Green_data, loc=row.Green_model, scale=sigma_sim_dead)
        except:
            return -np.inf

    return logSqrErr

def ODEfun(state, t, params):
    """
    ODE function of cells living/dying/undergoing early apoptosis

    params:
    state	the number of cells in a particular state (LIVE, DEAD, EARLY_APOPTOSIS, GONE)
    t 	time
    a 	parameter between LIVE -> LIVE (cell division)
    b 	parameter between LIVE -> DEAD
    c 	parameter between LIVE -> EARLY_APOPTOSIS
    d 	parameter between EARLY_APOPTOSIS -> DEATH
    e 	parameter between DEATH -> GONE
    """

    ## If we don't have the right number of parameters, then panic
    if params.shape[1] != 5:
        raise ValueError

    LIVE, DEAD, EARLY_APOPTOSIS = state[0], state[1], state[2] # GONE is state[3]
    dydt = np.full(4, 0.0, dtype=np.float)

    rates = rate_values(params, t)

    dydt[0] = rates[0]*LIVE - rates[1]*LIVE - rates[2]*LIVE
    dydt[1] = rates[1]*LIVE - rates[4]*DEAD + rates[3]*EARLY_APOPTOSIS
    dydt[2] = rates[2]*LIVE - rates[3]*EARLY_APOPTOSIS
    dydt[3] = rates[4]*DEAD
    return dydt

def mcFormat(mcParams):
    """ takes in mc data of list and returns equal length list of lists """
    if len(mcParams) % 5 > 0:
        raise ValueError("Length of mcParams must be a multiple of 5.")

    return np.reshape(mcParams, (-1, 5), order='F')

def simulate(params, t_interval):
    """
    Solves the ODE function given a set of initial values (y0),
    over a time interval (t_interval)
    
    params:
    params	list of parameters for model (a, b, c, d, e)
    t_interval 	time interval over which to solve the function

    y0 	list with the initial values for each state
    """
    out, infodict = odeint(ODEfun, [1.0, 0.0, 0.0, 0.0], t_interval, args=(params,), full_output=True, printmessg=False)
    #put values and time into pandas datatable
    out_table = pd.DataFrame(data=out,
                             index=t_interval,
                             columns=['Live', 'Dead', 'EarlyApoptosis', 'Gone'])
    out_table.insert(0, 'Time', t_interval)

    if infodict['message'] != 'Integration successful.':
        raise FloatingPointError(infodict['message'])

    return out_table

def paramsWithinLimits(params, t_int, maxVal):
    """
    This only checks that the parameters don't pass through the bound in the duration.
    It's still possible they start too high, which is handled by the lower and upper
    bounds hopefully.
    """

    # Iterate over the parameters
    for ii in range(params.shape[1]):
        # Copy so we don't mess with the original
        pval = params[:, ii].copy()

        # Move by offset so roots tell us when we pass over limit
        pval[-1] -= maxVal

        # Find roots
        outt = np.roots(pval)

        # If any of the roots fall within the time interval and are real, fail
        for jj in outt:
            if np.isreal(jj) and jj > t_int[0] and jj < t_int[1]:
                return False

    return True


class GrowthModel:
    def plotSimulation(self, paramV):
        """
        Plots the results from a simulation.
        TODO: Run simulation when this is called, and also plot observations.
        TODO: If selCol is None, then plot simulation but not observations.
        """
        import matplotlib.pyplot as plt
        
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
            

    def logL(self, paramV):
        """
        TODO: Run simulation using paramV, and compare results to observations in self.selCol
        """

        if self.selCol is None:
            raise ValueError

        # Return -inf for parameters out of bounds
        if not np.all(np.isfinite(paramV)):
            return -np.inf
        elif np.any(np.less(paramV, self.lb)) or np.any(np.greater(paramV, self.ub)):
            return -np.inf

        #format parameters to list of lists (except last 4 entries)
        params = mcFormat(paramV[:-4])

        # Check that the parameter values are reasonable over the interval
        if not paramsWithinLimits(params, (np.min(self.uniqueT), np.max(self.uniqueT)), 2.0):
            return -np.inf
        
        # Calculate model data table
        try:
            model = simulate(params, self.uniqueT)
        except FloatingPointError as error:
            return -np.inf
        
        # Power transform conversion constants
        paramV[-4:] = np.power(10, paramV[-4:])
        
        #scale model data table with conversion constants
        model['Confl'] = paramV[-4]*model['Live']
        model['Green'] = paramV[-3]*(model['Dead'] + model['EarlyApoptosis'])

        data_merge = self.expTable.merge(model, how='left', on='Time', suffixes=['_data', '_model'])
        
        #run likelihood function with modeled and experiemental data, with standard 
        #deviation given by last two entries in paramV
        return likelihood(data_merge, paramV[-2], paramV[-1])

    def setselCol(self, inV):
        self.selCol = inV

        # Make experimental data table
        data_frames = [self.data_confl.iloc[:,1], self.data_confl.iloc[:, self.selCol], self.data_green.iloc[:, self.selCol]]
        data = pd.concat(data_frames, axis = 1)
        data.columns = ['Time', 'Confl', 'Green']

        self.expTable = data

        #match time range and interval to experimental time range and interval
        self.uniqueT = np.sort(np.unique(self.expTable['Time'].as_matrix()))

    def __init__(self, loadFile=None, complexity=3, selCol = None):
        import os
        import itertools

        # If no filename is given use a default
        if loadFile is None:
            loadFile = "091916_H1299_cytotoxic_confluence"

        # Find path for csv files, on any machine wherein the repository exists.
        path = os.path.dirname(os.path.abspath(__file__))
        
        # Read in both observation files. Return as formatted pandas tables.
        # Data tables to be kept within class.
        self.data_confl = pd.read_csv(os.path.join(path, ('data/' + loadFile + '_confl.csv')), infer_datetime_format=True)
        self.data_green = pd.read_csv(os.path.join(path, ('data/' + loadFile + '_green.csv')), infer_datetime_format=True)

        # Parameter names
        ps = ['a', 'b', 'c', 'd', 'e']
        self.pNames = list(itertools.chain.from_iterable(itertools.repeat(x, complexity) for x in ps))
        self.pNames = self.pNames + ['conv_confl', 'conv_green', 'err_confl', 'err_green']

        # Specify lower bounds on parameters (log space)
        self.lb = np.full(len(self.pNames), -9, dtype=np.float64)

        # Specify upper bounds on parameters (log space)
        self.ub = np.full(len(self.pNames), 1, dtype=np.float64)
        self.ub[-4:] = 4

        # Set number of parameters
        self.Nparams = len(self.ub)

        # Save selected data column in class
        if not selCol is None:
            self.setselCol(selCol)
