import os
from scipy.integrate import odeint
from pylab import plot, figure, xlabel, ylabel, legend, show
import pandas as pd
import numpy as np

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
    from numpy.polynomial.polynomial import polyval
    np.seterr(over='raise')

    if time < 0:
        raise ValueError
    if len(parameters) == 0:
        return []

    list_of_rates = np.exp(polyval(time, np.asmatrix(parameters).T))

    return list_of_rates

def logpdf_sum(x, loc, scale):
    """ Calculate the likelihood of a set of observations applied to a normal distribution. """
    root2 = np.sqrt(2)
    root2pi = np.sqrt(2*np.pi)
    prefactor = - x.size * np.log(scale * root2pi)
    summand = -np.square((x - loc)/(root2 * scale))
    return  prefactor + np.nansum(summand)

def likelihood(table_sim, sigma_sim_live, sigma_sim_dead, table_exp):
    """
    Calculates the log likelihood of a simulation given the experimental data.

    parameters:
    table_sim	pandas datatable containing the simulated data for cells over time
    			assumes the table has rows with columns with the timepoints, the number of live, dead, early
    			apoptosis, and "gone" cells
    sigma_sim_live	the sigma used for the normal distribution for live cells
    sigma_sim_dead	the sigma used for the normal distribution for dead cells
    table_exp	a pandas datatable containing the experimental values for
                live/dead cells over time (can have multiple values for the same timepoint)
    			assumes that the table columns in the following order: time elapsed, live cells, dead cells
    """

    #get time points (time points should be in first column of table_exp)
    timepoints = pd.Series.unique(table_exp.iloc[:, 0])
    #for each time point, calculate likelihood of simulated data given experimental data at that time point.
    logSqrErr = 0
    for time in timepoints:
        #get simulated values at time point
        sim_at_timepoint = table_sim.loc[table_sim.iloc[:, 0] == time, :]
        mean_live = sim_at_timepoint.iloc[:, 1]
        mean_dead = sim_at_timepoint.iloc[:, 2]

        #get observed values at time point
        exp_at_timepoint = table_exp.loc[table_exp.iloc[:, 0] == time, :]
        observed_live = exp_at_timepoint.iloc[:, 1]
        observed_dead = exp_at_timepoint.iloc[:, 2]

        #calculate the density of the distribution and sum up across all observed values
        logSqrErr += logpdf_sum(observed_live, loc=mean_live, scale=sigma_sim_live)
        logSqrErr += logpdf_sum(observed_dead, loc=mean_dead, scale=sigma_sim_dead)

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
    if len(params) < 5:
        raise ValueError

    LIVE, DEAD, EARLY_APOPTOSIS, GONE = state[0], state[1], state[2], state[3]
    dydt = np.full(4, 0.0, dtype=np.float)

    rates = rate_values(params, t)

    dydt[0] = rates[0]*LIVE - rates[1]*LIVE - rates[2]*LIVE
    dydt[1] = rates[1]*LIVE - rates[4]*DEAD + rates[3]*EARLY_APOPTOSIS
    dydt[2] = rates[2]*LIVE - rates[3]*EARLY_APOPTOSIS
    dydt[3] = rates[4]*DEAD
    return dydt

def mcFormat(mcParams):
    """ takes in mc data of list and returns equal length list of lists """
    interval = len(mcParams)//5
    extra = len(mcParams)%5
    params = [mcParams[0:interval]] + [mcParams[interval:2*interval]] + [mcParams[2*interval:3*interval]] + [mcParams[3*interval:4*interval]] + [mcParams[4*interval:5*interval]]
    for i in range(extra):
        params[i].append(mcParams[5*interval+i])
    return params

def simulate(params, t_interval, y0):
    """
    Solves the ODE function given a set of initial values (y0),
    over a time interval (t_interval)
    
    params:
    params	list of parameters for model (a, b, c, d, e)
    t_interval 	time interval over which to solve the function

    y0 	list with the initial values for each state
    """
    out = odeint(ODEfun, y0, t_interval, args=(params,))
    #put values and time into pandas datatable
    out_table = pd.DataFrame(data=out,
                             index=t_interval,
                             columns=['Live', 'Dead', 'EarlyApoptosis', 'Gone'])
    out_table.insert(0, 'Time', t_interval)
    return out_table

class GrowthModel:
    def plotSimulation(self, paramV):
        """
        Plots the results from a simulation.
        TODO: Run simulation when this is called, and also plot observations.
        TODO: If selCol is None, then plot simulation but not observations.
        """
        #calculate model data table
        params = mcFormat(paramV[:-4])
        t_interval = np.arange(0, self.data_confl.iloc[-1, 1], (self.data_confl.iloc[2, 1] - self.data_confl.iloc[1,1]))
        if self.selCol:
            y0 = [self.data_confl.iloc[0, self.selCol], self.data_green.iloc[0, self.selCol], 0, 0]
        else:
            y0 = [10000, 0, 0, 0] 
        state = simulate(params, t_interval, y0)
        
        #plot model data table
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

        #format parameters to list of lists (excpet last 4 entries)
        params = mcFormat(paramV[:-4])
        
        #match time range and interval to experimental time range and interval
        t_interval = np.sort(np.unique(self.data_confl.iloc[:, 1].as_matrix()))
        
        #match initial cell numbers to experimental data
        #####DO I MATCH STARTING VALUE WITH IT? IS THIS WHERE CELL # PARAMS COME IN?
        y0 = [self.data_confl.iloc[0, self.selCol], self.data_green.iloc[0, self.selCol], 0, 0]
        
        #calculate model data table
        model = simulate(params, t_interval, y0)
        
        #make experimental data table
        data_frames = [self.data_confl.iloc[:,1], self.data_confl.iloc[:, self.selCol], self.data_green.iloc[:, self.selCol]]
        data = pd.concat(data_frames, axis = 1)
        
        #run likelihood function with modeled and experiemental data, with standard 
        #deviation given by last two entries in paramV
        logL = likelihood(model, paramV[-2], paramV[-1], data)
        return logL


    def __init__(self, loadFile=None, complexity=3, selCol = None):
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
        self.ub = np.full(len(self.pNames), 5, dtype=np.float64)

        # Set number of parameters
        self.Nparams = len(self.ub)

        # Save selected data column in class
        self.selCol = selCol
