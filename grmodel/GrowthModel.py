from scipy.integrate import odeint
from numpy import arange
from pylab import plot, figure, xlabel, ylabel, legend, show
import matplotlib.patches as mpatches
import pandas as pd

#calculates rate at the given time
def rate_values(parameters, time):
    from math import exp
    
    if time < 0:
        raise ValueError
        
    list_of_rates = []
    for rate_equation in parameters:
        poly_sum = 0
        for i, item in enumerate(rate_equation):
            poly_sum += item*time**i
        poly_sum = exp(poly_sum)
        list_of_rates.append(poly_sum)

    return list_of_rates

class GrowthModel(object):
    #takes in mc data of list and returns equal length list of lists
    def mcFormat(mcParams):
        interval = len(mcParams)//5
        extra = len(mcParams)%5
        params = [mcParams[0:interval]] + [mcParams[interval:2*interval]] + [mcParams[2*interval:3*interval]] + [mcParams[3*interval:4*interval]] + [mcParams[4*interval:5*interval]]
        for i in range(extra):
            params[i].append(mcParams[5*interval+i])
        return params

	#ODE function of cells living/dying/undergoing early apoptosis
	#
	#params:
	#state	the number of cells in a particular state (LIVE, DEAD, EARLY_APOPTOSIS, GONE)
	#t 	time
	#a 	function describing the parameter between LIVE -> LIVE (cell division)
	#b 	function describing the parameter between LIVE -> DEAD
	#c 	function describing the parameter between LIVE -> EARLY_APOPTOSIS
	#d 	function describing the parameter between EARLY_APOPTOSIS -> DEATH
	#e 	function describing the parameter between DEATH -> GONE
    def ODEfun(self, state, t, params):
        LIVE = state[0]
        DEAD = state[1]
        EARLY_APOPTOSIS = state[2]
        GONE = state[3]
        dydt = [0] * 4
        a,b,c,d,e = rate_values(params, t)
        dydt[0] = a*LIVE - b*LIVE - c*LIVE
        dydt[1] = b*LIVE - e*DEAD + d*EARLY_APOPTOSIS
        dydt[2] = c*LIVE - d*EARLY_APOPTOSIS
        dydt[3] = e*DEAD
        return dydt

	#solves the ODE function given a set of initial values (y0),
	#over a time interval (t_interval)
	#
	#params:
	#params	list of parameters for model (a, b, c, d, e)
	#t_interval 	time interval over which to solve the function

	#y0 	list with the initial values for each state
    def simulate(self, params, t_interval, y0):
    		out = odeint(self.ODEfun, y0, t_interval, args = (params,))
    		#put values and time into pandas datatable
    		out_table = pd.DataFrame(data=out, index=t_interval, columns = ['Live', 'Dead', 'EarlyApoptosis', 'Gone'])
    		out_table.insert(0, 'Time', t_interval)
    		return out_table

	#plots the results from a simulation
	#if animate is True then the line plot over time
	#plots the results from a simulation
	#if animate is True then the line plot over time
    def plotSimulation(self, state):
    		figure()
    		xlabel('Time')
    		ylabel('Number of Cells')
    		t_interval = state.iloc[:,0].values
    		plot(t_interval, state.iloc[:, 1], 'b-', label="live")
    		plot(t_interval, state.iloc[:, 2], 'r-', label="dead")
    		plot(t_interval, state.iloc[:, 3], 'g-', label="early apoptosis")
    		plot(t_interval, state.iloc[:, 4], 'k-', label="gone")

    		legend(loc='upper right')

    		show()


if __name__ == '__main__':
    t = arange(0, 10, 0.005)
    init_state = [10000, 0, 0, 0]
    params = GrowthModel.mcFormat([.0009, -.016, .01, .008, .0007, .005, -.001,-.0071, .0008, .005])
    out = GrowthModel().simulate(params, t, init_state)
    GrowthModel().plotSimulation(out)
