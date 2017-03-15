from scipy.integrate import odeint
from numpy import arange
from pylab import plot, figure, xlabel, ylabel, legend, show
import matplotlib.patches as mpatches

# to do: 1. mc gives params in a list not list of lists, 2. don't call funciton each time
#for every value a,b,c,e... 3. put rate function in here --> cleaner
#4. unit test 5. get this to work 
class GrowthModel(object):
    
    #calculates rate at the given time
    def rate_values(self, parameters, time):    
        from math import exp
        list_of_rates = []
        for rate_equation in parameters:
            poly_sum = 0
            for i in range(len(rate_equation)):
                poly_sum += rate_equation[i]*time**i
            poly_sum = exp(poly_sum) #change this??
            list_of_rates.append(poly_sum)
        return list_of_rates

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
        a,b,c,d,e = self.rate_values(params, t)
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
<<<<<<< HEAD
	#y0 	list with the initial values for each state 
    def simulate(self, params, t_interval, y0):
    		out = odeint(self.ODEfun, y0, t_interval, args = (params,))
    		return out

	#plots the results from a simulation
	#if animate is True then the line plot over time 
    def plotSimulation(self, state, t_interval, animate):
    		fig = figure()
    		xlabel('Time')
    		ylabel('Number of Cells')
    		if animate:
    			plot(t_interval, state[:, 0], 'b-', alpha = 0.2, label = "live")
    			plot(t_interval, state[:, 1], 'r-', alpha = 0.2,label = "dead")
    			plot(t_interval, state[:, 2], 'g-', alpha = 0.2,label = "early apoptosis")
    			plot(t_interval, state[:, 3], 'k-', alpha = 0.2,label = "gone")
    		else:
    			plot(t_interval, state[:, 0], 'b-', label = "live")
    			plot(t_interval, state[:, 1], 'r-', label = "dead")
    			plot(t_interval, state[:, 2], 'g-', label = "early apoptosis")
    			plot(t_interval, state[:, 3], 'k-', label = "gone")
    		legend(loc='upper right')
    		#show()
    		if animate:
    			def animate(i):
    			     plot(t_interval[0:i], state[0:i,0], 'b-')
    			     plot(t_interval[0:i], state[0:i,1], 'r-')
    			     plot(t_interval[0:i], state[0:i,2], 'g-')
    			     plot(t_interval[0:i], state[0:i,3], 'k-')
    
    			ani = animation.FuncAnimation(fig, animate, interval=1)
    		show()


if __name__ == '__main__':
    t = arange(0, 10, 0.005)
    init_state = [10000, 0, 0, 0]
    params = GrowthModel.mcFormat([.0009, -.016, .01, .008, .0007, .005, -.001,-.0071, .0008, .005])
    #will turn the list above into what mc gives us
    out = GrowthModel().simulate(params, t, init_state)
    GrowthModel().plotSimulation(out, t, True)
=======
	#y0 	list with the initial values for each state
	def simulate(self, params, t_interval, y0):
		out = odeint(self.ODEfun, y0, t_interval, args=tuple(params))
		return out

	#plots the results from a simulation
	#if animate is True then the line plot over time
	def plotSimulation(self, state, t_interval):
		figure()
		xlabel('Time')
		ylabel('Number of Cells')

		plot(t_interval, state[:, 0], 'b-', label="live")
		plot(t_interval, state[:, 1], 'r-', label="dead")
		plot(t_interval, state[:, 2], 'g-', label="early apoptosis")
		plot(t_interval, state[:, 3], 'k-', label="gone")

		legend(loc='upper right')

		show()


if __name__ == '__main__':
	t = arange(0, 200, 0.2)
	init_state = [10000, 0, 0, 0]
	params = [lambda t: 0.2, lambda t: 0.2, lambda t: 0.1, lambda t: 0.7, lambda t: .5]
	out = GrowthModel().simulate(params, t, init_state)
	GrowthModel().plotSimulation(out, t)
>>>>>>> origin/master
