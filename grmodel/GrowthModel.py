from scipy.integrate import odeint
from numpy import arange
from pylab import plot, figure, xlabel, ylabel, legend, show
import matplotlib.patches as mpatches

class GrowthModel(object):
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
	def ODEfun(self, state, t, a, b, c, d, e):
		LIVE = state[0]
		DEAD = state[1]
		EARLY_APOPTOSIS = state[2]
		GONE = state[3]
		dydt = [0] * 4
		dydt[0] = a(t)*LIVE - b(t)*LIVE - c(t)*LIVE
		dydt[1] = b(t)*LIVE - e(t)*DEAD + d(t)*EARLY_APOPTOSIS
		dydt[2] = c(t)*LIVE - d(t)*EARLY_APOPTOSIS
		dydt[3] = e(t)*DEAD
		return dydt

	#solves the ODE function given a set of initial values (y0),
	#over a time interval (t_interval)
	#
	#params:
	#params	list of parameters for model (a, b, c, d, e)
	#t_interval 	time interval over which to solve the function
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
