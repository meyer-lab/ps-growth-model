import scipy.stats
import pandas as pd
import numpy as np

def logpdf_sum(x, loc, scale):
    root2 = np.sqrt(2)
    root2pi = np.sqrt(2*np.pi)
    prefactor = - x.size * np.log(scale * root2pi)
    summand = -np.square((x - loc)/(root2 * scale))
    return  prefactor + np.nansum(summand)

# Calculates the log likelihood of a simulation given the experimental data.
#
#
# parameters:
# table_sim			pandas datatable containing the simulated data for cells over time
#					assumes the table has rows with columns with the timepoints, the number of live, dead, early
#					apoptosis, and "gone" cells
# sigma_sim_live	the sigma used for the normal distribution for live cells
# sigma_sim_dead	the sigma used for the normal distribution for dead cells
# table_exp	a pandas datatable containing the experimental values for live/dead cells over time (can have
#						multiple values for the same timepoint)
#						assumes that the table columns in the following order: time elapsed, live cells, dead cells
def likelihood(table_sim, sigma_sim_live, sigma_sim_dead, table_exp):
	#get time points (time points should be in first column of table_exp)
	timepoints = pd.Series.unique(table_exp.iloc[:, 0])
	#for each time point, calculate likelihood of simulated data given experimental data at that time point.
	logSqrErr = 0
	for time in timepoints:
		#get simulated values at time point
		sim_at_timepoint = table_sim.loc[table_sim.iloc[:, 0] == time,:]
		mean_live = table_sim.iloc[:, 1]
		mean_dead = table_sim.iloc[:, 2]

		#get observed values at time point
		exp_at_timepoint = table_exp.loc[table_exp.iloc[:, 0] == time,:]
		observed_live = exp_at_timepoint.iloc[:, 1]
		observed_dead = exp_at_timepoint.iloc[:, 2]

		#calculate the density of the distribution and sum up across all observed values
		logSqrErr += logpdf_sum(observed_live, loc=mean_live, scale=sigma_sim_live)
		logSqrErr += logpdf_sum(observed_dead, loc=mean_dead, scale=sigma_sim_dead)

	return logSqrErr


##simple example using one set of the experimental data as the simulated data

#load the data (took code from load_data.py)

data_confl = pd.read_csv("./091916_H1299_cytotoxic_confluence_confl.csv", infer_datetime_format=True)
#data_confl.shape #(111, 21)
data_confl = data_confl.ix[:, 0:19] #remove "blank" column
data_green = pd.read_csv("./091916_H1299_cytotoxic_confluence_green.csv", infer_datetime_format=True)
#data_green.shape #(111, 21)
data_green = data_green.ix[:, 0:19]

#get one experiment, in this case CTL 0 nM
frames = [data_green.iloc[:, 1], data_green.iloc[:, 3], data_confl.iloc[:, 3]]
table_exp = pd.concat(frames, axis =1)

table_sim = table_exp.iloc[0:31, :]

likelihood(table_sim, 1,1,table_exp)
