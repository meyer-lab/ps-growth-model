import matplotlib.pyplot as plt

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