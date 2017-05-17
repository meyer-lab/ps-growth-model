import numpy as np
import scipy as sp
import scipy.stats
import pandas

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m, m-h, m+h

def growth_rate_plot(file, complexity, column):
    f = h5py.File(file, 'r')
    columnData = '/column' + str(column) +'/data'
    dset = f[columnData]
    # Read in dataset to Pandas data frame
    pdset = pd.DataFrame(dset.value)
    
    #get columns with growth rate
    growth_rates = pdset.iloc[:,2:(2+complexity)]

    t_interval = np.arange(0, 72, 0.5)
    
    #evaluate each of the growth rates over the time interval
    rows_list = []
    for i in range(growth_rates.shape[0]):
        growth_rate = growth_rates.iloc[i,:].tolist()
        yest = pylab.polyval(growth_rate, t_interval)
        rows_list.append(yest)
        #rows_list.append(np.exp(yest)) #Initially had this here but I got a "FloatingPointError: overflow encountered in exp"
    df = pandas.DataFrame(rows_list)
    
    #for each time point, get the mean and 95% confidence interval
    rows_list_2 = []
    for i in range(df.shape[1]):
        timepoint = df.iloc[:,i]
        rows_list_2.append(np.exp(list(mean_confidence_interval(timepoint))))
    df_2 = pandas.DataFrame(rows_list_2)
    
    plt.figure(figsize=(10,10))
    plt.plot(t_interval, np.array(df_2.iloc[:,0]))
    plt.fill_between(t_interval, df_2.iloc[:,1], df_2.iloc[:,2], alpha = 0.3)
    plt.show()



#sgrowth_rate_plot('first_chain.h5', 3, 3)
