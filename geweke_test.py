# -*- coding: utf-8 -*-
"""
Created on Tue May 16 14:31:39 2017

@author: cstea
"""
import h5py
import numpy as np
#from pymc import geweke


def geweke_single_chain(chain1, chain2=None):
    """
    Perform the Geweke Diagnostic between two univariate chains. If two chains are input 
    instead of one, Student's t-test is performed instead.
    """
    from scipy.stats import ttest_ind

    len0 = chain1.shape[0]
    if chain2 is None:
        chain2 = chain1[int(np.ceil(len0/2)):len0]
        chain1 = chain1[0:int(np.ceil(len0*0.1))]
    statistic, pvalue = ttest_ind(chain1,chain2)
    return (statistic, pvalue)

def param_history(file = 'mcmc_chain.h5'):
    '''
    Makes a dictionary with each parameter as the key and it's value as a 
    list of lists, each inner list the history of that parameter for one of the
    76 parralel walkers. Then, for every column-data stored in the file, do 
    this and store it is a broader dictionary.
    '''
    file = h5py.File(file, 'r')
    column_dict = {}
    for i in range(3, 20):
        col = file['column' + str(i)]
        dataset = col['data']
        params_dict = {}
        #go through index of that parameter in the dataset
        for j in range(3, 18):  
            one_param = []
            #go through how many times each walker is updated
            for k in range(76):
                one_param_one_walker = []
                for l in range(dataset.shape[0] // 76):
                    one_param_one_walker.append(dataset[k+l*75, j])
                one_param.append(one_param_one_walker)
            params_dict['param' + str(j-2)] = one_param
            
        column_dict['column' + str(i)] = params_dict
            
    file.close()
    return column_dict
    

def geweke_single_col(column, file = 'mcmc_chain.h5'):
    '''get two outputs:
        1. a dictionary with a list of (t-statistic, p-value)
        for each of the 76 walkers under the key of that parameter 
        2. a dictionary with a lits of boolean values for if 
        that p-value is valid'''
    
    params_dict = param_history(file)['column' + str(column)]
    z_score_dict = {}
    converging_dict = {}
    for param, walkers_history in params_dict.items():
        z_scores = []
        converging = []
        
        
        
        for history in walkers_history:
            format_history = np.array(history)      
            z_score = geweke_single_chain(format_history)
            z_scores.append(z_score)
            converge = True
            if z_score[1] < .1:
                converge = False
            converging.append(converge)
            
######if we use the imported model geweke from pymc --> wasnt working LinAlgError: Singular matrix
#        for history in walkers_history:
#            format_history = np.array(history)      
#            z_score = geweke(format_history)
#            z_scores.append(z_score)
#            converge = True
#            for z in z_score:
#                if z[1] < -2 or z[1] > 2:
#                    converge = False
#            converging.append(converge)
            
        z_score_dict[param] = z_scores
        converging_dict[param] = converging
    
    return z_score_dict, converging_dict

def geweke(file = 'mcmc_chain.h5'):
    #run geweke_single_col through all columns
    columns_dict = {}
    for i in range(3, 20):
        values, correct = geweke_single_col(i, file)
        columns_dict['column' + str(i)] = [values, correct]
    return columns_dict

