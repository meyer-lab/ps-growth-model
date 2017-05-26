# -*- coding: utf-8 -*-
"""
Created on Tue May 16 14:31:39 2017

@author: cstea
"""
import h5py
import numpy as np
import matplotlib.pyplot as plt
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
    return pvalue

def walkers_info(col_num, dataset):    
    for param_num in range(3, 17):
        walkers_pvalues = []
        for k in range(76):
            one_param_one_walker = []
            for l in range(dataset.shape[0] // 76):
                one_param_one_walker.append(dataset[k+l*75, param_num])
            walkers_pvalues.append(geweke_single_chain(np.array(one_param_one_walker)))
        
        yield param_num, walkers_pvalues
        


def geweke_test(col_num = 3, file = 'first_chain.h5'):
    file = h5py.File(file, 'r')
    dataset = file['column' + str(col_num) + '/data']
    
    for param_num, walkers_pvalues in walkers_info(col_num, dataset):
        x = np.arange(1, len(walkers_pvalues)+1, 1)
        y = np.array(walkers_pvalues)
        plt.title('paramater ' + str(param_num))
        plt.scatter(x, y)
        plt.xlabel('walker number')
        plt.ylabel('p-value')        
        plt.show()
    
    file.close()

def pass_geweke_test(min_pval = .0001 , col_num = 3, file = 'first_chain.h5'):
    file = h5py.File(file, 'r')
    dataset = file['column' + str(col_num) + '/data']
    
    flag = True
    for param_num, walkers_pvalues in walkers_info(col_num, dataset):
        for i in range(len(walkers_pvalues)):
            if walkers_pvalues[i] < min_pval:
                print('Failed at parameter ' + str(param_num) + ', walker ' + str(i) + '. ' + 'P-value was ' + str(walkers_pvalues[i]))
                flag = False
    
    return flag

def all_cols_pass_geweke_test(min_pval = .0001, file = 'first_chain.h5'):
    flag = True
    for i in range(3, 20):
#        print('entered column ' + str(i))
        if not pass_geweke_test(min_pval, i, file):
            print('Column ' + str(i) + ' failed.')
            flag = False
    return flag
    
def see_col(col_num, file = 'first_chain.h5'):
    '''Useful for debugging'''
    file = h5py.File(file, 'r')
    dataset = file['column' + str(col_num) + '/data']
    print(dataset)
    
#if __name__ == '__main__':
#    geweke_test()
