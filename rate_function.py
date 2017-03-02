# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 15:31:46 2017

@author: cstea
"""

def rate_values(parameters, time):
    '''our_parameters is a list of lists; the value in each index of the inner
    list is the coeficient of time to the power of that index. The outer list
    represents the number of functions. Returns a list of values of e raised 
    to the power of the calculated rate (for each set of parameters) as to
    ensure no value is negative. 
    '''
    
    from math import exp
    
    list_of_rates = []
    for rate_equation in parameters:
        poly_sum = 0
        for i in range(len(rate_equation)):
            poly_sum += rate_equation[i]*time**i
        poly_sum = exp(poly_sum)
        list_of_rates.append(poly_sum)
    return list_of_rates