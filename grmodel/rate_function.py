# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 15:31:46 2017

@author: cstea
"""

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
        
    >>> rate_values([[.01, 1, .9,],[3, .7, 1, -2], [-7, -1, 1]], 2)
    [273.14423800475663, 0.0005004514334406108, 0.006737946999085467]
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

def a(t, parameters):
    return rate_values(t, parameters)[0]

def b(t, parameters):
    return rate_values(t, parameters)[1]

def c(t, parameters):
    return rate_values(t, parameters)[2]

def d(t, parameters):
    return rate_values(t, parameters)[3]

def e(t, parameters):   
    return rate_values(t, parameters)[4]