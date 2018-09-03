""" This Python script implements lsq to fit the parameters IC50_X1,
    IC50_X2, hill_X1, hill_X2, Econ and a in the model defined """
import numpy as np
from scipy.optimize import least_squares
from .drugCombination import concentration_effects, makeAdditiveData


def model(X, coeffs, hold):
    """ Define the model that will be fit """
    IC1 = [coeffs[0], coeffs[2]]
    IC2 = [coeffs[1], coeffs[3]]
    E_con = coeffs[4]

    if hold:
        a = 0.0
    else:
        a = coeffs[5]

    X1 = X[0]
    X2 = X[1]
    return concentration_effects(IC1, IC2, a, E_con, X1, X2)


def residuals(coeffs, Y, X, hold):
    """ Define the residuals function that we want to minimize using the least squares method """
    return Y - model(X, coeffs, hold)


def lsqFitting(df=makeAdditiveData(t=72.0, a=0.0, E_con=1.0), hold=False):
    """ Least Squares Curve Fitting
    This function returns the best fitted coefficients, the sthe fitted number of live cells """
    lnum = df['lnum'].values

    X = np.array((df['X1'].values, df['X2'].values), dtype=np.float64)

    # Starting point information
    if hold: # IC1, IC2, hill1, hill2, E_con
        c0 = {'low': np.array([0.0, 0.0, -5.0, -5.0, 0.1]), 'high': np.array([5.0, 10.0, 0.0, 0.0, 1.0])}
        B = [(0, 0, -np.inf, -np.inf, 0), (np.inf, np.inf, 0, 0, np.inf)]
    else: # IC1, IC2, hill1, hill2, E_con, a
        c0 = {'low': np.array([0.0, 0.0, -5.0, -5.0, 0.1, -5.0]), 'high': np.array([5.0, 10.0, 0.0, 0.0, 1.0, 5.0])}
        B = [(0, 0, -np.inf, -np.inf, 0, -np.inf), (np.inf, np.inf, 0, 0, np.inf, np.inf)]

    for ii in range(2):
        # Least squares method to find the parameters that minimize the residuals
        try:
            c = least_squares(residuals, np.random.uniform(**c0), bounds=B, args=(lnum, X, hold), verbose=1)
        except ValueError:
            # ValueError when the residuals are infinitely large
            continue

        if ii == 0:
            cSave = c
        elif cSave.cost > c.cost:
            cSave = c

    return cSave.x, cSave.cost
