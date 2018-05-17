import numpy as np
from scipy.optimize import leastsq
from drugCombination import concentration_effects, makeAdditiveData

# This Python script implements lsq to fit the parameters IC50_X1,
# IC50_X2, hill_X1, hill_X2, Econ and a in the model defined


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
    residuals = Y - model(X, coeffs, hold)

    residuals = residuals.flatten()
    return residuals


def lsqFitting(df=makeAdditiveData(t=72.0, a=0.0, E_con=1.0), hold=False):
    """ Least Squares Curve Fitting """
    lnum = df.pivot(index='X1', columns='X2', values='lnum').as_matrix()

    X = np.array((df['X1'].values, df['X2'].values), dtype=np.float64)

    if hold:
        c0 = np.array([1.0, 2.0, -1.0, -1.0, 0.6], dtype=np.float64)
    else:
        c0 = np.array([1.0, 2.0, -1.0, -1.0, 0.6, 0.0], dtype=np.float64)

    c, flag = leastsq(residuals, c0, args=(lnum, X, hold))

    if hold:
        print('IC50_X1=', c[0], 'IC50_X2=', c[1], 'hill_X1=', c[2], 'hill_X2=', c[3], 'E_con=', c[4])
    else:
        print('IC50_X1=', c[0], 'IC50_X2=', c[1], 'hill_X1=', c[2], 'hill_X2=', c[3], 'E_con=', c[4], 'a=', c[5])

    lnum_fit = model(X, c, hold)
    # df['error'] = abs(df['lnum'] - df['lnum_fit'])
    # this should be added to a unit test code

    return lnum_fit

# TODO: Need to not be running the code as a script
M = lsqFitting()
print(M)
