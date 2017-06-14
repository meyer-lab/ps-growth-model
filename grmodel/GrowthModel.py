import numpy as np
from numba import jit

np.seterr(over='raise')


def logpdf_sum(x, loc, scale):
    """
    Calculate the likelihood of an observation applied to a
    normal distribution.
    """
    prefactor = - np.log(scale * np.sqrt(2 * np.pi))
    summand = -np.square((x - loc) / (np.sqrt(2) * scale))
    return prefactor + summand


@jit
def ODEfun(ss, t, rates):
    """
    ODE function of cells living/dying/undergoing early apoptosis

    params:
    ss   the number of cells in a particular state
            (LIVE, DEAD, EARLY_APOPTOSIS)
    t   time
    a   parameter between LIVE -> LIVE (cell division)
    b   parameter between LIVE -> DEAD
    c   parameter between LIVE -> EARLY_APOPTOSIS
    d   parameter between EARLY_APOPTOSIS -> DEATH
    e   parameter between DEATH -> GONE
    """
    LIVE = np.exp((rates[0] - rates[1] - rates[2]) * t)

    return [rates[1] * LIVE - rates[4] * ss[0] + rates[3] * ss[1],
            rates[2] * LIVE - rates[3] * ss[1]]


@jit
def jacFun(state, t, rates):
    return np.array([[-rates[4], rates[3]], [0.0, -rates[3]]])


def simulate(params, t_interval):
    """
    Solves the ODE function given a set of initial values (y0),
    over a time interval (t_interval)

    params:
    params	list of parameters for model (a, b, c, d, e)
    t_interval 	time interval over which to solve the function

    y0 	list with the initial values for each state
    """
    from scipy.integrate import odeint

    def liveNum(t):
        return np.exp(params[0] - params[1] - params[2] * t)

    out, infodict = odeint(ODEfun, [0.0, 0.0], t_interval, Dfun=jacFun,
                           args=(params,), full_output=True, mxstep=5000)

    # Calculate live cell numbers
    live = np.expand_dims(liveNum(t_interval), axis=1)

    # Add numbers to the output matrix
    out = np.concatenate((live, out), axis=1)

    if infodict['message'] != 'Integration successful.':
        raise FloatingPointError(infodict['message'])

    return (t_interval, out)


class GrowthModel:

    def logL(self, paramV):
        """
        Run simulation using paramV, and compare results to observations in
        self.selCol
        """

        # Return -inf for parameters out of bounds
        if not np.all(np.isfinite(paramV)):
            return -np.inf
        elif np.any(np.less(paramV, self.lb)):
            return -np.inf
        elif np.any(np.greater(paramV, self.ub)):
            return -np.inf

        paramV = np.power(10, paramV.copy())

        # Calculate model data table
        try:
            model = simulate(paramV, self.uniqueT)
        except FloatingPointError:
            return -np.inf

        # Scale model data table with conversion constants
        confl_mod = paramV[-4] * np.interp(self.expTable[0],
                                           model[0], np.sum(model[1], axis=1))
        green_mod = paramV[-3] * np.interp(self.expTable[0],
                                           model[0],
                                           model[1][:, 1] + model[1][:, 2])

        # Run likelihood function with modeled and experiemental data, with
        # standard deviation given by last two entries in paramV
        try:
            logSqrErr = np.sum(logpdf_sum(self.expTable[1],
                                          loc=confl_mod, scale=paramV[-2]))
            logSqrErr += np.sum(logpdf_sum(self.expTable[2],
                                           loc=green_mod, scale=paramV[-1]))

            # Specify preference for the conversion constants to be similar
            logSqrErr += logpdf_sum(np.log(paramV[-3] / paramV[-4]), 0.0, 0.1)
        except FloatingPointError:
            return -np.inf

        return logSqrErr

    def __init__(self, selCol, loadFile=None):
        """ Initialize class. """
        from os.path import join, dirname, abspath
        import pandas

        # If no filename is given use a default
        if loadFile is None:
            loadFile = "091916_H1299_cytotoxic_confluence"

        # Find path for csv files in the repository.
        pathcsv = join(dirname(abspath(__file__)), 'data/' + loadFile)

        # Read in both observation files. Return as formatted pandas tables.
        # Data tables to be kept within class.
        data_confl = pandas.read_csv(pathcsv + '_confl.csv',
                                     infer_datetime_format=True)
        data_green = pandas.read_csv(pathcsv + '_green.csv',
                                     infer_datetime_format=True)

        # Pull out selected column data
        self.selCol = selCol

        # Make experimental data table - Time, Confl, Green
        self.expTable = [data_confl.iloc[:, 1].as_matrix(),
                         data_confl.iloc[:, self.selCol].as_matrix(),
                         data_green.iloc[:, self.selCol].as_matrix()]

        # Match time range and interval to experimental time range and interval
        self.uniqueT = np.sort(np.unique(self.expTable[0]))
        self.tRange = (np.min(self.uniqueT), np.max(self.uniqueT))

        # Parameter names
        self.pNames = ['a', 'b', 'c', 'd', 'e', 'conv_confl',
                       'conv_green', 'err_confl', 'err_green']

        # Specify lower bounds on parameters (log space)
        self.lb = np.full(len(self.pNames), -6.0, dtype=np.float64)
        self.lb[-4:-2] = -2.0

        # Specify upper bounds on parameters (log space)
        self.ub = np.full(len(self.pNames), 0.0, dtype=np.float64)
        self.ub[-4:-2] = 4.0

        # Set number of parameters
        self.Nparams = len(self.ub)
