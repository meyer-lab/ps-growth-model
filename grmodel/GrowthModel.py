import numpy as np
from numba import jit

np.seterr(over='raise')


@jit(nopython=True, cache=True)
def logpdf_sum(x, loc, scale):
    """
    Calculate the likelihood of an observation applied to a
    normal distribution.
    """
    prefactor = - np.log(scale * np.sqrt(2 * np.pi))
    summand = -np.square((x - loc) / (np.sqrt(2) * scale))
    return prefactor + summand


@jit(nopython=True, cache=True)
def preCalc(t, params):
    # Growth rate
    GR = params[0] - params[1] - params[2]

    liveNum = np.exp(GR * t)

    # Number of early apoptosis cells at start is 0.0
    Cone = -params[2] / (GR + params[3])

    eapop = params[2] / (GR + params[3]) * np.exp(GR * t)
    eapop = eapop + Cone * np.exp(-params[3] * t)

    return (liveNum, eapop)


def simulate(params, ts):
    """
    Solves the ODE function given a set of initial values (y0),
    over a time interval (ts)

    params:
    params	list of parameters for model (a, b, c, d, e)
    ts 	time interval over which to solve the function

    y0 	list with the initial values for each state

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
    from scipy.integrate import odeint

    def ODEfun(ss, t):
        lnum, eap = preCalc(t, params)

        return params[1] * lnum - params[4] * ss + params[3] * eap

    out = odeint(ODEfun, 0.0, ts)

    # Calculate precalc cell numbers
    lnum, eap = preCalc(ts, params)

    # Add numbers to the output matrix
    out = np.concatenate((np.expand_dims(lnum, axis=1),
                          out,
                          np.expand_dims(eap, axis=1)), axis=1)

    return out


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
            model = simulate(paramV, self.expTable[0])
        except FloatingPointError:
            return -np.inf

        # Scale model data table with conversion constants
        confl_mod = paramV[-4] * np.sum(model, axis=1)
        green_mod = paramV[-3] * model[:, 1] + model[:, 2]

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
            loadFile = "091916_H1299_confluence"

        # Property list
        properties = {'confl':'_phase.csv',
                      'apop':'_green.csv',
                      'dna':'_red.csv'}

        # Find path for csv files in the repository.
        pathcsv = join(dirname(abspath(__file__)), 'data/' + loadFile)

        # Pull out selected column data
        self.selCol = selCol

        # Get dict started
        self.expTable = dict()

        # Read in both observation files. Return as formatted pandas tables.
        # Data tables to be kept within class.
        for key, value in properties.items():
            # Read input file
            try:
                data = pandas.read_csv(pathcsv + value)

                # Write data into array
                self.expTable[key] = data.iloc[:, self.selCol].as_matrix()

                # Time should be the same in every file
                self.timeV = data.iloc[:, 1].as_matrix()
            except FileNotFoundError:
                continue

        # Sort the measurements by time
        IDXsort = np.argsort(self.timeV)
        self.timeV = self.timeV[IDXsort]

        # Apply sort to the other data
        for key, value in self.expTable.items():
            self.expTable[key] = value[IDXsort]

        # Parameter names
        self.pNames = ['a', 'b', 'c', 'd', 'e', 'conv_confl',
                       'conv_green', 'err_confl', 'err_green']

        # Specify lower bounds on parameters (log space)
        self.lb = np.full(len(self.pNames), -6.0)
        self.lb[-4:-2] = -2.0
        self.lb[0] = -5.0

        # Specify upper bounds on parameters (log space)
        self.ub = np.full(len(self.pNames), 0.0)
        self.ub[-4:-2] = 4.0

        # Set number of parameters
        self.Nparams = len(self.ub)
