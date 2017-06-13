import numpy as np

np.seterr(over='raise')


def rate_values(parameters, time):
    '''Calculates the rates of each cellular process at a given time

    Returns a list of values of e raised to the power of teh calculated rate
    (for each set of parameters) as to ensure no negative value. The input is
    a list of lists, each index of the inner list a coeficient of time to the
    power of that index. The outer list represents the number of functions.

    Arguments:
        parameters (list of lists): contains number of functions and
            parameters of each function
        time (float): time at which rate is calculated

    Returns:
        list of the rates for each each function
    '''

    if time < 0:
        raise ValueError

    tt = np.power(time,
                  np.arange(parameters.shape[0], 0, -1, dtype=np.float64)).T

    return np.exp(np.matmul(parameters.T, tt))


def logpdf_sum(x, loc, scale):
    """
    Calculate the likelihood of an observation applied to a
    normal distribution.
    """
    prefactor = - np.log(scale * np.sqrt(2 * np.pi))
    summand = -np.square((x - loc) / (np.sqrt(2) * scale))
    return prefactor + summand


def ODEfun(state, t, params):
    """
    ODE function of cells living/dying/undergoing early apoptosis

    params:
    state   the number of cells in a particular state
            (LIVE, DEAD, EARLY_APOPTOSIS)
    t   time
    a 	parameter between LIVE -> LIVE (cell division)
    b 	parameter between LIVE -> DEAD
    c 	parameter between LIVE -> EARLY_APOPTOSIS
    d 	parameter between EARLY_APOPTOSIS -> DEATH
    e 	parameter between DEATH -> GONE
    """

    # If we don't have the right number of parameters, then panic
    if params.shape[1] != 5:
        raise ValueError

    LIVE, DEAD, EARLY_APOPTOSIS = state[0], state[1], state[2]

    rates = rate_values(params, t)

    return np.array([(rates[0] - rates[1] - rates[2]) * LIVE,
                     rates[1] * LIVE - rates[4] * DEAD + rates[3] * EARLY_APOPTOSIS,
                     rates[2] * LIVE - rates[3] * EARLY_APOPTOSIS], dtype=np.float64)


def mcFormat(mcParams):
    """ takes in mc data of list and returns equal length list of lists """
    if len(mcParams) % 5 > 0:
        raise ValueError("Length of mcParams must be a multiple of 5.")

    return np.reshape(mcParams, (-1, 5), order='F')


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

    out, infodict = odeint(ODEfun,
                           [1.0, 0.0, 0.0],
                           t_interval,
                           args=(params,),
                           full_output=True,
                           printmessg=False,
                           mxstep=2000)

    if infodict['message'] != 'Integration successful.':
        raise FloatingPointError(infodict['message'])

    return (t_interval, out)


def paramsWithinLimits(params, t_int, maxVal):
    """
    This only checks that the parameters don't pass through
    the bound in the duration. It's still possible they start
    too high, which is handled by the lower and upper
    bounds hopefully.
    """

    # Iterate over the parameters
    for ii in range(params.shape[1]):
        # Copy so we don't mess with the original
        pval = params[:, ii].copy()

        # Move by offset so roots tell us when we pass over limit
        pval[-1] -= maxVal

        # Find roots
        outt = np.roots(pval)

        # If any of the roots fall within the time interval and are real, fail
        for jj in outt:
            if np.isreal(jj) and jj > t_int[0] and jj < t_int[1]:
                return False

    return True


class GrowthModel:

    def likelihood(self, confl_model, green_model, sigma_sim_live, sigma_sim_dead):
        """
        Calculates the log likelihood of a simulation given the experimental data.

        parameters:
        table_sim   merged table with the transformed simulation predictions
        sigma_sim_live  the sigma used for the normal dist. for live cells
        sigma_sim_dead  the sigma used for the normal dist. for dead cells
        """

        try:
            logSqrErr = np.sum(logpdf_sum(self.expTable[1], loc=confl_model, scale=sigma_sim_live))
            logSqrErr += np.sum(logpdf_sum(self.expTable[2], loc=green_model, scale=sigma_sim_dead))
        except FloatingPointError:
            return -np.inf

        return logSqrErr

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

        # Format parameters to list of lists (except last 4 entries)
        params = mcFormat(paramV[:-4])

        # Check that the parameter values are reasonable over the interval
        if not paramsWithinLimits(params, self.tRange, 2.0):
            return -np.inf

        # Calculate model data table
        try:
            model = simulate(params, self.uniqueT)
        except FloatingPointError:
            return -np.inf

        # Power transform conversion constants
        paramV[-4:] = np.power(10, paramV[-4:])

        # Scale model data table with conversion constants
        confl_mod = paramV[-4] * np.interp(self.expTable[0], model[0], np.sum(model[1], axis=1))
        green_mod = paramV[-3] * np.interp(self.expTable[0], model[0], model[1][:, 1] + model[1][:, 2])

        # Run likelihood function with modeled and experiemental data, with
        # standard deviation given by last two entries in paramV
        likel = self.likelihood(confl_mod, green_mod, paramV[-2], paramV[-1])

        if np.isnan(likel) is True:
            print('Got a NaN which shouldn\'t happen.')
            return -np.inf

        return likel

    def __init__(self, selCol, loadFile=None, complexity=2):
        """ Initialize class. """
        from os.path import join, dirname, abspath
        import itertools as itt
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
        ps = ['a', 'b', 'c', 'd', 'e']
        self.pNames = list(itt.chain.from_iterable(itt.repeat(x, complexity) for x in ps))
        self.pNames = self.pNames + ['conv_confl', 'conv_green', 'err_confl', 'err_green']

        # Specify lower bounds on parameters (log space)
        self.lb = np.full(len(self.pNames), -9, dtype=np.float64)

        # Specify upper bounds on parameters (log space)
        self.ub = np.full(len(self.pNames), 2.0, dtype=np.float64)
        self.ub[-4:-2] = 4

        # Set number of parameters
        self.Nparams = len(self.ub)

        # Save the specified complexity
        self.complexity = complexity
