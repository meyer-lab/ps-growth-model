"""
This module handles experimental data, by fitting a growth and death rate for each condition separately.
"""
from os.path import join, dirname, abspath
import pandas
import numpy as np
import pymc3 as pm
import theano.tensor as T


fitKwargs = {"tune": 3000, "progressbar": False, "target_accept": 0.9}


def theanoCore(timeV, div, deathRate, apopfrac, d):
    """ Assemble the core growth model. """
    # Make a vector of time and one for time-constant values
    timeV = T._shared(timeV)
    constV = T.ones_like(timeV)  # pylint: disable=no-member

    # Calculate the growth rate
    GR = T.outer(div - deathRate, constV)
    # cGDd is used later
    cGRd = T.outer(deathRate * apopfrac, constV) / (GR + d)

    # b is the rate straight to death
    b = T.outer(deathRate * (1 - apopfrac), constV)

    lnum = T.exp(GR * timeV)

    # Number of early apoptosis cells at start is 0.0
    eap = cGRd * (lnum - T.exp(-d * timeV))

    # Calculate dead cells via apoptosis and via necrosis
    deadnec = b * (lnum - 1) / GR
    deadapop = d * cGRd * (lnum - 1) / GR + cGRd * (T.exp(-d * timeV) - 1)

    return (lnum, eap, deadapop, deadnec)


def convSignal(lnum, eap, deadapop, deadnec, conversions):
    """ Sums up the cell populations to link number of cells to image area. """
    conv, offset = conversions
    confl_exp = (lnum + eap + deadapop + deadnec) * conv[0]
    apop_exp = (eap + deadapop) * conv[1] + offset[0]
    dna_exp = (deadapop + deadnec) * conv[2] + offset[1]

    return (confl_exp, apop_exp, dna_exp)


def conversionPriors(conv0):
    """ Sets the various fluorescence conversion priors. """
    # Set up conversion rates
    confl_conv = pm.Lognormal("confl_conv", np.log(conv0), 0.1)
    apop_conv = pm.Lognormal("apop_conv", np.log(conv0) - 2.06, 0.1)
    dna_conv = pm.Lognormal("dna_conv", np.log(conv0) - 1.85, 0.1)

    # Priors on conv factors
    pm.Lognormal("confl_apop", -2.06, 0.0647, observed=apop_conv / confl_conv)
    pm.Lognormal("confl_dna", -1.85, 0.125, observed=dna_conv / confl_conv)
    pm.Lognormal("apop_dna", 0.222, 0.141, observed=dna_conv / apop_conv)

    # Offset values for apop and dna
    apop_offset = pm.Lognormal("apop_offset", np.log(0.1), 0.1)
    dna_offset = pm.Lognormal("dna_offset", np.log(0.1), 0.1)
    return ((confl_conv, apop_conv, dna_conv), (apop_offset, dna_offset))


def deathPriors(numApop):
    """ Setup priors for cell death parameters. """
    # Rate of moving from apoptosis to death, assumed invariant wrt. treatment
    d = pm.Lognormal("d", np.log(0.001), 0.5)

    # Fraction of dying cells that go through apoptosis
    apopfrac = pm.Beta("apopfrac", 1.0, 1.0, shape=numApop)

    return d, apopfrac


def build_model(conv0, doses, timeV, expTable):
    """ Builds then returns the pyMC model. """
    growth_model = pm.Model()

    with growth_model:
        conversions = conversionPriors(conv0)
        d, apopfrac = deathPriors(len(doses))

        # Specify vectors of prior distributions
        # Growth rate
        div = pm.Uniform("div", lower=0.0, upper=0.035, shape=len(doses))

        # Rate of entering apoptosis or skipping straight to death
        deathRate = pm.Lognormal("deathRate", np.log(0.001), 0.5, shape=len(doses))

        lnum, eap, deadapop, deadnec = theanoCore(timeV, div, deathRate, apopfrac, d)

        # Convert model calculations to experimental measurement units
        confl_exp, apop_exp, dna_exp = convSignal(lnum, eap, deadapop, deadnec, conversions)

        # Fit model to confl, apop, dna, and overlap measurements
        if "confl" in expTable.keys():
            # Observed error values for confl
            confl_obs = T.reshape(confl_exp, (-1,)) - expTable["confl"]

            pm.Normal("dataFit", sd=T.std(confl_obs), observed=confl_obs)
        if "apop" in expTable.keys():
            # Observed error values for apop
            apop_obs = T.reshape(apop_exp, (-1,)) - expTable["apop"]

            pm.Normal("dataFita", sd=T.std(apop_obs), observed=apop_obs)
        if "dna" in expTable.keys():
            # Observed error values for dna
            dna_obs = T.reshape(dna_exp, (-1,)) - expTable["dna"]

            pm.Normal("dataFitd", sd=T.std(dna_obs), observed=dna_obs)

    return growth_model


class GrowthModel:
    """ Model for fitting data incorporating cell death response. """

    def performFit(self):
        """ Run NUTS sampling"""
        model = build_model(self.conv0, self.doses, self.timeV, self.expTable)

        samples = pm.sample(model=model, **fitKwargs)
        self.df = pm.backends.tracetab.trace_to_dataframe(samples)

    def __init__(self, loadFile, firstCols=2, comb=None, interval=True):
        """Import experimental data"""
        # Property list
        properties = {"confl": "_confluence_phase.csv", "apop": "_confluence_green.csv", "dna": "_confluence_red.csv"}

        # Find path for csv files in the repository.
        pathcsv = join(dirname(abspath(__file__)), "data/singles/" + loadFile)

        # Pull out selected column data
        self.doses = []
        self.drugs = []
        selconv0 = []

        # Get dict started
        self.expTable = dict()

        # Read in both observation files. Return as formatted pandas tables.
        # Data tables to be kept within class.
        for key, value in properties.items():
            # Read input file
            dataset = pandas.read_csv(pathcsv + value)
            # Subtract control
            dataset1 = dataset.iloc[:, 2: len(dataset.columns)]
            dataset1.sub(dataset1["Control"], axis=0)
            data = pandas.concat([dataset.iloc[:, 0:2], dataset1], axis=1, sort=False)

            # If interval=False, filter for endpoint data
            if not interval:
                # Keep data within an hour of the beginning or end
                data = data.loc[(data["Elapsed"] < 1.0) | (max(data["Elapsed"]) - data["Elapsed"] < 1.0)]

            # Get phase confl was t=0 for confl_conv calculation
            if key == "confl":
                conv0 = np.mean(data.loc[data["Elapsed"] == 0].iloc[:, firstCols:])

            # Set the time vector
            self.timeV = data.iloc[:, 1].values

            assert len(data.columns) > firstCols + 1
            for col in range(firstCols, len(data.columns)):
                # Set the name of the condition we're considering
                condName = data.columns.values[col]

                # For data with combination therapies
                if comb is not None:
                    # Represent dose with a tuple of len(2) in each case
                    # If control
                    if "Control" in condName:
                        drug = "Control"
                        dose = (0, 0)
                    # If only the combination drug
                    elif condName.split(" ")[0] == comb:
                        drug = comb
                        dose = (0, float(condName.split(" ")[1]))
                    # If contains drug besides the combination drug
                    elif "blank" not in condName.lower():
                        try:  # Both combination drug and another drug
                            drug1str = condName.split(", ")[0]
                            combstr = condName.split(", ")[1]
                            dose = (float(drug1str.split(" ")[1]), float(combstr.split(" ")[1]))
                            drug = drug1str.split(" ")[0] + "+" + combstr.split(" ")[0]
                        except IndexError:  # Only the other drug
                            drug = condName.split(" ")[0]
                            dose = (condName.split(" ")[1], 0)

                    # Add data to expTable
                    self.expTable.setdefault(key, []).append(data.iloc[:, col].values)

                    # Append to class variables once per column of data
                    if key == "confl":
                        self.drugs.append(drug)
                        self.doses.append(dose)
                        selconv0.append(conv0[col - firstCols])

                else:  # For data without combinations
                    if "blank" not in condName.lower():
                        # Add the name of the condition we're considering
                        try:
                            drug = condName.split(" ")[0]
                            dose = condName.split(" ")[1]
                        except IndexError:
                            drug = "Control"
                            dose = 0

                        # Add data to expTable
                        self.expTable.setdefault(key, []).append(data.iloc[:, col].values)

                        # Append to class variables once per column of data
                        if key == "confl":
                            self.drugs.append(drug)
                            self.doses.append(dose)
                            selconv0.append(conv0[col - firstCols])
            # Reshape experimental data into 1D array
            self.expTable[key] = np.array(self.expTable[key]).reshape((-1,))

        # Record averge conv0 for confl prior
        self.conv0 = np.mean(selconv0)
