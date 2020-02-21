"""
Dose response analysis to assess the uncertainty that exists when one only uses the live cell number.
"""
from os.path import join, dirname, abspath
import numpy as np
import pymc3 as pm
import theano.tensor as T
import pandas as pd


class doseResponseModel:
    """ pymc3 model of just using the live cell number. """

    def build_model(self):
        """ Builds then returns the pyMC model. """
        M = pm.Model()

        with M:
            # The three values here are div and deathrate
            # Assume just one IC50 for simplicity
            lIC50 = pm.Normal("IC50s", 2.0)

            Emin_growth = pm.Uniform("Emin_growth", lower=0.0, upper=self.Emax_growth)
            Emax_death = pm.Lognormal("Emax_death", -2.0, 2.0)

            # Import drug concentrations into theano vector
            drugCs = T._shared(self.drugCs)

            # Drug term since we're using constant IC50 and hill slope
            drugTerm = 1.0 / (1.0 + T.pow(10.0, (lIC50 - drugCs) * pm.Lognormal("hill", 1.0)))

            # Do actual conversion to parameters for each drug condition
            growthV = self.Emax_growth + (Emin_growth - self.Emax_growth) * drugTerm

            # Calculate the growth rate
            # _Assuming deathrate in the absence of drug is zero
            GR = growthV - Emax_death * drugTerm

            # Calculate the number of live cells
            lnum = T.exp(GR * self.time)

            # Normalize live cell data to control, as is similar to measurements
            # Residual between model prediction and measurement
            residual = self.lObs - (lnum / lnum[0])

            pm.Normal("dataFitlnum", sd=T.std(residual), observed=residual)

        return M

    def __init__(self, Drug):
        """ Load data and setup. """
        filename = join(dirname(abspath(__file__)), "data/initial-data/2017.07.10-H1299-celltiter.csv")
        data = pd.read_csv(filename)

        # Response should be normalized to the control
        data["response"] = data["CellTiter"] / np.mean(data.loc[data["Conc (nM)"] == 0.0, "CellTiter"])

        # Put the dose on a log scale as well
        data["logDose"] = np.log10(data["Conc (nM)"] + 0.1)

        dataLoad = data[data["Drug"] == Drug]

        # Handle data import here
        self.drugCs = dataLoad["logDose"].values
        self.time = 72.0

        # Based on control kinetic data
        self.Emax_growth = 0.0315

        self.lObs = dataLoad["response"].values

        # Build the model
        self.model = self.build_model()
        self.trace = pm.sample(progressbar=False, chains=2, target_accept=0.9, model=self.model)
