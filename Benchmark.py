#! /usr/bin/env python3

from emcee import EnsembleSampler
from grmodel import GrowthModel
from grmodel.fitFuncs import getUniformStart

# Load model
grM = GrowthModel.GrowthModel(4)

# Get uniform distribution of positions for start
p0, ndims, nwalkers = getUniformStart(grM)

# Set up sampler and sample
EnsembleSampler(nwalkers, ndims, grM.logL).run_mcmc(p0, N=1E3)
