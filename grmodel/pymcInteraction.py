"""
This module handles experimental data for drug interaction.
"""
import numpy as np
import pymc3 as pm
import theano
import theano.tensor as T


def build_model(X1, X2, timeV, conv0=0.1, confl=None, apop=None, dna=None):
    ''' Builds then returns the pyMC model. '''

    assert(X1.shape == X2.shape)

    growth_model = pm.Model()

    with growth_model:
        # Rate of moving from apoptosis to death, assumed invariant wrt. treatment
        d = pm.Lognormal('d', np.log(0.01), 1)

        # hill coefs for drug 1, 2; first death then growth
        hill = pm.Lognormal('hill', 0.0, 0.1, shape=2)

        # IL50 for drug 1, 2; first death then growth
        IC50 = pm.Lognormal('IC50_drug1_death', -1., 0.1, shape=2)

        # E_con values; first death then growth
        E_con = pm.Lognormal('E_con', -1.0, 0.1, shape=2)

        # Fraction of dying cells that go through apoptosis
        apopfrac = pm.Beta('apopfrac', 2., 2.)

        # Calculate the death rate, which should be flipped
        death_drug_one = T.pow(X1, hill[0]) / (T.pow(IC50[0], hill[0]) + T.pow(X1, hill[0]))
        death_drug_two = T.pow(X2, hill[0]) / (T.pow(IC50[0], hill[0]) + T.pow(X1, hill[0]))
        death_drug_comb = death_drug_one + death_drug_two - death_drug_one*death_drug_two
        death_rates = E_con[0] * death_drug_comb

        # Calculate the growth rate
        growth_drug_one = T.pow(X1, hill[1]) / (T.pow(IC50[1], hill[1]) + T.pow(X1, hill[1]))
        growth_drug_two = T.pow(X2, hill[1]) / (T.pow(IC50[1], hill[1]) + T.pow(X1, hill[1]))
        growth_drug_comb = growth_drug_one + growth_drug_two - growth_drug_one*growth_drug_two
        growth_rates = E_con[1] * (1 - growth_drug_comb)

        # Test the dimension of growth_rates
        growth_rates = T.opt.Assert('growth_rates did not match X1 size')(growth_rates, T.eq(growth_rates.size, X1.size))

        # Make a vector of time and one for time-constant values
        timeV = T._shared(timeV)
        constV = T.ones_like(timeV, dtype=theano.config.floatX)

        # Calculate the growth rate
        GR = T.outer(growth_rates - death_rates, constV)

        # cGDd is used later
        cGRd = T.outer(death_rates * apopfrac, constV) / (GR + d)

        # b is the rate straight to death
        b = T.outer(death_rates * (1 - apopfrac), constV)

        # Calculate the number of live cells
        lnum = T.exp(GR * timeV)

        # Just here to ensure combinationOp isn't optimized out
        pm.Normal('lnum', 3.0, 10.0, observed=lnum[0][-1])

        # Test the size of lnum
        lnum = T.opt.Assert('lnum did not match X1*timeV size')(lnum, T.eq(lnum.size, X1.size*timeV.size))

        # Number of early apoptosis cells at start is 0.0
        eap = cGRd * (lnum - pm.math.exp(-d * timeV))

        # Calculate dead cells via apoptosis and via necrosis
        deadnec = b * (lnum - 1) / GR
        deadapop = d * cGRd * (lnum - 1) / GR + cGRd * (pm.math.exp(-d * timeV) - 1)

        # Set up conversion rates
        confl_conv = pm.Lognormal('confl_conv', np.log(conv0), 0.1)
        apop_conv = pm.Lognormal('apop_conv', np.log(conv0) - 2.06, 0.2)
        dna_conv = pm.Lognormal('dna_conv', np.log(conv0) - 1.85, 0.2)

        # Priors on conv factors
        pm.Lognormal('confl_apop', -2.06, 0.0647, observed=apop_conv / confl_conv)
        pm.Lognormal('confl_dna', -1.85, 0.125, observed=dna_conv / confl_conv)
        pm.Lognormal('apop_dna', 0.222, 0.141, observed=dna_conv / apop_conv)

        # Offset values for apop and dna
        apop_offset = pm.Lognormal('apop_offset', -1., 0.1)
        dna_offset = pm.Lognormal('dna_offset', -1., 0.1)

        # Compare to experimental observation
        if confl is not None:
            confl_exp = (lnum + eap + deadapop + deadnec) * confl_conv
            confl_obs = T.flatten(confl_exp - confl)
            pm.Normal('confl_fit', sd=T.std(confl_obs), observed=confl_obs)

        if apop is not None:
            apop_exp = (eap + deadapop) * apop_conv + apop_offset
            apop_obs = T.flatten(apop_exp - apop)
            pm.Normal('apop_fit', sd=T.std(apop_obs), observed=apop_obs)

        if dna is not None:
            dna_exp = (deadapop + deadnec) * dna_conv + dna_offset
            dna_obs = T.flatten(dna_exp - dna)
            pm.Normal('dna_fit', sd=T.std(dna_obs), observed=dna_obs)

        pm.Deterministic('logp', growth_model.logpt)

    return growth_model
