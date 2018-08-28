"""
This module handles experimental data for drug interaction.
"""
import numpy as np
import pymc3 as pm
import theano
import theano.tensor as T
from .drugCombination import concentration_effects


class Eop(T.Op):
    ''' Theano Op for finding E with a drug interaction '''
    itypes = [T.dscalar] * 6
    otypes = [T.dvector]

    def __init__(self, X1, X2):
        assert(X1.shape == X2.shape)
        self.X1 = X1
        self.X2 = X2

    def infer_shape(self, node, i0_shapes):
        assert(len(i0_shapes) == 6)
        return [(self.X1.shape, )]

    def perform(self, node, inputs, outputs):
        outt = concentration_effects([inputs[0], inputs[1]],
                                     [inputs[2], inputs[3]],
                                     inputs[4], inputs[5],
                                     self.X1, self.X2)

        # Assert that the output we're passing back is the right shape
        assert(outt.shape == self.infer_shape(node, inputs)[0][0])

        outputs[0][0] = np.array(outt)

    def grad(self, inputs, g):
        """ Calculate the runCkineOp gradient. """
        raise NotImplementedError('This Op does not yet support a gradient.')


def build_model(X1, X2, timeV, conv0=None, expTable=None):
    ''' Builds then returns the pyMC model. '''

    assert(X1.shape == X2.shape)

    growth_model = pm.Model()

    with growth_model:
        # Rate of moving from apoptosis to death, assumed invariant wrt. treatment
        d = pm.Lognormal('d', np.log(0.01), 1)

        # a death/growth values
        a = pm.Normal('a', 0., 0.5, shape=2)

        # hill coefs for drug 1, 2; first death then growth
        hill = -pm.Lognormal('hill', -1., 0.1, shape=2)

        # IL50 for drug 1, 2; first death then growth
        IC50 = pm.Lognormal('IC50_drug1_death', -1., 0.1, shape=2)

        # E_con values
        E_con_death = pm.Lognormal('E_con_death', 0., 0.1)
        E_con_growth = pm.Lognormal('E_con_growth', 0., 0.1)

        # Fraction of dying cells that go through apoptosis
        apopfrac = pm.Beta('apopfrac', 2., 2.)

        # Make theano Op for calculating drug combination values
        combinationOp = Eop(X1, X2)

        # Calculate the death rate
        death_rates = combinationOp(IC50[0], hill[0], IC50[1],
                                    hill[1], a[0], E_con_death)

        # Calculate the growth rate
        growth_rates = combinationOp(IC50[0], hill[0], IC50[1],
                                     hill[1], a[1], E_con_growth)

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

        # Test the size of lnum
        lnum = T.opt.Assert('lnum did not match X1*timeV size')(lnum, T.eq(lnum.size, X1.size*timeV.size))

        # Number of early apoptosis cells at start is 0.0
        eap = cGRd * (lnum - pm.math.exp(-d * timeV))

        # Calculate dead cells via apoptosis and via necrosis
        deadnec = b * (lnum - 1) / GR
        deadapop = d * cGRd * (lnum - 1) / GR + cGRd * (pm.math.exp(-d * timeV) - 1)

        if expTable is not None: # Only try and fit to data if we get a time vector
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

            # Convert model calculations to experimental measurement units
            confl_exp = (lnum + eap + deadapop + deadnec) * confl_conv
            apop_exp = (eap + deadapop) * apop_conv + apop_offset
            dna_exp = (deadapop + deadnec) * dna_conv + dna_offset

            # Filler just to use variables
            pm.Normal('death_fit', 1., 0.5, observed=T.flatten(dna_exp))
            pm.Normal('growth_fit', 1., 0.5, observed=T.flatten(confl_exp))
            pm.Normal('apop_fit', 1., 0.5, observed=T.flatten(apop_exp))

        pm.Deterministic('logp', growth_model.logpt)

    return growth_model
