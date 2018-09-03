"""
This module handles experimental data for drug interaction.
"""
import numpy as np
import pymc3 as pm
import theano
import theano.tensor as T
from sympy import symbols, solve, Function, diff, Derivative, ImmutableMatrix
from sympy.matrices import matrix_multiply_elementwise
from .drugCombination import concentration_effects


class Eop(T.Op):
    ''' Theano Op for finding E with a drug interaction '''
    itypes = [T.dvector, T.dvector, T.dscalar, T.dscalar]
    otypes = [T.dvector]

    def __init__(self, X1, X2):
        assert(X1.shape == X2.shape)
        self.X1 = X1
        self.X2 = X2
        self.EopGrad = EopGrad(self)

    def infer_shape(self, node, i0_shapes):
        assert(len(i0_shapes) == 4)
        assert(len(i0_shapes[0]) == 2)
        assert(len(i0_shapes[1]) == 2)
        return [(self.X1.shape, )]

    def calcConcEffects(self, inputs):
        outt = concentration_effects([inputs[0][0], inputs[1][0]],
                                     [inputs[0][1], inputs[1][1]],
                                     inputs[2], inputs[3],
                                     self.X1, self.X2)
        return outt

    def perform(self, node, inputs, outputs):
        outt = self.calcConcEffects(inputs)

        # Assert that the output we're passing back is the right shape
        assert(outt.shape == self.infer_shape(node, inputs)[0][0])

        outputs[0][0] = np.array(outt)

    def grad(self, inputs, g):
        """ Calculate the Eop gradient. """
        outputs = self.EopGrad(*inputs)

        return [outputs[0]*g, outputs[1]*g, outputs[2]*g, outputs[3]*g]


class EopGrad(T.Op):
    ''' Theano Grad Op for finding E with a drug interaction '''
    itypes = [T.dvector, T.dvector, T.dscalar, T.dscalar]
    otypes = [T.dmatrix, T.dmatrix, T.dvector, T.dvector]

    def __init__(self, parentOp):
        self.parentOp = parentOp

        Econ, IC501, IC502, m1, m2, a = symbols('Econ IC501 IC502 m1 m2 a', real=True)
        E = Function('Efunc')(Econ, IC501, IC502, m1, m2, a)
        self.x, self.Esym = symbols('x Esym')

        normE = E / (Econ - E)
        DD1 = ImmutableMatrix(self.parentOp.X1) / IC501
        DD2 = ImmutableMatrix(self.parentOp.X2) / IC502
        mm1 = 1.0 / m1
        mm2 = 1.0 / m2

        self.vars = (IC501, m1, IC502, m2, a, Econ)

        drugOne = DD1 * (normE**(-mm1))
        drugTwo = DD2 * (normE**(-mm2))
        drugInteract = a * matrix_multiply_elementwise(DD1, DD2) * (normE**(-(mm1 + mm2)/2.0))

        expression = drugOne + drugTwo + drugInteract
        expression = expression.applyfunc(lambda x: x - 1)

        # Generate expressions for the derivatives
        self.derivs = list()

        for ii, var in enumerate(self.vars):
            exprD = diff(expression, var)
            exprD = exprD.subs(Derivative(E, var), self.x)
            exprD = exprD.subs(E, self.Esym)
            self.derivs.append(exprD)

    def perform(self, node, inputs, outputs):
        """ Calculate EopGrad. """

        # Fill in most of the input values
        derivs = self.derivs.copy()

        for jj in range(len(derivs)):
            derivs[jj] = derivs[jj].subs(self.vars[0], inputs[0][0])
            derivs[jj] = derivs[jj].subs(self.vars[1], inputs[1][0])
            derivs[jj] = derivs[jj].subs(self.vars[2], inputs[0][1])
            derivs[jj] = derivs[jj].subs(self.vars[3], inputs[1][1])
            derivs[jj] = derivs[jj].subs(self.vars[4], inputs[2])
            derivs[jj] = derivs[jj].subs(self.vars[5], inputs[3])

        # Calculate E
        outt = self.calcConcEffects(inputs)
        assert(self.parentOp.X1.shape == outt.shape)

        # Plug in E and solve
        grad = np.zeros((self.parentOp.X1.size, len(self.derivs)))

        for ii in np.ndindex(*grad.shape):
            expr = derivs[ii[1]][ii[0]].subs(self.Esym, outt[ii[0]])
            grad[ii] = solve(expr, self.x)[0]

        outputs[0][0] = grad[:, np.array([0, 2])]
        outputs[1][0] = grad[:, np.array([1, 3])]
        outputs[2][0] = grad[:, 4]
        outputs[3][0] = grad[:, 5]

    def grad(self, inputs, g):
        """ Calculate the EopGrad gradient. """
        raise NotImplementedError('This Op does not support a gradient.')


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
        death_rates = combinationOp(IC50, hill, a[0], E_con_death)

        # Calculate the growth rate
        growth_rates = combinationOp(IC50, hill, a[1], E_con_growth)

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
