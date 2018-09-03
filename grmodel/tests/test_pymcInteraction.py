import unittest
import numpy as np
import pymc3 as pm
from theano.tests import unittest_tools as utt
from ..pymcInteraction import build_model, Eop


class TestInteractionMethods(unittest.TestCase):
    def test_InteractionModel(self):
        X1range = np.logspace(-1., 2., num=4)
        timeV = np.linspace(0., 72., num=10)

        M = build_model(X1range, X1range, timeV)

        pm.find_MAP(model=M, method='L-BFGS-B')

    @unittest.skip('Having an error with test values')
    def test_OpGradient(self):
        """ Verify the derivative passed back by Eop. """
        X1range = np.logspace(-1., 2., num=4)

        ptT = (np.array([1., 2.]), np.array([0.5, 1.]), np.array(-0.1), np.array(1.))

        utt.verify_grad(Eop(X1range, X1range), ptT)
