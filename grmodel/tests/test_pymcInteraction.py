import unittest
import numpy as np
import pymc3 as pm
from ..pymcInteraction import build_model


class TestInteractionMethods(unittest.TestCase):
    """ Tests the interaction model methods. """

    def test_InteractionModel(self):
        """ This tests that we can build the interaction model. """
        X1range = np.logspace(-1.0, 2.0, num=4)
        timeV = np.linspace(0.0, 72.0, num=10)

        M = build_model(X1range, X1range, timeV)

        pm.find_MAP(model=M, method="L-BFGS-B")
