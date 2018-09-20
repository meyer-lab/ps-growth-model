import unittest
import numpy as np
import pymc3 as pm
from ..pymcInteraction import build_model


class TestInteractionMethods(unittest.TestCase):
    def test_InteractionModel(self):
        X1range = np.logspace(-1., 2., num=4)
        timeV = np.linspace(0., 72., num=10)

        M = build_model(X1range, X1range, timeV)

        pm.find_MAP(model=M, method='L-BFGS-B')
