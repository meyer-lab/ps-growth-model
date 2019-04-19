import unittest
import numpy as np
from hypothesis import given
from hypothesis.strategies import floats
from ..drugCombination import concentration_effect, drugs

cRng = floats(0.1, 10)


class TestDrugCombination(unittest.TestCase):
    @given(IC1=cRng, IC2=cRng, X1=cRng, X2=cRng, hill1=floats(-4, -0.1),
           hill2=floats(-4, -0.1), a=floats(-1, 10))
    def test_concentration_effect(self, IC1, IC2, hill1, hill2, a, X1, X2):
        """ Test that we can successfully solve for drug interaction effects. """
        E_con = 1.0
        args = ([IC1, hill1], [IC2, hill2], a, E_con, X1, X2)

        E = concentration_effect(*args)

        soln = drugs(E, *args)

        # Test that we actually did find a solution
        if not np.isnan(E):
            self.assertAlmostEqual(soln, 0.0, 5, str(E))
