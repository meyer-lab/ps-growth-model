import unittest
from hypothesis import given
from hypothesis.strategies import floats
from ..drugCombination import concentration_effect, drug, drugs


class TestDrugCombination(unittest.TestCase):
    @given(IC1=floats(0.1, 0.9), IC2=floats(0.1, 0.9),
           hill1=floats(-4, -0.1), hill2=floats(-4, -0.1),
           a=floats(0.1, 0.9), X1=floats(0.1, 0.9), X2=floats(0.1, 0.9))
    def test_concentration_effect(self, IC1, IC2, hill1, hill2, a, X1, X2):
        """ Test that we can successfully solve for drug interaction effects. """
        E_con = 1.0

        E = concentration_effect([IC1, hill1], [IC2, hill2], a, E_con, X1, X2)

        soln = drug([IC1, hill1], X1, E, E_con) + drug([IC2, hill2], X2, E, E_con) + drugs([IC1, hill1], [IC2, hill2], a, E_con, X1, X2, E)

        # Test that we actually did find a solution
        self.assertAlmostEqual(soln, 1.0)
