import unittest
from ..pymcGrowth import GrowthModel


class TestgrMethods(unittest.TestCase):
    """ Tests the growth model methods. """

    def test_model(self):
        """ Test that we can build the growth model. """
        GrowthModel("030317-2-R1_H1299", comb="R")
