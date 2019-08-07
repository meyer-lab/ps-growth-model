import unittest
import pymc3 as pm
from ..pymcGrowth import GrowthModel, build_model


class TestgrMethods(unittest.TestCase):
    def test_model(self):
        GR = GrowthModel(loadFile="030317-2-R1_H1299")

        GR.importData(2, comb='R')

        #model = build_model(GR.conv0, GR.doses, GR.timeV, GR.expTable)

        #self.assertEqual(len(GR.expTable), 3)
        #self.assertIsInstance(model, pm.Model)
