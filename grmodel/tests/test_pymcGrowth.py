import unittest
import pymc3 as pm
from ..pymcGrowth import GrowthModel


class TestgrMethods(unittest.TestCase):
    def test_model(self):
        GR = GrowthModel()

        GR.importData(2)

        model = GR.build_model()

        self.assertEqual(len(GR.expTable), 3)
        self.assertIsInstance(model, pm.Model)

    def test_MAP(self):
        GR = GrowthModel()

        GR.importData(2)

        with GR.model:
            start, nuts = pm.sampling.init_nuts(n_init=10,
                                                progressbar=False)

        self.assertEqual(len(start), 9)
