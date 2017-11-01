import unittest
import pymc3 as pm
from ..pymcGrowth import MultiSample, GrowthModel


class TestgrMethods(unittest.TestCase):
    def test_MultiSample(self):
        a = MultiSample()
        a.loadModels(2)
        
    def test_model(self):
        GR = GrowthModel(loadFile = "030317-2-R1_H1299")

        GR.importData(2, 'NVB', comb='R')

        model = GR.build_model()

        self.assertEqual(len(GR.expTable), 44)
        self.assertIsInstance(model, pm.Model)

    def test_MAP(self):
        GR = GrowthModel(loadFile = "030317-2-R1_H1299")

        GR.importData(2, 'NVB', comb='R')

        with GR.model:
            start, nuts = pm.sampling.init_nuts(n_init=10,
                                                progressbar=False)

        self.assertEqual(len(start), 52)
