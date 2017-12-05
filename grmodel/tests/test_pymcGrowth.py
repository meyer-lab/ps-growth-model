import unittest
import pymc3 as pm
from ..pymcGrowth import MultiSample, GrowthModel


class TestgrMethods(unittest.TestCase):
    def test_MultiSample(self):
        a = MultiSample()
        a.loadModels(2)

    def test_model(self):
        GR = GrowthModel(loadFile="030317-2-R1_H1299")

        GR.importData(2)

        model = GR.build_model()

        self.assertEqual(len(GR.expTable), 4)
        self.assertIsInstance(model, pm.Model)

    def test_MAP(self):
        GR = GrowthModel(loadFile="030317-2-R1_H1299")

        GR.importData(2)

        with GR.model:
            start, nuts = pm.sampling.init_nuts(n_init=2,
                                                progressbar=False)

        self.assertEqual(len(start), 4)
